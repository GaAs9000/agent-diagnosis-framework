"""Single-episode lifecycle. Does not contain benchmark or provider-specific rules.

Every external operation is preceded by an fsync'ed event. Recoverable exceptions
retain a finalized evidence package; abrupt process death leaves the valid prefix.
The production registries contain no fixture backend or unrestricted host runtime.
"""
from __future__ import annotations
import copy
import json
import math
from dataclasses import replace
import os
from pathlib import Path
import re
import shutil
import signal
import time
import uuid
from .contracts import VERSION, Result, NativeValidatedCall, HTTPResult, ProtocolError, EvidenceError, ConfigurationError, EvaluationSpec, L1Error
from .config import validate
from ..storage import canonical, digest, file_digest, regular_file, read_json, write_json, write_bytes, confined, utc_now, copy_file
from ..recording.state import StateStore, restore
from ..evidence.live import Journal, seal, verify, validate_snapshot
from ..runtime.tools import PI_TOOLS, TOOLS, active_harness, tools_for_config, validate_call, _tool_error_result

OUTPUT_LIMIT_FEEDBACK = (
    'Model output reached its token limit. The response may be incomplete; '
    'no tool from that response was executed. Submit smaller, complete tool calls.'
)

SYSTEM = '''You are the tested agent in an execution-and-evidence sandbox. Complete the user's task using the supplied native tools. Original input files are in /workspace/input. Put ONLY final deliverables in /workspace/submission. Preserve required filenames unless the task says otherwise. Do not put reference inputs into submission unless requested. You may use /workspace for scripts and intermediate files. Shell calls run in fresh Linux containers without network. Only /workspace persists; working directory, environment variables, /tmp and background processes do not persist. Do not claim to have run a tool in text: use actual native tool calls. No scoring rubrics or grader are accessible to you.'''

PI_SYSTEM = '''You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
- read: Read file contents
- bash: Execute bash commands (ls, grep, find, etc.)
- edit: Make precise file edits with exact text replacement, including multiple disjoint edits in one call
- write: Create or overwrite files

In addition to the tools above, you may have access to other custom tools depending on the project.

Guidelines:
- Use bash for file operations like ls, rg, find
- Use read to examine files instead of cat or sed
- Use edit for precise changes (edits[].oldText must match exactly)
- When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls
- Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.
- Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.
- Use write only for new files or complete rewrites
- Be concise in your responses
- Show file paths clearly when working with files'''


def _inject_user_intervention(conversation: dict, message: str) -> dict:
    new = copy.deepcopy(conversation)
    if 'messages' in new and isinstance(new['messages'], list):
        new['messages'].append({'role': 'user', 'content': message})
    elif 'input' in new and isinstance(new['input'], list):
        new['input'].append({'role': 'user', 'content': message})
    elif 'contents' in new and isinstance(new['contents'], list):
        new['contents'].append({'role': 'user', 'parts': [{'text': message}]})
    return new


def _estimate_message_tokens(msg: dict) -> int:
    """Conservative chars/4 token estimator matching pi-coding-agent@0.73.1 compaction.js."""
    chars = 0
    content = msg.get('content')
    if isinstance(content, str):
        chars += len(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get('type') == 'text' and isinstance(block.get('text'), str):
                    chars += len(block['text'])
                elif block.get('type') in ('image', 'image_url'):
                    chars += 4800
    for tc in msg.get('tool_calls') or []:
        if isinstance(tc, dict):
            fn = tc.get('function') or {}
            chars += len(str(fn.get('name') or '')) + len(str(fn.get('arguments') or ''))
    return max(1, (chars + 3) // 4)


class _PiContextUnsatisfiable(L1Error):
    code = 'pi_context_unsatisfiable'


def _maybe_compact_pi_conversation(
    conversation: dict,
    comp_cfg: dict | None,
    last_usage_tokens: int | None,
    request_count: int,
    *,
    usage_messages: list[dict] | None = None,
    usage_kind: str = 'total_tokens',
) -> tuple[dict, dict | None]:
    """Deterministic Pi context compaction preserving system prompt, initial task prompt,
    file operation tracking (readFiles/modifiedFiles), and recent turn boundaries.
    """
    if not isinstance(comp_cfg, dict) or not comp_cfg.get('enabled', False):
        return conversation, None
    messages = conversation.get('messages')
    if not isinstance(messages, list):
        return conversation, None

    context_window = int(comp_cfg.get('context_window_tokens', 131072))
    reserve_tokens = int(comp_cfg.get('reserve_tokens', 16384))
    keep_recent_tokens = int(comp_cfg.get('keep_recent_tokens', 20000))

    # Bind usage to the history actually sent, not to the newly accepted tool
    # results. Keep a high-water calibration across compactions/cold restores:
    # chars/4 and Pi's 1200-token image estimate can greatly undercount vision.
    calibration = copy.deepcopy(conversation.get('_pi_context_usage') or {})
    scale = max(1.0, calibration.get('estimate_multiplier', 1.0))
    image_extra = max(0, calibration.get('extra_tokens_per_image', 0))
    def image_count(history):
        return sum(
            block.get('type') in ('image', 'image_url')
            for message in history if isinstance(message.get('content'), list)
            for block in message['content'] if isinstance(block, dict)
        )
    reported = last_usage_tokens if type(last_usage_tokens) is int and last_usage_tokens > 0 else 0
    source_messages = usage_messages if usage_messages is not None else messages
    source_estimate = sum(_estimate_message_tokens(m) for m in source_messages)
    if reported and request_count > calibration.get('at_request', -1):
        # Ten percent headroom on an observed underestimate; this is an
        # empirical upper estimate, not a claim to reproduce the tokenizer.
        if reported > source_estimate:
            scale = max(scale, reported * 1.1 / max(1, source_estimate))
            images = image_count(source_messages)
            if images:
                # Also attribute the full unexplained excess to images. A
                # global ratio alone loses its protection when compaction
                # discards much text but retains the expensive image blocks.
                image_extra = max(image_extra, math.ceil((reported * 1.1 - source_estimate) / images))
        calibration = {
            'at_request': request_count,
            'reported_tokens': reported,
            'reported_usage_kind': usage_kind,
            'source_messages_sha256': digest(source_messages),
            'source_message_count': len(source_messages),
            'source_tokens_estimate': source_estimate,
            'estimate_multiplier': scale,
            'extra_tokens_per_image': image_extra,
        }
    def estimate(history):
        base = sum(_estimate_message_tokens(m) for m in history)
        return max(math.ceil(base * scale), base + image_count(history) * image_extra)

    context_tokens = estimate(messages)
    new_conv = copy.deepcopy(conversation)
    if calibration:
        new_conv['_pi_context_usage'] = calibration
    if context_tokens <= context_window - reserve_tokens:
        return new_conv, None

    def unsatisfiable():
        raise _PiContextUnsatisfiable(
            f'Pi context cannot fit {context_window - reserve_tokens} tokens while preserving '
            f'system, initial task and latest complete turn (estimated {context_tokens}; '
            f'usage provenance: {json.dumps(calibration, sort_keys=True)}).'
        )

    # Find last assistant message index so we never cut inside the final observation turn
    last_assistant_idx = -1
    for idx in range(len(messages) - 1, 1, -1):
        if messages[idx].get('role') == 'assistant':
            last_assistant_idx = idx
            break
    if last_assistant_idx <= 2:
        unsatisfiable()

    # Walk backwards from end to find cut point keeping ~keep_recent_tokens
    # Never cut at a 'tool' message (must keep assistant + its tool results together)
    valid_cuts = []
    pending_ids = set()
    for idx, message in enumerate(messages):
        if idx > 2 and idx <= last_assistant_idx and not pending_ids and message.get('role') in ('user', 'assistant'):
            valid_cuts.append(idx)
        if message.get('role') == 'assistant':
            pending_ids.update(call['id'] for call in message.get('tool_calls', []) if call.get('id'))
        elif message.get('role') == 'tool':
            pending_ids.discard(message.get('tool_call_id'))
    if pending_ids:
        unsatisfiable()
    if not valid_cuts:
        unsatisfiable()

    accumulated = 0
    cut_idx = valid_cuts[-1]
    for idx in range(len(messages) - 1, 1, -1):
        accumulated += math.ceil(_estimate_message_tokens(messages[idx]) * scale)
        if accumulated >= keep_recent_tokens:
            candidates = [c for c in valid_cuts if c <= idx]
            cut_idx = candidates[-1] if candidates else valid_cuts[-1]
            break

    if cut_idx <= 2:
        unsatisfiable()

    prev_records = list(conversation.get('_pi_compaction') or [])
    # Shorten only at complete turns until the entire retained history fits.
    # Check every candidate: a summary's tracked-file list can itself grow.
    candidates = [candidate for candidate in valid_cuts if candidate >= cut_idx]
    for candidate in candidates:
        to_summarize = messages[2:candidate]
        if len(to_summarize) == 1 and str(to_summarize[0].get('content', '')).startswith('[PI CONTEXT COMPACTION SUMMARY'):
            continue
        new_messages, summary_msg, read_files, modified_files = _pi_compacted_messages(
            messages, candidate, prev_records, request_count, context_tokens,
            context_window - reserve_tokens,
        )
        tokens_after_est = estimate(new_messages)
        if tokens_after_est < context_tokens and tokens_after_est <= context_window - reserve_tokens:
            cut_idx = candidate
            break
    else:
        unsatisfiable()

    receipt = {
        'compaction_index': len(prev_records) + 1,
        'at_request': request_count,
        'tokens_before': context_tokens,
        'tokens_after_estimate': tokens_after_est,
        'context_window_tokens': context_window,
        'reserve_tokens': reserve_tokens,
        'keep_recent_tokens': keep_recent_tokens,
        'compacted_message_count': cut_idx - 2,
        'kept_message_count': len(new_messages),
        'cut_index': cut_idx,
        'read_files': sorted(read_files),
        'modified_files': sorted(modified_files),
        'summary_sha256': digest(summary_msg),
        'usage_calibration': calibration,
        'tokens_before_uncalibrated': sum(_estimate_message_tokens(m) for m in messages),
        'estimation_method': 'chars4_image1200_observed_ratio_and_image_excess_high_water_1.1',
    }
    new_conv['messages'] = new_messages
    new_conv['_pi_compaction'] = prev_records + [receipt]
    return new_conv, receipt


def _pi_compacted_messages(messages, cut_idx, prev_records, request_count, context_tokens, threshold):
    """Build the deterministic summary without altering native retained messages."""
    read_files: set[str] = set()
    modified_files: set[str] = set()
    for rec in prev_records:
        read_files.update(rec.get('read_files') or [])
        modified_files.update(rec.get('modified_files') or [])
    digest_lines: list[str] = []
    for m in messages[2:cut_idx]:
        role = m.get('role')
        if role == 'assistant':
            for tc in m.get('tool_calls') or []:
                fn = (tc.get('function') or {}) if isinstance(tc, dict) else {}
                name = str(fn.get('name') or '')
                raw_args = fn.get('arguments') or '{}'
                try:
                    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args if isinstance(raw_args, dict) else {})
                except Exception:
                    parsed_args = {}
                if not isinstance(parsed_args, dict):
                    parsed_args = {}
                path_arg = parsed_args.get('path')
                if name == 'read' and isinstance(path_arg, str):
                    read_files.add(path_arg)
                    digest_lines.append(f"- tool `read`: {path_arg}")
                elif name in ('write', 'edit') and isinstance(path_arg, str):
                    modified_files.add(path_arg)
                    digest_lines.append(f"- tool `{name}`: {path_arg}")
                elif name in ('bash', 'run'):
                    cmd_str = (str(parsed_args.get('command') or '').strip().splitlines() or [''])[0][:120]
                    digest_lines.append(f"- tool `{name}`: {cmd_str}")

    comp_index = len(prev_records) + 1
    summary_text = (
        f"[PI CONTEXT COMPACTION SUMMARY #{comp_index} at request {request_count}]\n"
        f"Context tokens before compaction: {context_tokens} (threshold: {threshold}).\n"
        f"Read files ({len(read_files)}): {', '.join(sorted(read_files)) if read_files else '(none)'}\n"
        f"Modified files ({len(modified_files)}): {', '.join(sorted(modified_files)) if modified_files else '(none)'}\n"
        f"Compacted actions ({len(digest_lines)}):\n"
        + ("\n".join(digest_lines[-20:]) if digest_lines else "- (no tool actions in compacted prefix)")
    )
    summary_msg = {'role': 'user', 'content': summary_text}
    new_messages = messages[:2] + [summary_msg] + messages[cut_idx:]
    return new_messages, summary_msg, read_files, modified_files


def build_execution_contract_prompt(config: dict) -> str:
    agent_cfg = config.get('agent', {})
    is_pi = active_harness(config) == 'pi'
    default_base = PI_SYSTEM if is_pi else SYSTEM
    base = agent_cfg.get('system_prompt') or default_base
    budget = config.get('budget', {})
    reqs = budget.get('max_model_requests', 20)
    tools = budget.get('max_tool_calls', 40)
    wall = budget.get('wall_time_seconds', 1200)
    tokens = budget.get('max_total_tokens')
    token_str = f"- Token Budget: At most {tokens} total tokens.\n" if tokens else ""

    # Clean any previously attached contract or legacy budget phrases to avoid stacking or mismatch
    clean_base = re.sub(r'\[SANDBOX EXECUTION CONTRACT & RUNTIME CONSTRAINTS\].*', '', base, flags=re.DOTALL)
    clean_base = re.sub(r'You operate under a strict (?:execution )?budget\b.*?\.(?:\s+|$)', '',
                        clean_base).rstrip()
    if not clean_base:
        clean_base = default_base.rstrip()

    if is_pi:
        contract_block = (
            "[SANDBOX EXECUTION CONTRACT & RUNTIME CONSTRAINTS]\n"
            f"- Model Request Budget: At most {reqs} model requests.\n"
            f"- Tool Execution Budget: At most {tools} tool invocations.\n"
            f"- Wall Clock Limit: {wall} seconds timeout.\n"
            f"{token_str}"
            "- Workspace Boundary: /workspace is persistent; create intermediate scripts and output files here. Original input files are in /workspace/input.\n"
            "- Final Submission: Place ONLY final deliverables in /workspace/submission with required filenames. Do not copy reference inputs into submission unless requested.\n"
            "- Stateless Tools: bash executions run in fresh, isolated Linux containers without network; env/tmp and background processes do not persist across calls.\n"
            "- Autonomous Verification: No grader or test rubric is accessible. Conclude promptly once deliverables are verified.\n\n"
            "Current working directory: /workspace"
        )
    else:
        contract_block = (
            "[SANDBOX EXECUTION CONTRACT & RUNTIME CONSTRAINTS]\n"
            f"- Model Request Budget: At most {reqs} model requests.\n"
            f"- Tool Execution Budget: At most {tools} tool invocations.\n"
            f"- Wall Clock Limit: {wall} seconds timeout.\n"
            f"{token_str}"
            "- Workspace Boundary: /workspace is persistent; create intermediate scripts and output files here.\n"
            "- Final Submission: Place ONLY final deliverables in /workspace/submission with required filenames.\n"
            "- Stateless Tools: Shell executions run in fresh, isolated Linux containers without network; env/tmp do not persist.\n"
            "- Autonomous Verification: No grader or test rubric is accessible. Conclude promptly once deliverables are verified."
        )
    prompt = f"{clean_base}\n\n{contract_block}"
    if re.search(r'\{(max_model_requests|max_tool_calls|wall_time_seconds|max_total_tokens)\}', prompt):
        raise ConfigurationError('Unformatted budget placeholder left in system prompt')
    return prompt


def components(config):
    from ..models.registry import registry as models
    from ..benchmarks.registry import registry as benchmarks
    from ..runtime.registry import registry as runtimes
    from ..evaluation.registry import registry as evaluators
    out={}
    for section,factory in [('model',models),('benchmark',benchmarks),('runtime',runtimes),('evaluation',evaluators)]:
        registry=factory()
        if config['plugins']['load_installed']: registry.load_installed()
        elif section=='model' and config['model']['adapter']=='opencode-chat':
            from ..models.opencode import OpenCodeChatBackend
            registry.register('opencode-chat',OpenCodeChatBackend)
        out[section]=registry.create(config[section]['adapter'],config[section])
    return out


def _copy_tree(source:Path,destination:Path):
    destination.mkdir(parents=True,exist_ok=False)
    for p in sorted(source.rglob('*')):
        if p.is_symlink(): raise EvidenceError('symlink in source evidence')
        if p.is_file(): copy_file(p,destination,p.relative_to(source).as_posix())


def _request_token_estimate(body: bytes, parameters: dict) -> float:
    """Admission heuristic: one token per UTF-8 byte, plus maximum output.

    Includes native tools/images in the serialized input. Without an explicit
    output cap reserve 65,536 output tokens; this is a heuristic, not tokenizer
    or provider-limit evidence. It never changes the native request parameters.
    """
    caps=[parameters.get(k) for k in ('maxOutputTokens','max_completion_tokens','max_tokens')]
    caps=[v for v in caps if type(v) in (int,float) and v>0]
    return float(len(body)+max(caps,default=65536))


def _receipt_tokens(receipt) -> float | None:
    """Use successful provider usage, including usage on malformed replies."""
    if receipt.status is None or not 200<=receipt.status<300 or receipt.metadata.get('transport_error'):
        return None
    usage=receipt.metadata.get('usage')
    if not isinstance(usage,dict):
        try:
            value=json.loads(receipt.body)
        except (ValueError,UnicodeError):
            return None
        if not isinstance(value,dict): return None
        usage=value.get('usageMetadata',value.get('usage'))
    if not isinstance(usage,dict): return None
    total=usage.get('totalTokenCount',usage.get('total_tokens'))
    return float(total) if type(total) is int and total>=0 else None


def digest_bytes(value: bytes) -> str:
    import hashlib
    return hashlib.sha256(value).hexdigest()


def _check_native_upgrade(previous: dict, current: dict) -> bool:
    """The only permitted harness transition for response consumption."""
    old = copy.deepcopy(previous)
    if active_harness(old) != 'pi' or old['agent'].get('version') != '0.73.1':
        return False
    previous_revision = old['agent'].get('harness_revision')
    previous_policy = old['agent'].get('validation_policy')
    target_revision = current['agent'].get('harness_revision')
    if (previous_policy, previous_revision) not in (
        (None, None), ('pi-native-0.73.1', 'l1-pi-native-errors-v1'),
    ):
        return False
    if target_revision not in ('l1-pi-native-errors-v1', 'l1-pi-native-errors-context-v2'):
        return False
    if previous_revision == 'l1-pi-native-errors-v1' and target_revision != 'l1-pi-native-errors-context-v2':
        return False
    if current['model']['timeout_seconds'] != old['model']['timeout_seconds']:
        if (old['model']['timeout_seconds'], current['model']['timeout_seconds']) != (300, 600):
            return False
        old['model']['timeout_seconds'] = 600
    old['agent'].update(validation_policy='pi-native-0.73.1', harness_revision=target_revision)
    return old == current


def _check_context_upgrade(previous, current):
    return (previous['agent'].get('harness_revision') == 'l1-pi-native-errors-v1'
            and current['agent'].get('harness_revision') == 'l1-pi-native-errors-context-v2'
            and previous['model']['timeout_seconds'] == current['model']['timeout_seconds'] == 600
            and bool((current['agent'].get('compaction') or {}).get('enabled'))
            and _check_native_upgrade(previous, current))


def _saved_response_source(parent: Path, manifest: dict, cp: dict, response_id: str) -> dict:
    """Bind a complete, rejected sole-edit response to its untouched prefix."""
    if manifest['recording_mode'] != 'live' or manifest['status']['termination'] != 'protocol_error':
        raise EvidenceError('Saved-response continuation requires sealed live protocol_error evidence')
    if not re.fullmatch(r'r\d{6}', response_id):
        raise ConfigurationError('resume_response_id must be r followed by six digits')
    if int(response_id[1:]) != cp['budget_used']['model_requests']:
        raise EvidenceError('Saved response must immediately follow the source checkpoint')
    events = [json.loads(line) for line in (parent/'trajectory.jsonl').read_text().splitlines()]
    requests = [e for e in events if e['kind'] == 'model.request']
    if not requests or requests[-1]['data']['request_id'].split(':')[-1] != response_id:
        raise EvidenceError('Only the final parent response can be consumed')
    request = requests[-1]
    rid = request['data']['request_id']
    responses = [e for e in events if e['kind'] == 'model.response' and e['data']['request_id'] == rid]
    calls = [e for e in events if e['kind'] == 'tool.call' and e['data']['request_id'] == rid]
    if len(responses) != 1 or len(calls) != 1 or calls[0]['data']['name'] != 'edit':
        raise EvidenceError('Saved response must contain a sole rejected edit call')
    cid = calls[0]['data']['call_id']
    if not any(e['kind'] == 'tool.rejected' and e['data']['call_id'] == cid for e in events):
        raise EvidenceError('Saved edit lacks a rejection receipt')
    if any(e['kind'] in ('tool.started', 'tool.result') and e['data'].get('call_id') == cid for e in events):
        raise EvidenceError('Saved edit already started; partial batches are unsupported')
    response = responses[0]
    metadata = read_json(parent/response['data']['http']['path'])
    if response['data']['status_code'] != 200 or metadata.get('body_truncated') or metadata.get('transport_error'):
        raise EvidenceError('Saved response must be a complete successful HTTP 200 receipt')
    snapshots = [e for e in events if e['kind'] == 'state.snapshot']
    final = read_json(parent/snapshots[-1]['data']['snapshot']['path'])
    source = read_json(parent/f"states/snapshots/{cp['snapshot_id']}.json")
    if source['content_sha256'] != final['content_sha256']:
        raise EvidenceError('Source checkpoint state differs from parent final state')
    return {'request': request, 'response': response, 'call': calls[0], 'metadata': metadata,
            'request_bytes': (parent/request['data']['body']['path']).read_bytes(),
            'response_bytes': (parent/response['data']['body']['path']).read_bytes()}


def _initial_upgrade_request(parent: Path, manifest: dict, cp: dict) -> bytes:
    if manifest['recording_mode'] != 'live' or cp['checkpoint_id'] != 'cp000000':
        raise EvidenceError('Native cold upgrade requires a live initial checkpoint')
    used = cp.get('budget_used', {})
    if any(used.get(key) != 0 for key in ('model_requests', 'tool_calls', 'known_total_tokens')):
        raise EvidenceError('Native cold upgrade requires zero consumed requests, tools and tokens')
    if cp.get('pending_calls') or cp.get('pending_observations'):
        raise EvidenceError('Native cold upgrade requires empty pending calls and observations')
    snap = read_json(parent/f"states/snapshots/{cp['snapshot_id']}.json")
    if snap.get('label') != 'initial':
        raise EvidenceError('Native cold upgrade requires original initial workspace state')
    history = [json.loads(line) for line in (parent/'trajectory.jsonl').read_text().splitlines()]
    initial = next((e for e in history if e['kind'] == 'state.snapshot'), None)
    if initial is None or initial['data']['snapshot_id'] != cp['snapshot_id']:
        raise EvidenceError('Native cold upgrade checkpoint is not the original initial snapshot')
    request = next((e for e in history if e['kind'] == 'model.request'), None)
    if request is None or not request['data']['request_id'].endswith(':r000000'):
        raise EvidenceError('Native cold upgrade lacks frozen origin request zero')
    return (parent/request['data']['body']['path']).read_bytes()


def _validate_actor_transition(binding:dict, previous_config:dict, config:dict,
                               parent_cp:dict, parent_index_sha256:str,
                               parent_checkpoint_sha256:str)->dict:
    """Bind the sole supported actor change to an exact sealed chat checkpoint."""
    expected={'schema_version':'cross-actor-model-transition-v1',
              'source_config_sha256':digest(previous_config),'target_config_sha256':digest(config),
              'source_model':previous_config['model'],'target_model':config['model'],
              'parent_index_sha256':parent_index_sha256,
              'parent_checkpoint_sha256':parent_checkpoint_sha256}
    if not isinstance(binding,dict) or binding != expected:
        raise ConfigurationError('Actor transition must exactly bind source/target configs, models and parent evidence')
    source=previous_config['model']; target=config['model']
    if (source['adapter'] not in ('openai-compatible','opencode-chat')
            or target['adapter'] != 'opencode-chat' or target.get('provider') != 'opencode-go'
            or target['model'] != 'deepseek-v4.1-flash' or source == target):
        raise ConfigurationError('Actor transition requires compatible OpenAI chat history to OpenCode Go DeepSeek V4.1 Flash')
    before=copy.deepcopy(previous_config); after=copy.deepcopy(config)
    before.pop('model'); after.pop('model')
    before.get('agent',{}).pop('intervention',None)
    after.get('agent',{}).pop('intervention',None)
    if before != after:
        raise ConfigurationError('Actor transition may change only model configuration and an explicit branch intervention')
    if source.get('supports_images',True) != target.get('supports_images',True):
        raise ConfigurationError('Actor transition must preserve image support and available tools')
    history=parent_cp.get('conversation')
    if (not isinstance(history,dict) or not isinstance(history.get('messages'),list)
            or not history['messages'] or 'contents' in history or 'input' in history
            or any(not isinstance(message,dict) or message.get('role') not in
                   ('system','user','assistant','tool') for message in history['messages'])):
        raise ConfigurationError('Actor transition requires native Chat Completions messages; history conversion is forbidden')
    return copy.deepcopy(expected)


def run(config:dict,output:Path,*,injected:dict|None=None,parent:Path|None=None,checkpoint_id:str|None=None,request_limiter=None,resume_response_id:str|None=None,allow_pi_native_upgrade:bool=False,resume_context_overflow_id:str|None=None,execution_observer=None,actor_transition:dict|None=None)->dict:
    """Optionally admit every HTTP attempt via acquire(estimate, timeout)->bool.

    reconcile(estimate, actual) retains unknown usage with actual=None. An
    admitted attempt that never sends refunds its reservation with actual=0.
    The limiter is external runtime state and never enters resolved config.
    An optional remaining_seconds() also bounds transport and tool deadlines.
    Optional per-run execution_observer binds external provenance before the
    first container; observation failure prevents that container from starting.
    """
    config=copy.deepcopy(validate(config)); output=Path(output).absolute()
    if (config['agent'].get('validation_policy') == 'pi-native-0.73.1'
            and (config['agent'].get('compaction') or {}).get('enabled')
            and config['agent'].get('harness_revision') != 'l1-pi-native-errors-context-v2'):
        raise ConfigurationError('Native Pi compaction execution requires harness_revision=l1-pi-native-errors-context-v2; migrate the config explicitly')
    config['agent']['system_prompt']=build_execution_contract_prompt(config)
    if execution_observer is not None:
        execution_observer.validate(copy.deepcopy(config))
    if output.is_symlink() or any(p.is_symlink() for p in output.parents): raise EvidenceError('symlink output path')
    if actor_transition is not None and (parent is None or checkpoint_id is None
            or allow_pi_native_upgrade or resume_response_id is not None or resume_context_overflow_id is not None):
        raise ConfigurationError('Actor transition requires a cold parent checkpoint and cannot combine with native upgrade or saved-response continuation')
    if resume_response_id is not None and (parent is None or checkpoint_id is None):
        raise ConfigurationError('Saved-response continuation requires parent and checkpoint_id')
    if resume_context_overflow_id is not None:
        if parent is None or checkpoint_id is None or resume_response_id is not None or not allow_pi_native_upgrade:
            raise ConfigurationError('Context-overflow continuation requires parent, checkpoint and explicit native upgrade; cannot consume a saved response')
    if allow_pi_native_upgrade and (parent is None or (resume_response_id is None and resume_context_overflow_id is None and checkpoint_id != 'cp000000')):
        raise ConfigurationError('Native upgrade requires saved-response consumption or original checkpoint zero')
    parent_cp=None; lineage=None; parent_manifest=None; saved_response=None; initial_upgrade_request=None; context_source=None
    if parent is not None:
        parent=Path(parent).resolve()
        pv=verify(parent)
        if not pv['verified']: raise EvidenceError('Parent evidence does not verify')
        parent_manifest=read_json(parent/'manifest.json')
        if parent_manifest['recording_mode']!='live' and injected is None:
            raise EvidenceError('Test/recovered evidence cannot qualify a production resume')
        if checkpoint_id is None or not __import__('re').fullmatch(r'cp\d{6}',checkpoint_id): raise ConfigurationError('checkpoint_id required')
        cp_file=confined(parent,f'checkpoints/{checkpoint_id}.json')
        if not cp_file.is_file(): raise EvidenceError(f'Checkpoint {checkpoint_id} not found in parent evidence')
        parent_cp=read_json(cp_file)
        if not parent_cp['qualification']['eligible']: raise EvidenceError('Checkpoint is not qualified')
        previous_config=read_json(parent/'raw/resolved_config.json')
        base_config=copy.deepcopy(config); prev_base=copy.deepcopy(previous_config)
        child_iv=base_config.get('agent',{}).pop('intervention',None)
        parent_iv=prev_base.get('agent',{}).pop('intervention',None)
        upgrade = allow_pi_native_upgrade and _check_native_upgrade(previous_config, config)
        if resume_context_overflow_id is not None and not _check_context_upgrade(previous_config, config):
            raise ConfigurationError('Context overflow requires only native-v1 to context-v2 transition with unchanged 600-second timeout and budgets')
        transition=None
        if actor_transition is not None:
            transition=_validate_actor_transition(actor_transition,previous_config,config,parent_cp,
                pv['index_sha256'],file_digest(cp_file))
        if base_config!=prev_base and not upgrade and transition is None:
            raise ConfigurationError('Cold resume currently requires the same resolved config; branch policy changes are not silently applied')
        if child_iv!=parent_iv:
            if child_iv is None or child_iv['at_request']<=parent_cp.get('budget_used',{}).get('model_requests',0):
                raise ConfigurationError('Branch intervention on cold resume must target a request strictly after the checkpoint')
        lineage={'parent_episode_id':parent_manifest['episode_id'],'parent_checkpoint_id':checkpoint_id,
                 'parent_index_sha256':pv['index_sha256'],'parent_checkpoint_sha256':file_digest(parent/f'checkpoints/{checkpoint_id}.json')}
        if transition is not None:
            lineage['actor_transition']=transition
        if allow_pi_native_upgrade:
            if child_iv != parent_iv:
                raise ConfigurationError('Native upgrade cannot change intervention')
            if config['agent'].get('validation_policy') != 'pi-native-0.73.1':
                raise ConfigurationError('Native upgrade requires explicit native policy')
            lineage['config_upgrade'] = {'policy':'pi-native-0.73.1', 'harness_revision':config['agent']['harness_revision'],
                'source_config_sha256':digest(previous_config), 'target_config_sha256':digest(config),
                'applied':bool(upgrade), 'mode':'context_overflow' if resume_context_overflow_id is not None else ('saved_response' if resume_response_id is not None else 'initial_checkpoint')}
            if resume_response_id is None and resume_context_overflow_id is None:
                initial_upgrade_request = _initial_upgrade_request(parent, parent_manifest, parent_cp)
                lineage['config_upgrade']['origin_request_sha256'] = digest_bytes(initial_upgrade_request)
        if resume_context_overflow_id is not None:
            from ..evidence.live import context_overflow_source
            context_source=context_overflow_source(parent,checkpoint_id,resume_context_overflow_id)
            lineage['context_overflow']=context_source['binding']
        if resume_response_id is not None:
            if config['agent'].get('validation_policy') != 'pi-native-0.73.1':
                raise ConfigurationError('Saved-response continuation requires explicit native validation policy')
            if child_iv != parent_iv:
                raise ConfigurationError('Saved-response continuation cannot change intervention')
            saved_response = _saved_response_source(parent, parent_manifest, parent_cp, resume_response_id)
            lineage['response_use'] = {'source_request_id': saved_response['request']['data']['request_id'],
                'source_call_id': saved_response['call']['data']['call_id'],
                'source_request_sha256': digest_bytes(saved_response['request_bytes']),
                'source_response_sha256': digest_bytes(saved_response['response_bytes']),
                'source_response_event_sha256': saved_response['response']['sha256'],
                'source_trajectory_head_sha256': read_json(parent/'evidence_index.json')['trajectory_head_sha256'],
                'new_http_attempts_during_consume': 0}
    eid='ep_'+uuid.uuid4().hex
    created=utc_now(); started=time.monotonic()
    inherited=parent_cp.get('budget_used',{}) if parent_cp else {}
    elapsed_before=inherited.get('agent_seconds',0.0)
    request_count=inherited.get('model_requests',0); tool_count=inherited.get('tool_calls',0)
    token_count=inherited.get('known_total_tokens',0); usage_complete=True
    if saved_response is not None:
        # Parent totals already include the response that failed validation.
        usage_complete=parent_manifest['usage']['all_requests_have_usage']
        elapsed_before=parent_manifest['usage']['agent_phase_seconds']
    if context_source is not None:
        usage=parent_manifest['usage']
        request_count=usage['model_requests_including_parent']
        tool_count=usage['tool_calls_including_parent']
        token_count=usage['known_total_tokens_including_parent']
        usage_complete=usage['all_requests_have_usage']
        elapsed_before=usage['agent_phase_seconds']
    deadline=started+max(0,config['budget']['wall_time_seconds']-elapsed_before)
    external_remaining=None
    remaining_seconds=getattr(request_limiter,'remaining_seconds',None)
    if callable(remaining_seconds):
        external_remaining=remaining_seconds()
        if type(external_remaining) not in (int,float) or not math.isfinite(external_remaining) or external_remaining<0:
            raise ConfigurationError('External admission deadline must be finite and nonnegative')
        deadline=min(deadline,started+external_remaining)
    output.mkdir(parents=True,exist_ok=False,mode=0o700)
    journal=Journal(output,eid); write_json(output,'raw/resolved_config.json',config)
    start_data={'config':journal.ref('raw/resolved_config.json'),'lineage':lineage}
    if external_remaining is not None: start_data['external_wall_seconds']=external_remaining
    journal.emit('episode.started',start_data)
    if actor_transition is not None:
        write_json(output,'lineage/actor_transition.json',lineage['actor_transition'])
        journal.emit('actor.transitioned',{'transition':journal.ref('lineage/actor_transition.json')})
    workspace=output.parent/('.l1-work-'+eid)
    workspace.mkdir(mode=0o700)
    # Shared host directories may inherit setgid even with mode=0700. The
    # declared portable restore profile excludes special permission bits.
    os.chmod(workspace,0o700)
    store=StateStore(output,config['runtime']['max_workspace_bytes'],config['runtime']['max_files'])
    last_snapshot=None; cp_records=[]; coverage=[]; mode='live'; task_info=None; runtime_info=None
    agent_info={'configuration':config['agent'],'model_configuration':config['model'],'identity':None}
    grade={'projection':None,'official_raw':None,'not_run_reason':'not_started'}
    status={'termination':'initializing','protocol_validity':'unknown','execution_validity':'unknown',
            'evaluation_validity':'not_run','task_outcome':'unknown','evidence_completeness':'complete'}
    conversation=None; task=None; spec=None; prepared=False; runtime=None; model=None; evaluator=None
    pending=[]; validation_errors_pending=[]; components_ready=False; native_validator=None; new_http_attempts=0; saved_response_consumptions=0; native_policy=config['agent'].get('validation_policy') == 'pi-native-0.73.1'
    def missing(code,detail):
        coverage.append({'code':code,'detail':detail}); status['evidence_completeness']='incomplete'
    def snapshot(label,tool_result_event_id=None):
        nonlocal last_snapshot
        if runtime is not None and not getattr(runtime,'is_quiescent',True):
            raise EvidenceError('runtime_not_quiescent')
        links={'tool_result':tool_result_event_id} if tool_result_event_id else {}
        current=store.capture(workspace,label)
        journal.emit('state.snapshot',{'snapshot_id':current['snapshot_id'],
                       'snapshot':journal.ref(f"states/snapshots/{current['snapshot_id']}.json")},links=links)
        if last_snapshot is not None:
            event=journal.emit('state.transition',{'before':last_snapshot['snapshot_id'],'after':current['snapshot_id']},links=links)
            journal.state_change({'episode_id':eid,'event_id':event['event_id'],
                'before_snapshot_id':last_snapshot['snapshot_id'],'after_snapshot_id':current['snapshot_id'],
                'changes':StateStore.diff(last_snapshot,current)})
        last_snapshot=current
        return current
    def checkpoint():
        if not config['recording']['checkpoints'] or last_snapshot is None: return
        q=runtime.qualification()
        reasons=list(q.get('reason_codes',[]))
        if any(item['mode'] & 0o7000 for item in last_snapshot['files'] + last_snapshot['directories']):
            reasons.append('unsupported_special_permission_bits')
        if request_count>=config['budget']['max_model_requests']: reasons.append('request_budget_exhausted')
        if time.monotonic()>=deadline: reasons.append('wall_time_budget_exhausted')
        token_limit=config['budget']['max_total_tokens']
        if token_limit is not None and (not usage_complete or token_count>=token_limit): reasons.append('token_budget_or_usage_unavailable')
        if reasons: q={**q,'eligible':False,'reason_codes':reasons}
        if mode!='live': q={**q,'eligible':bool(injected) and q['eligible'],'test_only':True}
        cp_id=f'cp{len(cp_records):06d}'
        cp={'schema_version':VERSION,'episode_id':eid,'checkpoint_id':cp_id,'snapshot_id':last_snapshot['snapshot_id'],
            'conversation':copy.deepcopy(conversation),'pending_observations':copy.deepcopy(pending),'pending_calls':[],
            'qualification':q,'config_sha256':digest(config),
            'budget_used':{'model_requests':request_count,'tool_calls':tool_count,'known_total_tokens':token_count,
                           'agent_seconds':time.monotonic()-started+elapsed_before},
            'lineage':lineage}
        relative=f'checkpoints/{cp_id}.json'; write_json(output,relative,cp)
        cp_records.append({'checkpoint_id':cp_id,'path':relative,'eligible':q['eligible']})
        journal.emit('checkpoint.created',{'checkpoint':journal.ref(relative)})
    try:
        if execution_observer is not None:
            execution_observer.bind_workspace(workspace)
            binding=execution_observer.bind_episode(eid,digest(config))
            write_json(output,'runtime/prospective/anchor.json',binding)
            journal.emit('provenance.bound',{'anchor':journal.ref('runtime/prospective/anchor.json')})
        if injected is not None:
            comp=injected; mode='test_fixture'
        elif parent_cp is None:
            comp=components(config)
        else:
            # No source checkout or dataset download is needed to resume a sealed
            # episode: benchmark sources, state and native conversation are embedded.
            from ..models.registry import registry as mr
            from ..runtime.registry import registry as rr
            from ..evaluation.registry import registry as er
            comp={}
            for section,maker in [('model',mr),('runtime',rr),('evaluation',er)]:
                r=maker()
                if config['plugins']['load_installed']: r.load_installed()
                elif section=='model' and config['model']['adapter']=='opencode-chat':
                    from ..models.opencode import OpenCodeChatBackend
                    r.register('opencode-chat',OpenCodeChatBackend)
                settings=copy.deepcopy(config[section])
                if section=='runtime': settings['image']=parent_cp['qualification']['image_id']
                comp[section]=r.create(settings['adapter'],settings)
        runtime=comp['runtime']; model=comp['model']; evaluator=comp['evaluation']; components_ready=True
        if hasattr(runtime,'set_event_sink'):
            runtime.set_event_sink(lambda kind,data:journal.emit(kind,data))
        if execution_observer is not None:
            if not hasattr(runtime,'set_execution_observer'):
                raise ConfigurationError('Runtime does not support prospective execution observation')
            def observe_container(cid,inspection):
                receipt=execution_observer.observe_container(cid,copy.deepcopy(inspection))
                relative=f'runtime/prospective/containers/{cid}.json'
                write_json(output,relative,{'container_id':cid,'inspection':inspection,'receipt':receipt})
                journal.emit('runtime.container.inspected',{'container_id':cid,'inspection':journal.ref(relative)})
                return receipt
            runtime.set_execution_observer(observe_container)
        model.native_pi_validation=native_policy
        agent_info['identity']=model.identity()
        if native_policy:
            from ..runtime.pi_native_validation import NativePiValidator
            native_validator=NativePiValidator()
            native_validator.start()
            agent_info['native_validation']=native_validator.dependency_identity()
            if execution_observer is not None:
                execution_observer.validate_native_identity(copy.deepcopy(agent_info['native_validation']))
        if parent_cp is None:
            journal.emit('benchmark.load.started',{'adapter':config['benchmark']['adapter'],'task_id':config['benchmark']['task_id']})
            task=comp['benchmark'].load(config['benchmark']['task_id'])
            task_info={'benchmark':task.benchmark,'task_id':task.task_id,'prompt':task.prompt,
                       'requirements':task.requirements,'provenance':task.provenance,
                       'evaluation':{'kind':task.evaluation.kind,'entrypoint':task.evaluation.entrypoint,'metadata':task.evaluation.metadata}}
            for relative,source in task.private_files.items():
                item=copy_file(source,output,'benchmark/'+relative)
                expected=task.provenance.get('private_files_sha256',{}).get(relative)
                if expected is not None and item['sha256']!=expected: raise EvidenceError('benchmark source changed during freeze')
            write_json(output,'benchmark/description.json',task_info)
            (workspace/'input').mkdir(); (workspace/'submission').mkdir()
            assets=[]
            for asset in task.assets:
                record=copy_file(asset.path,workspace,'input/'+asset.name)
                if record['sha256']!=asset.sha256 or record['size_bytes']!=asset.size_bytes:
                    raise EvidenceError('task input changed during staging')
                assets.append({'name':asset.name,'workspace_path':'/workspace/input/'+asset.name,
                               'sha256':asset.sha256,'size_bytes':asset.size_bytes})
            public={'benchmark':task.benchmark,'task_id':task.task_id,'prompt':task.prompt,'assets':assets}
            write_json(output,'benchmark/public_task.json',public)
            prompt=task.prompt+'\n\nInput files in /workspace/input:\n'+'\n'.join(a['name'] for a in assets)
            conversation=model.initial(config['agent']['system_prompt'],prompt)
            spec=replace(task.evaluation,source_root=output/'benchmark/source')
        else:
            _copy_tree(parent/'benchmark',output/'benchmark')
            task_info=read_json(output/'benchmark/description.json')
            e=task_info['evaluation']
            spec=EvaluationSpec(e['kind'],output/'benchmark/source',e['entrypoint'],task_info['task_id'],e['metadata'])
            ps=read_json(parent/f"states/snapshots/{parent_cp['snapshot_id']}.json")
            validate_snapshot(parent,ps); restore(parent,ps,workspace)
            conversation=copy.deepcopy(parent_cp['conversation'])
            # Pending observations are embedded as native history. Their delivery
            # references belong to the parent; preserve provenance, do not invent
            # duplicate child tool results to satisfy child event linkage.
            write_json(output,'lineage/parent_checkpoint.json',parent_cp)
            write_json(output,'lineage/parent_snapshot.json',ps)
            journal.emit('episode.restored',{'parent_index_sha256':lineage['parent_index_sha256'],
                'parent_snapshot_content_sha256':ps['content_sha256'],'prior_observations_in_native_history':len(parent_cp['pending_observations'])})
        journal.emit('benchmark.loaded',{'description':journal.ref('benchmark/description.json'),'public_task':journal.ref('benchmark/public_task.json')})
        # Trusted verifier files may be read by the unprivileged grader container;
        # never grant the agent a mount to this private directory.
        (output/'benchmark/source').mkdir(parents=True,exist_ok=True)
        for p in (output/'benchmark/source').rglob('*'):
            os.chmod(p,0o755 if p.is_dir() else 0o644)
        os.chmod(output/'benchmark/source',0o755)
        if initial_upgrade_request is not None:
            if canonical(model.request(conversation, tools_for_config(config))) != initial_upgrade_request:
                raise EvidenceError('Reconstructed initial request differs from frozen origin request zero')
            write_bytes(output,'lineage/origin_request_zero.bin',initial_upgrade_request)
            journal.emit('config.upgraded',{'upgrade':lineage['config_upgrade'],
                'origin_request':journal.ref('lineage/origin_request_zero.bin')})
        if saved_response is not None:
            reconstructed=canonical(model.request(conversation, tools_for_config(config)))
            if reconstructed != saved_response['request_bytes']:
                raise EvidenceError('Reconstructed checkpoint request differs from frozen parent request')
            # Reparse original bytes with the new validation boundary, preserving native history.
            source_reply=model.parse(saved_response['response_bytes'], saved_response['request']['data']['request_id'])
            if len(source_reply.calls) != 1 or source_reply.calls[0].name != 'edit':
                raise EvidenceError('Saved response projection differs from sole-edit restriction')
            write_json(output,'lineage/response_use.json',lineage['response_use'])
            write_bytes(output,'lineage/source_request.bin',saved_response['request_bytes'])
            write_bytes(output,'lineage/source_response.bin',saved_response['response_bytes'])
            copy_file(parent/'evidence_index.json',output,'lineage/source_evidence_index.json')
            copy_file(parent/'manifest.json',output,'lineage/source_manifest.json')
            copy_file(parent/'trajectory.jsonl',output,'lineage/source_trajectory.jsonl')
            journal.emit('response.bound',{'binding':journal.ref('lineage/response_use.json'),
                'request':journal.ref('lineage/source_request.bin'),'response':journal.ref('lineage/source_response.bin'),
                'source_evidence_index':journal.ref('lineage/source_evidence_index.json'),
                'source_manifest':journal.ref('lineage/source_manifest.json'),
                'source_trajectory':journal.ref('lineage/source_trajectory.jsonl')})
        if context_source is not None:
            if model.request(conversation,tools_for_config(config)) != context_source['failed_request']:
                raise EvidenceError('Context checkpoint request differs from frozen failed request')
            for relative in sorted(context_source['files'] | {'evidence_index.json'}):
                copy_file(parent/relative,output,'lineage/context_source/'+relative)
            journal.emit('context_overflow.carried',lineage['context_overflow'])
            journal.emit('config.upgraded',{'upgrade':lineage['config_upgrade']})
        runtime_info=runtime.preflight()
        if parent_cp and runtime_info.get('image_id')!=parent_cp['qualification']['image_id']:
            raise EvidenceError('Restored runtime image does not match checkpoint')
        write_json(output,'runtime/provenance.json',runtime_info)
        journal.emit('runtime.ready',{'provenance':journal.ref('runtime/provenance.json')})
        if hasattr(os,'getuid') and os.getuid()==0 and mode=='live':
            for p in [workspace,*workspace.rglob('*')]: os.chown(p,65532,65532)
        snapshot('restored_initial' if parent_cp else 'initial')
        if parent_cp and last_snapshot['content_sha256']!=ps['content_sha256']:
            raise EvidenceError('Restored file/directory state differs from checkpoint')
        prepared=True
        status.update(protocol_validity='valid',execution_validity='valid',termination='running')
        iv_cfg = config['agent'].get('intervention')
        parent_iv = previous_config.get('agent', {}).get('intervention') if parent_cp is not None else None
        if iv_cfg and iv_cfg != parent_iv and request_count == iv_cfg['at_request'] - 1:
            conversation = _inject_user_intervention(conversation, iv_cfg['message'])
        if context_source is not None:
            conversation, comp_receipt = _maybe_compact_pi_conversation(
                conversation,config['agent']['compaction'],context_source['binding']['last_success_prompt_tokens'],
                request_count,usage_messages=context_source['usage_messages'],usage_kind='prompt_tokens')
            if comp_receipt is not None: journal.emit('session.compacted',comp_receipt)
        checkpoint()
        while request_count<config['budget']['max_model_requests']:
            remaining=deadline-time.monotonic()
            if remaining<=0:
                status['termination']='wall_time_budget'; break
            rid=f'{eid}:r{request_count:06d}'
            request_count+=1
            tools=tools_for_config(config)
            body=model.request(conversation,tools)
            relative=f'raw/model/{rid.split(":")[-1]}/request.json'
            write_bytes(output,relative,canonical(body))
            consuming=saved_response is not None
            request_event=journal.emit('model.request',{'request_id':rid,'body':journal.ref(relative),
                'parameters_declared':config['model']['parameters'],
                **({'source':'saved_response','http_attempts':0} if consuming else {})})
            prefix=relative.removesuffix('request.json')
            request_bytes=canonical(body)
            if initial_upgrade_request is not None:
                if request_bytes != initial_upgrade_request:
                    raise EvidenceError('First fresh upgraded request differs from frozen origin request zero')
                initial_upgrade_request=None
            if consuming and request_bytes != saved_response['request_bytes']:
                raise EvidenceError('Consumed request differs from frozen parent request')
            estimated_tokens=_request_token_estimate(request_bytes,config['model']['parameters']) if request_limiter is not None else None
            admission_expired=False
            transport_attempts=0
            stop_phase='wall_deadline'
            def send_attempt():
                nonlocal admission_expired, transport_attempts, stop_phase, new_http_attempts
                rem=deadline-time.monotonic()
                if rem<=0:
                    admission_expired=True; return None
                if request_limiter is not None:
                    if not request_limiter.acquire(estimated_tokens,timeout=rem):
                        stop_phase='admission_timeout'
                        admission_expired=True; return None
                    rem=deadline-time.monotonic()
                    if rem<=0:
                        request_limiter.reconcile(estimated_tokens,0.0)
                        stop_phase='deadline_after_admission'
                        admission_expired=True; return None
                try:
                    transport_attempts+=1; new_http_attempts+=1
                    result=model.send(request_bytes,min(rem,config['model']['timeout_seconds']))
                except BaseException:
                    if request_limiter is not None: request_limiter.reconcile(estimated_tokens,None)
                    raise
                if request_limiter is not None: request_limiter.reconcile(estimated_tokens,_receipt_tokens(result))
                return result
            receipt=None
            retry_cfg=config['model'].get('retry')
            if consuming:
                receipt=HTTPResult(200,saved_response['response_bytes'],saved_response['metadata'])
                journal.emit('response.consumed',{'request_id':rid,'new_http_attempts':0,
                    'binding':journal.ref('lineage/response_use.json')},links={'request':request_event['event_id']})
                saved_response_consumptions+=1
                saved_response=None
            elif retry_cfg:
                max_attempts=retry_cfg.get('max_attempts',1)
                retry_statuses=set(retry_cfg.get('retry_on_status',[429,500,502,503,504]))
                retry_transport=bool(retry_cfg.get('retry_on_transport_error',True))
                backoffs=list(retry_cfg.get('backoff_seconds',[2,8,30]))
                honor_ra=bool(retry_cfg.get('honor_retry_after',True))
                for attempt in range(1,max_attempts+1):
                    attempt_receipt=send_attempt()
                    if attempt_receipt is None: break
                    receipt=attempt_receipt
                    ap=f'{prefix}attempt_{attempt:02d}/'
                    write_bytes(output,ap+'response.bin',receipt.body)
                    write_json(output,ap+'http.json',receipt.metadata)
                    truncated=bool(receipt.metadata.get('body_truncated'))
                    terr=bool(receipt.metadata.get('transport_error'))
                    retryable=not truncated and ((terr and retry_transport) or (not terr and receipt.status in retry_statuses))
                    if not retryable or attempt>=max_attempts: break
                    rem_after=deadline-time.monotonic()
                    if rem_after<=0:
                        admission_expired=True; break
                    base_delay=float(backoffs[min(attempt-1,len(backoffs)-1)]) if backoffs else 0.0
                    jitter=(int(digest({'rid':rid,'attempt':attempt})[:4],16)/65535.0)*(0.25*base_delay) if base_delay>0 else 0.0
                    delay=base_delay+jitter
                    if honor_ra:
                        ra=(receipt.metadata.get('headers') or {}).get('retry-after')
                        if ra is not None:
                            try:
                                ra_val=float(ra)
                                if ra_val>=0: delay=max(delay,ra_val)
                            except (ValueError,TypeError): pass
                    delay=min(delay,max(0.0,rem_after))
                    journal.emit('model.retry',{'request_id':rid,'attempt':attempt,'status_code':receipt.status,
                        'transport_error':receipt.metadata.get('transport_error'),'delay_seconds':delay,
                        'body':journal.ref(ap+'response.bin'),'http':journal.ref(ap+'http.json')},
                        links={'request':request_event['event_id']})
                    if delay>0 and (mode=='live' or delay<=0.05): time.sleep(delay)
                    if time.monotonic()>=deadline:
                        stop_phase='retry_backoff'
                        admission_expired=True; break
            else:
                receipt=send_attempt()
            if receipt is None:
                status['termination']='wall_time_budget'
                # Admission never produced a model continuation. Preserve the
                # budget stop reason without qualifying it as valid execution.
                status['execution_validity']='invalid'
                journal.emit('model.error',{'request_id':rid,'reason':'wall_time_budget',
                    'stop_phase':stop_phase,'transport_attempts':transport_attempts},links={'request':request_event['event_id']})
                break
            write_bytes(output,prefix+'response.bin',receipt.body)
            write_json(output,prefix+'http.json',receipt.metadata)
            response_event=journal.emit('model.response',{'request_id':rid,'body':journal.ref(prefix+'response.bin'),
                'http':journal.ref(prefix+'http.json'),'status_code':receipt.status},links={'request':request_event['event_id']})
            transport_invalid=(receipt.metadata.get('transport_error') or receipt.metadata.get('body_truncated')
                or receipt.status is None or not 200<=receipt.status<300)
            if transport_invalid:
                status['execution_validity']='invalid'
                if receipt.metadata.get('body_truncated'): missing('model_body_capped','Response prefix preserved; full body unavailable')
            if admission_expired or time.monotonic()>=deadline:
                status['termination']='wall_time_budget'
                journal.emit('model.error',{'request_id':rid,'reason':'wall_time_budget',
                    'stop_phase':stop_phase,'transport_attempts':transport_attempts},links={'response':response_event['event_id']})
                break
            if transport_invalid:
                status['termination']='model_transport_error'
                journal.emit('model.error',{'request_id':rid,'reason':status['termination']},links={'response':response_event['event_id']})
                break
            for observation in pending:
                journal.emit('observation.delivered',{'request_id':rid,'call_id':observation['call_id'],
                    'native_id':observation['native_id'],'observation':observation['observation'],
                    'assurance':'request_body_accepted_by_endpoint_not_attestation_of_attention'},
                    links={'request':request_event['event_id'],'result':observation['result_event_id']})
            pending=[]
            output_feedback_delivered=conversation.pop('_output_limit_feedback_pending',[])
            for feedback in output_feedback_delivered:
                journal.emit('model.output_limit_feedback_delivered',{
                    'request_id':rid,'source':feedback,
                    'assurance':'request_body_accepted_by_endpoint_not_attestation_of_attention'},
                    links={'request':request_event['event_id']})
            reply=model.parse(receipt.body,rid)
            journal.emit('model.parsed',{'request_id':rid,'usage':reply.usage,'reported_model_version':reply.model_version,
                'finish_reason':reply.finish_reason,'text':reply.text},links={'response':response_event['event_id']})
            usage=reply.usage or {}
            current_tokens=usage.get('totalTokenCount',usage.get('total_tokens'))
            if type(current_tokens) is int and current_tokens>=0:
                token_count+=current_tokens
            else: usage_complete=False
            if consuming:
                if request_count != int(resume_response_id[1:])+1 or token_count != parent_manifest['usage']['known_total_tokens_including_parent']:
                    raise EvidenceError('Consumed response cumulative request/token budget differs from parent')
                journal.emit('response.prefix_accounted',{'model_requests':request_count,'known_total_tokens':token_count,
                    'source_call_id':lineage['response_use']['source_call_id'],'new_http_attempts':0})
            output_limited=(reply.finish_reason=='length' and native_policy
                and config['model']['adapter']=='opencode-chat'
                and config['model'].get('options',{}).get('truncated_tool_call_policy')=='output_limit_error_feedback_v1')
            if output_feedback_delivered:
                journal.emit('model.response_after_output_limit',{'request_id':rid,
                    'prior_feedback':output_feedback_delivered,'finish_reason':reply.finish_reason,
                    'subsequent_calls':[{'name':c.name,'arguments':c.arguments} for c in reply.calls],
                    'interpretation':'observed_sequence_only_no_causal_attribution'},
                    links={'response':response_event['event_id']})
            if output_limited:
                limit_event=journal.emit('model.output_limit_feedback_created',{
                    'request_id':rid,'finish_reason':reply.finish_reason,
                    'policy':'output_limit_error_feedback_v1','feedback':OUTPUT_LIMIT_FEEDBACK,
                    'call_ids':[c.call_id for c in reply.calls],
                    'execution_policy':'reject_all_calls_in_length_response_without_repair'},
                    links={'response':response_event['event_id']})
            if not reply.calls and not output_limited:
                conversation=model.accept(conversation,reply,[])
                status['termination']='assistant_finished'; break
            if validation_errors_pending:
                journal.emit('model.response_after_validation_error',{'request_id':rid,
                    'prior_error_call_ids':validation_errors_pending,
                    'subsequent_calls':[{'name':c.name,'arguments':c.arguments} for c in reply.calls],
                    'interpretation':'observed_sequence_only_no_causal_attribution'})
                validation_errors_pending=[]
            call_events={}
            for call in reply.calls:
                ce=journal.emit('tool.call',{'request_id':rid,'call_id':call.call_id,'native_id':call.native_id,
                    'name':call.name,'arguments':call.arguments},links={'response':response_event['event_id']})
                call_events[call.call_id]=ce
            results=[]; interrupted_batch=False
            for index,call in enumerate(reply.calls):
                if tool_count>=config['budget']['max_tool_calls'] or time.monotonic()>=deadline:
                    reason='tool_budget' if tool_count>=config['budget']['max_tool_calls'] else 'wall_time_budget'
                    for skipped in reply.calls[index:]:
                        journal.emit('tool.skipped',{'call_id':skipped.call_id,'reason':reason},links={'call':call_events[skipped.call_id]['event_id']})
                    status['termination']=reason; interrupted_batch=True; break
                validation=None; execution_call=call
                if output_limited:
                    tool_count+=1
                    journal.emit('tool.output_limit_rejected',{'call_id':call.call_id,
                        'policy':'output_limit_error_feedback_v1','arguments_unchanged':True,
                        'tool_executed':False},links={'call':call_events[call.call_id]['event_id']})
                elif native_policy:
                    tool_count+=1
                    if context_source is not None:
                        journal.emit('tool.charged',{'call_id':call.call_id,'tool_calls_including_parent':tool_count},
                            links={'call':call_events[call.call_id]['event_id']})
                    try:
                        validation=native_validator.validate(call.name,call.arguments,call_id=call.native_id or call.call_id,
                            deadline_monotonic=deadline)
                    except Exception as exc:
                        if time.monotonic()>=deadline:
                            for skipped in reply.calls[index:]:
                                journal.emit('tool.skipped',{'call_id':skipped.call_id,'reason':'wall_time_budget'},
                                    links={'call':call_events[skipped.call_id]['event_id']})
                            status['termination']='wall_time_budget'; interrupted_batch=True; break
                        journal.emit('tool.error',{'call_id':call.call_id,'error_type':type(exc).__name__,'message':str(exc)},
                            links={'call':call_events[call.call_id]['event_id']})
                        for skipped in reply.calls[index+1:]:
                            journal.emit('tool.skipped',{'call_id':skipped.call_id,'reason':'native_validation_infrastructure_error'},
                                links={'call':call_events[skipped.call_id]['event_id']})
                        raise
                    journal.emit('tool.validation',{'call_id':call.call_id, **validation},
                        links={'call':call_events[call.call_id]['event_id']})
                    if time.monotonic()>=deadline:
                        for skipped in reply.calls[index:]:
                            journal.emit('tool.skipped',{'call_id':skipped.call_id,'reason':'wall_time_budget'},
                                links={'call':call_events[skipped.call_id]['event_id']})
                        status['termination']='wall_time_budget'; interrupted_batch=True; break
                    if not validation['ok']: validation_errors_pending.append(call.call_id)
                    if validation['ok']:
                        execution_call=NativeValidatedCall(call.call_id,call.name,validation['validated_arguments'],call.native_id)
                try:
                    if not native_policy and not output_limited: validate_call(call,config['model']['supports_images'],harness=active_harness(config))
                except ProtocolError:
                    for rejected in reply.calls[index:]:
                        journal.emit('tool.rejected',{'call_id':rejected.call_id,'reason':'protocol_error'},links={'call':call_events[rejected.call_id]['event_id']})
                    raise
                if not native_policy and not output_limited: tool_count+=1
                if not output_limited and (validation is None or validation['ok']):
                    journal.emit('tool.started',{'call_id':call.call_id,'before_snapshot_id':last_snapshot['snapshot_id']},links={'call':call_events[call.call_id]['event_id']})
                try:
                    if output_limited:
                        actual=_tool_error_result(OUTPUT_LIMIT_FEEDBACK,'opencode_output_limit_v1',utc_now())
                    elif validation is not None and not validation['ok']:
                        actual=Result(copy.deepcopy(validation['native_result']),
                            provenance={'pi_native_validation_error':True, 'validation':validation['provenance'], 'quiescent':True})
                    else:
                        execution_remaining=deadline-time.monotonic()
                        if execution_remaining<=0:
                            for skipped in reply.calls[index:]:
                                journal.emit('tool.skipped',{'call_id':skipped.call_id,'reason':'wall_time_budget'},
                                    links={'call':call_events[skipped.call_id]['event_id']})
                            status['termination']='wall_time_budget'; interrupted_batch=True; break
                        actual=runtime.execute(execution_call,workspace,min(config['runtime']['tool_timeout_seconds'],execution_remaining))
                except Exception as exc:
                    journal.emit('tool.error',{'call_id':call.call_id,'error_type':type(exc).__name__,'message':str(exc)},links={'call':call_events[call.call_id]['event_id']})
                    for skipped in reply.calls[index+1:]:
                        journal.emit('tool.skipped',{'call_id':skipped.call_id,'reason':'earlier_tool_error'},links={'call':call_events[skipped.call_id]['event_id']})
                    raise
                key=call.call_id.split(':',1)[1].replace(':','_'); tp='raw/tools/'+key+'/'
                write_bytes(output,tp+'stdout.bin',actual.stdout); write_bytes(output,tp+'stderr.bin',actual.stderr)
                write_json(output,tp+'result.json',{'call_id':call.call_id,'observation':actual.observation,'provenance':actual.provenance})
                result_data={'call_id':call.call_id,'result':journal.ref(tp+'result.json'),
                    'stdout':journal.ref(tp+'stdout.bin'),'stderr':journal.ref(tp+'stderr.bin')}
                if actual.image:
                    write_bytes(output,tp+'image.bin',actual.image); result_data['image']=journal.ref(tp+'image.bin')
                re=journal.emit('tool.result',result_data,links={'call':call_events[call.call_id]['event_id']})
                if actual.observation.get('output_complete') is False:
                    missing('tool_output_capped','Tool stopped at capture limit; retained prefix is not full output')
                snapshot('after_'+key,re['event_id'])
                delivered=copy.deepcopy(actual.observation)
                limit=config['recording']['observation_limit_chars']; cut=[]
                for field in ('stdout','stderr','output'):
                    if isinstance(delivered.get(field),str) and len(delivered[field])>limit:
                        delivered[field]=delivered[field][:limit]; cut.append(field)
                if cut: delivered['observation_truncation']={'fields':cut,'limit_chars':limit,'raw_retained':True}
                op='observations/'+key+'.json'; write_json(output,op,delivered)
                pending.append({'call_id':call.call_id,'native_id':call.native_id,'result_event_id':re['event_id'],'observation':journal.ref(op)})
                results.append((call,Result(delivered,actual.stdout,actual.stderr,actual.image,actual.image_mime,actual.provenance)))
            if interrupted_batch: break
            usage_messages = conversation.get('messages')
            conversation=model.accept(conversation,reply,results)
            if output_limited:
                if not reply.calls:
                    conversation=_inject_user_intervention(conversation,OUTPUT_LIMIT_FEEDBACK)
                conversation.setdefault('_output_limit_feedback_pending',[]).append({
                    'source_request_id':rid,'source_event_id':limit_event['event_id'],
                    'feedback_sha256':digest(OUTPUT_LIMIT_FEEDBACK),
                    'delivery_role':'tool' if reply.calls else 'user',
                    'call_ids':[c.call_id for c in reply.calls]})
            context_usage = usage.get('prompt_tokens', usage.get('promptTokenCount'))
            context_usage_kind = 'prompt_tokens'
            if type(context_usage) is not int or context_usage <= 0:
                context_usage = current_tokens
                context_usage_kind = 'total_tokens'
            conversation, comp_receipt = _maybe_compact_pi_conversation(
                conversation,
                config['agent'].get('compaction'),
                context_usage,
                request_count,
                usage_messages=usage_messages,
                usage_kind=context_usage_kind,
            )
            if comp_receipt is not None:
                journal.emit('session.compacted', comp_receipt)
            if iv_cfg and request_count == iv_cfg['at_request'] - 1:
                conversation = _inject_user_intervention(conversation, iv_cfg['message'])
            checkpoint()
            token_limit=config['budget']['max_total_tokens']
            if token_limit is not None and (not usage_complete or token_count>=token_limit):
                status['termination']='token_usage_unavailable' if not usage_complete else 'token_budget'; break
        else:
            status['termination']='request_budget'
    except KeyboardInterrupt:
        status['termination']='interrupted'; status['execution_validity']='invalid'
        journal.emit('episode.interrupted',{'reason':'keyboard_interrupt'})
        missing('interrupted_operation','The operation in flight may lack a terminal receipt')
    except Exception as exc:
        code=getattr(exc,'code','infrastructure_error')
        status['termination']=code; status['execution_validity']='invalid'
        if isinstance(exc,ProtocolError): status['protocol_validity']='invalid'
        if isinstance(exc,EvidenceError): missing(code,str(exc))
        journal.emit('episode.error',{'error_type':type(exc).__name__,'code':code,'message':str(exc)})
    finally:
        if native_validator is not None: native_validator.close()
        agent_phase_elapsed=time.monotonic()-started+elapsed_before
        # The controller attempts final state and scoring even after agent failure.
        # A grade failure is never allowed to overwrite the agent stop reason.
        final_state_captured=False
        if prepared and runtime is not None:
            try:
                if not getattr(runtime,'is_quiescent',True): raise EvidenceError('runtime_not_quiescent')
                final=store.capture(workspace,'final')
                journal.emit('state.snapshot',{'snapshot_id':final['snapshot_id'],'snapshot':journal.ref(f"states/snapshots/{final['snapshot_id']}.json")})
                if last_snapshot is not None:
                    transition=journal.emit('state.transition',{'before':last_snapshot['snapshot_id'],'after':final['snapshot_id']})
                    journal.state_change({'episode_id':eid,'event_id':transition['event_id'],'before_snapshot_id':last_snapshot['snapshot_id'],
                        'after_snapshot_id':final['snapshot_id'],'changes':StateStore.diff(last_snapshot,final)})
                last_snapshot=final
                final_state_captured=True
            except Exception as exc:
                missing('final_snapshot_failed',str(exc)); status['execution_validity']='invalid'
                journal.emit('state.error',{'phase':'final','error_type':type(exc).__name__,'message':str(exc)})
        if prepared and evaluator is not None and final_state_captured:
            try:
                submission=workspace/'submission'
                # Snapshot safety checks include this tree; copy to a separate frozen
                # submission so no later agent changes or references enter grading.
                submission_copy=output/'artifacts/submission'; submission_copy.mkdir(parents=True,exist_ok=False)
                artifact_records=[]
                for item in last_snapshot['files']:
                    if not item['path'].startswith('submission/'): continue
                    relative=item['path'][len('submission/'):]
                    source=output/item['blob']
                    record=copy_file(source,submission_copy,relative)
                    os.chmod(submission_copy/relative,0o644)
                    artifact_records.append(record)
                for p in [submission_copy,*submission_copy.rglob('*')]:
                    if p.is_dir(): os.chmod(p,0o755)
                write_json(output,'artifacts/inventory.json',{'episode_id':eid,'snapshot_id':last_snapshot['snapshot_id'],'files':artifact_records})
                journal.emit('evaluation.started',{'adapter':config['evaluation']['adapter'],'submission_files':len(artifact_records)})
                grade=evaluator.evaluate(spec,submission_copy,output/'grading')
                status['evaluation_validity']=grade['projection']['evaluation_validity']
                status['task_outcome']=grade['projection']['task_outcome']
                journal.emit('evaluation.finished',{'evaluation_validity':status['evaluation_validity'],'task_outcome':status['task_outcome']})
            except Exception as exc:
                status['evaluation_validity']='invalid'
                grade={'projection':{'evaluation_validity':'invalid','task_outcome':'unknown','reason':'grader_exception'},
                       'official_raw':None,'error_type':type(exc).__name__,'message':str(exc)}
                journal.emit('evaluation.error',{'error_type':type(exc).__name__,'message':str(exc)})
        else:
            grade={'projection':None,'official_raw':None,'not_run_reason':'final_state_unavailable_or_episode_not_prepared'}
        if not (output/'artifacts/inventory.json').exists():
            write_json(output,'artifacts/inventory.json',{'episode_id':eid,'snapshot_id':None,'files':[]})
        write_json(output,'official_grade.json',grade)
        write_json(output,'checkpoints/index.json',{'schema_version':VERSION,'episode_id':eid,'checkpoints':cp_records})
        if conversation is not None: write_json(output,'raw/final_conversation.json',conversation)
        if not prepared:
            missing('execution_not_prepared','No qualified initial workspace/runtime boundary')
        if mode!='live':
            if status['execution_validity']!='invalid': status['execution_validity']='unknown'
            coverage.append({'code':'test_fixture','detail':'Injected components; NOT a real model/Docker qualification'})
        coverage.extend([{'code':'object_state_not_collected','detail':'Only file/directory state is supported'},
                         {'code':'provider_hidden_state_unknown','detail':'No claim of deterministic model response replay'},
                         {'code':'actual_billing_unknown','detail':'Provider usage retained; actual monetary cost is null'}])
        journal.emit('episode.finished',{'status':copy.deepcopy(status)})
        counts={
            'events':len(journal.events),
            **{key:sum(e['kind']==kind for e in journal.events) for key,kind in
               [('model_requests','model.request'),('model_responses','model.response'),('tool_calls','tool.call'),
                ('tool_results','tool.result'),('observations_delivered','observation.delivered'),('snapshots','state.snapshot')]}}
        manifest={'schema_version':VERSION,'episode_id':eid,'recording_mode':mode,'created_at':created,'finished_at':utc_now(),
            'task':task_info,'agent':agent_info,'runtime':runtime_info,'status':status,'counts':counts,'coverage':coverage,
            'capabilities':{'state_scope':'files_and_directories','checkpoint_scope':'cold_workspace_and_native_conversation',
                'eligible_checkpoints':sum(x['eligible'] for x in cp_records),'deterministic_model_replay':False,
                'object_state':False,'official_batch_runner':False},
            'lineage':lineage,'usage':{'known_total_tokens_including_parent':token_count,'all_requests_have_usage':usage_complete and sum(e['kind']=='model.parsed' for e in journal.events)==sum(e['kind']=='model.request' for e in journal.events),
                'actual_cost_usd':None,'new_http_attempts':new_http_attempts,'saved_response_consumptions':saved_response_consumptions,'model_requests_including_parent':request_count,'tool_calls_including_parent':tool_count,'budget_includes_parent':bool(parent_cp),'agent_phase_seconds':agent_phase_elapsed,
                'token_limit_enforcement':'post_response_threshold_not_hard_preallocation'},'config_sha256':digest(config)}
        write_json(output,'manifest.json',manifest)
        seal(output,journal.events)
        shutil.rmtree(workspace,ignore_errors=True)
    report=verify(output)
    return {'episode_id':eid,'output':str(output),'verification':report,'status':status,
            'ready_for_live_claim':mode=='live' and status['execution_validity']=='valid'}
