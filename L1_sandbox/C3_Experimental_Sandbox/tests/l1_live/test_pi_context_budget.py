"""Offline regressions for observed Pi context budgets; no model requests."""
import copy
import json
from pathlib import Path

import pytest

from c3_sandbox.l1.core.contracts import ConfigurationError, L1Error
from c3_sandbox.l1.core.orchestrator import (
    _check_native_upgrade, _estimate_message_tokens, _maybe_compact_pi_conversation, run,
)
from c3_sandbox.l1.storage import digest


CFG = dict(enabled=True, context_window_tokens=12000, reserve_tokens=2000, keep_recent_tokens=3000)


def turn(index, size=8000, image=False, batch=False):
    ids = [f'call-{index}-{part}' for part in range(2 if batch else 1)]
    messages = [{'role': 'assistant', 'content': 'step', 'tool_calls': [
        {'id': key, 'type': 'function', 'function': {'name': 'read', 'arguments': '{"path":"input.txt"}'}}
        for key in ids
    ]}]
    for key in ids:
        messages.append({'role': 'tool', 'tool_call_id': key, 'content': 'x' * size})
        if image:
            messages.append({'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]})
    return messages


def history():
    return {'messages': [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'initial task'}]
            + turn(1) + turn(2) + turn(3) + turn(4) + turn(5)}


def test_reported_usage_still_triggers_after_previous_compaction():
    conv = history()
    conv['_pi_compaction'] = [{'read_files': ['old.txt'], 'modified_files': ['saved.txt']}]
    original = copy.deepcopy(conv)
    result, receipt = _maybe_compact_pi_conversation(conv, CFG, 30000, 10)
    assert receipt is not None
    assert receipt['tokens_before'] >= 30000
    assert receipt['tokens_after_estimate'] <= 10000
    assert result['messages'][:2] == conv['messages'][:2]
    assert result['messages'][-2:] == conv['messages'][-2:]
    assert 'old.txt' in receipt['read_files']
    assert 'saved.txt' in receipt['modified_files']
    assert conv == original


@pytest.mark.parametrize('raw_arguments', ['[]', 'null', '1'])
def test_compaction_preserves_rejected_nonobject_tool_calls(raw_arguments):
    conv = history()
    conv['messages'][2]['tool_calls'][0]['function']['arguments'] = raw_arguments
    original = copy.deepcopy(conv)
    result, receipt = _maybe_compact_pi_conversation(conv, CFG, 30000, 10)
    assert receipt is not None
    assert receipt['tokens_after_estimate'] <= 10000
    assert result['messages'][:2] == original['messages'][:2]
    assert conv == original


def test_stale_usage_cannot_recompact_same_history():
    conv = history()
    result, receipt = _maybe_compact_pi_conversation(conv, CFG, 30000, 10)
    repeated, receipt2 = _maybe_compact_pi_conversation(result, CFG, 30000, 10)
    assert receipt is not None
    assert receipt2 is None
    assert repeated == result
    assert repeated['_pi_context_usage']['source_messages_sha256'] == digest(conv['messages'])


def test_new_observation_calibrates_again_after_compaction_and_restore():
    compacted, _ = _maybe_compact_pi_conversation(history(), CFG, 30000, 10)
    compacted = json.loads(json.dumps(compacted))
    source = copy.deepcopy(compacted['messages'])
    compacted['messages'] += turn(6, 500)
    result, receipt = _maybe_compact_pi_conversation(compacted, CFG, 25000, 11, usage_messages=source)
    assert receipt is not None
    assert result['_pi_context_usage']['at_request'] == 11
    assert result['_pi_context_usage']['source_messages_sha256'] == digest(source)
    assert receipt['tokens_after_estimate'] <= 10000


def test_complete_multitool_batch_and_native_ids_survive():
    conv = history()
    conv['messages'] += turn(6, 500, image=True, batch=True)
    result, receipt = _maybe_compact_pi_conversation(conv, CFG, 13000, 10)
    assert receipt is not None
    assert result['messages'][-5:] == conv['messages'][-5:]
    pending = set()
    for message in result['messages']:
        if message['role'] == 'assistant':
            assert not pending
            pending.update(call['id'] for call in message.get('tool_calls', []))
        elif message['role'] == 'tool':
            assert message['tool_call_id'] in pending
            pending.remove(message['tool_call_id'])
    assert not pending


@pytest.mark.parametrize('kind', ['initial', 'latest', 'summary_only', 'incomplete_batch'])
def test_unsatisfiable_history_is_explicit(kind):
    conv = history()
    if kind == 'initial':
        conv['messages'][1]['content'] = 'task' * 20000
    elif kind == 'latest':
        conv['messages'][-1]['content'] = 'result' * 20000
    elif kind == 'summary_only':
        conv['messages'] = conv['messages'][:2] + [
            {'role': 'user', 'content': '[PI CONTEXT COMPACTION SUMMARY #1]'}
        ] + turn(1, 100000)
    else:
        conv['messages'].pop()
        conv['messages'][3]['content'] *= 4
    with pytest.raises(L1Error) as error:
        _maybe_compact_pi_conversation(conv, CFG, None, 10)
    assert error.value.code == 'pi_context_unsatisfiable'


def test_calibration_persists_without_compaction_and_accounts_for_retained_images():
    conv = history()
    conv['messages'] += turn(6, 100, image=True)
    high_budget = {**CFG, 'context_window_tokens': 100000}
    calibrated, receipt = _maybe_compact_pi_conversation(conv, high_budget, 40000, 10)
    assert receipt is None
    assert calibrated['_pi_context_usage']['extra_tokens_per_image'] > 0
    # Removing text cannot erase the measured image surcharge.
    calibrated['messages'] = calibrated['messages'][:2] + turn(6, 100, image=True)
    with pytest.raises(L1Error, match='Pi context cannot fit'):
        _maybe_compact_pi_conversation(calibrated, CFG, 40000, 10)


def test_disabled_compaction_preserves_history():
    conv = history()
    result, receipt = _maybe_compact_pi_conversation(conv, {'enabled': False}, 100000, 10)
    assert result is conv
    assert receipt is None


def test_oversized_initial_history_cannot_silently_skip_compaction():
    conv = {'messages': [{'role': 'system', 'content': 'system'},
                         {'role': 'user', 'content': 'task' * 20000}]}
    with pytest.raises(L1Error) as error:
        _maybe_compact_pi_conversation(conv, CFG, None, 1)
    assert error.value.code == 'pi_context_unsatisfiable'


def test_native_v1_compaction_execution_requires_explicit_revision(config, tmp_path):
    config['agent'] = {'name': 'pi-coding-agent', 'version': '0.73.1', 'harness': 'pi',
                       'validation_policy': 'pi-native-0.73.1', 'harness_revision': 'l1-pi-native-errors-v1',
                       'compaction': CFG}
    with pytest.raises(ConfigurationError, match='migrate the config explicitly'):
        run(config, tmp_path / 'blocked')
    assert not (tmp_path / 'blocked').exists()


def test_upgrade_only_permits_frozen_harness_and_timeout_transition():
    old = {'agent': {'harness': 'pi', 'version': '0.73.1', 'validation_policy': 'pi-native-0.73.1',
                     'harness_revision': 'l1-pi-native-errors-v1'}, 'model': {'timeout_seconds': 300, 'max_tokens': 8192}}
    new = copy.deepcopy(old)
    new['agent']['harness_revision'] = 'l1-pi-native-errors-context-v2'
    assert _check_native_upgrade(old, new)
    new['model']['timeout_seconds'] = 600
    assert _check_native_upgrade(old, new)
    new['model']['max_tokens'] = 4000
    assert not _check_native_upgrade(old, new)


def test_recorded_004_repeated_compaction_context_overflow():
    # Optional real-artifact regression in the server mirror. Historical raw
    # records are read only; CI without the evidence still runs synthetic cases.
    sandbox = Path(__file__).resolve().parents[2]
    segment = sandbox / 'runs/v3_qwen36_pi_dev45_r22_20261018/segments/officeval_004.r02'
    if not segment.is_dir():
        pytest.skip('recorded 004 evidence is not installed')
    config = json.loads((segment / 'raw/resolved_config.json').read_text())['agent']['compaction']
    conv = json.loads((segment / 'checkpoints/cp000003.json').read_text())['conversation']
    source = json.loads((segment / 'raw/model/r000010/request.json').read_text())['messages']
    assert conv['_pi_compaction'][-1]['tokens_after_estimate'] == 42369
    assert conv['messages'] == source
    result, receipt = _maybe_compact_pi_conversation(conv, config, 114090, 11, usage_messages=source)
    assert receipt is not None
    assert receipt['tokens_after_estimate'] <= config['context_window_tokens'] - config['reserve_tokens']
    assert result['messages'][:2] == conv['messages'][:2]
    assert receipt['tokens_before'] > sum(_estimate_message_tokens(m) for m in source)
    assert _maybe_compact_pi_conversation(result, config, 114090, 11)[1] is None
