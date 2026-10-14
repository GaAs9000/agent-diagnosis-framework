"""Collect actual public observations and freeze a task-bound D4 panel."""
import asyncio
import json
import argparse
from pathlib import Path

from c3_sandbox.evidence._storage import encode
from c3_sandbox.r1.docker_runtime import DockerRuntime
from c3_sandbox.r1.probes import render_pair, render_probe
from c3_sandbox.r1.public_evidence import collect_public_evidence
from c3_sandbox.r1.qualification import TOOL_SCHEMAS
from c3_sandbox.r1.tokenizer_client import TokenizerClient

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    with path.open('xb') as stream:
        stream.write(encode(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runs/e2-r1')
    parser.add_argument('--output-name', default='probe-audit-v2')
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in ('', '.', '..'):
        raise ValueError('output name must be a single directory name')
    root = args.run_root.absolute()
    folder = root / args.output_name
    folder.mkdir(mode=0o700)
    baseline = json.loads((root / 'originals/officeval_002/result.json').read_bytes())
    checkpoint = baseline['checkpoints'][0]
    messages = json.loads(Path(checkpoint['messages_path']).read_bytes())
    tool_event = next(event for event in baseline['agent_result']['events']
                      if event['event_index'] == checkpoint['event_index'])
    if tool_event['type'] != 'tool_result':
        raise RuntimeError('checkpoint is not a completed native tool boundary')
    with DockerRuntime(baseline['runtime']['image_digest'], folder / 'runtime') as actor:
        actor.seed_workspace(Path(checkpoint['workspace']))
        before = actor.verify_workspace(Path(checkpoint['workspace']))
        public = collect_public_evidence(actor,
            preworkspace=Path(baseline['output_dir']) / 'public-input',
            last_tool_result=tool_event['raw_result'])
        after = actor.verify_workspace(Path(checkpoint['workspace']))
        audit = actor.audit_boundary()
        if not audit['qualified']:
            raise RuntimeError('read-only observation collector left unsupported state')
    write(folder / 'public_evidence.json', public)
    write(folder / 'collection_audit.json', {'before': before, 'after': after,
        'boundary': audit, 'prior_tool_event': tool_event,
        'checkpoint_id': checkpoint['checkpoint_id'],
        'no_hidden_grader_inputs': True,
        'information_source': 'read-only fixed collector inside actor with public files only'})
    tokenizer = TokenizerClient()
    rendered = render_pair('visibility.expanded', public, token_counter=tokenizer.text_count)
    for probe in ('original', 'feedback.structured'):
        rendered[probe] = render_probe(probe, public, token_counter=tokenizer.text_count)
    panel = {probe: value.to_dict() for probe, value in rendered.items()}
    full_counts = {}
    for probe, value in panel.items():
        resumed = messages + ([{'role': 'user', 'content': value['rendered_text']}]
                              if value['rendered_text'] else [])
        full_counts[probe] = asyncio.run(tokenizer.count_messages(resumed, TOOL_SCHEMAS))
    if full_counts['visibility.expanded'] != full_counts['visibility.expanded.matched']:
        raise RuntimeError('actual full prompt token costs differ')
    write(folder / 'rendered_panel.json', panel)
    write(folder / 'token_audit.json', {'model_fingerprint': baseline['model_qualification']['model_fingerprint'],
        'full_prompt_tokens': full_counts, 'target_matched_equal': True,
        'same_message_position': len(messages), 'same_context_window': 16384,
        'no_extra_model_turns_granted': True, 'control_inertness_assumed': False})
    print(json.dumps({'public_inventory': public, 'full_prompt_tokens': full_counts,
                      'target_matched_equal': True}))


if __name__ == '__main__':
    main()
