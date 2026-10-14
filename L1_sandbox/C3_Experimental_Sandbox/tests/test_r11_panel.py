"""Synthetic fixtures only: no Docker, SDK generation, or tokenizer network."""
import copy
import json
from pathlib import Path

import pytest

from c3_sandbox.evidence._storage import encode
from c3_sandbox.experiment_scheduler import ExecutionOutcome, ExperimentScheduler
from c3_sandbox.manifests import canonical_hash
from c3_sandbox.replay import ReplayQualification, STATE_CONTRACT_V1
from c3_sandbox.r1 import r11_panel as panel
from c3_sandbox.r1.docker_runtime import restore_snapshot, snapshot_workspace
from c3_sandbox.r1.probes import INVENTORY_FIELDS, render_pair, render_probe


def write(path, value):
    path.write_bytes(encode(value))


@pytest.fixture
def prepared_case(tmp_path, monkeypatch):
    ws = tmp_path / 'source'
    ws.mkdir()
    (ws / 'document.docx').write_bytes(b'old')
    prior = snapshot_workspace(ws)
    (ws / 'document.docx').write_bytes(b'edited')
    final = snapshot_workspace(ws)
    runtime = {'image_digest': 'sha256:' + 'a' * 64, 'config': {}, 'model_config': {},
               'tool_config': {}, 'environment': {}}
    messages = [{'role': 'user', 'content': 'task'}, {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'actual', 'type': 'function', 'function': 'workspace_python', 'arguments': {'code': 'model edit'}}]},
        {'role': 'tool', 'content': 'ok', 'tool_call_id': 'actual', 'function': 'workspace_python'}]
    replay = ReplayQualification(tmp_path / 'store')
    identifier = replay.register_checkpoint(ws, messages, 'b' * 64, runtime, STATE_CONTRACT_V1, is_synthetic=False)
    write(tmp_path / 'messages.json', messages)
    write(tmp_path / 'source_audit.json', {'qualified': True, 'inspect': {'State': {'Pid': 10}}})
    write(tmp_path / 'before.json', prior)
    write(tmp_path / 'after.json', final)
    raw = {'exit_code': 0, 'stdout': 'done', 'stdout_bytes': 4, 'stderr_bytes': 0}
    event = {'event_index': 1, 'type': 'tool_result', 'tool_call_id': 'actual', 'function': 'workspace_python',
             'arguments': {'code': 'model edit'}, 'raw_result': raw, 'error': None}
    dispatch = {**event, 'before_path': str(tmp_path / 'before.json'), 'after_path': str(tmp_path / 'after.json'),
                'snapshot_ineligible_reasons': []}
    cp = {'checkpoint_id': identifier, 'event_index': 1, 'workspace': str(ws), 'runtime_identity': canonical_hash(runtime),
          'messages_path': str(tmp_path / 'messages.json'), 'source_audit_path': str(tmp_path / 'source_audit.json')}
    baseline = {'task_id': 'task', 'runtime': runtime, 'system_identity': 'b' * 64, 'replay_store': str(tmp_path / 'store'),
                'is_synthetic': False, 'execution_valid': True, 'task_completed': True, 'checkpoints': [cp],
                'model_qualification': {'model_fingerprint': 'c' * 64, 'tokenizer_fingerprint': {'sha256': 'd' * 64}},
                'agent_result': {'events': [event]}, 'tool_dispatch': [dispatch]}
    selection = {'kind': 'S', 'checkpoint_id': identifier, 'reference': cp}
    observed = {}
    class Runtime:
        def __init__(self, image, root):
            self.ws = Path(root) / 'workspace'
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def seed_workspace(self, source):
            restore_snapshot(snapshot_workspace(source), self.ws)
        def verify_workspace(self, source):
            assert snapshot_workspace(source) == snapshot_workspace(self.ws)
            return {'verified': True, 'expected_paths': 1, 'actual_paths': 1}
        def audit_boundary(self):
            return {'qualified': True, 'inspect': {'State': {'Pid': 11}}}
    def collect(actor, *, checkpoint, last_tool_result):
        observed['prior'] = checkpoint
        observed['raw'] = last_tool_result
        inventory = {key: {} if key == 'extension_counts' else 0 for key in INVENTORY_FIELDS}
        inventory.update(file_count=1, docx_count=1, total_bytes=6, extension_counts={'.docx': 1})
        return {'inventory': inventory, 'feedback': {'exit_code': 0, 'stdout_bytes': 4, 'stderr_bytes': 0,
                                                    'files_changed': 1, 'parse_error_count': 0}}
    class Tokenizer:
        def text_count(self, text):
            return len(text)
        async def count_messages_details(self, messages, tools):
            return {'token_count': sum(len(m.get('content', '')) for m in messages),
                    'tokenizer_fingerprint': {'sha256': 'd' * 64}, 'prompt_token_id_sha256': 'e' * 64}
    monkeypatch.setattr(panel, 'DockerRuntime', Runtime)
    monkeypatch.setattr(panel, '_audit', lambda *args: None)
    monkeypatch.setattr(panel, 'collect_public_evidence', collect)
    monkeypatch.setattr(panel, 'TokenizerClient', Tokenizer)
    return baseline, selection, observed, replay


def test_preparation_uses_last_dispatch_before_state_and_equal_full_prompts(tmp_path, prepared_case):
    baseline, selection, observed, _ = prepared_case
    prepared = panel.prepare_panel(baseline, selection, tmp_path / 'prepare', 'E2-S')
    assert observed['prior'] == json.loads(Path(baseline['tool_dispatch'][0]['before_path']).read_text())
    assert observed['raw'] == baseline['agent_result']['events'][0]['raw_result']
    assert set(prepared['rendered']) == {'original', 'target', 'matched', 'competing'}
    assert prepared['rendered']['target']['probe_id'] == 'visibility.compact'
    assert prepared['rendered']['competing']['probe_id'] == 'feedback.aggregate'
    tokens = prepared['token_audit']['full_prompt_tokens']
    assert tokens['visibility.compact'] == tokens['visibility.compact.matched']
    assert prepared['token_audit']['same_message_position'] == 3
    assert prepared['source_review']['checks']['unchanged_complete_workspace'] is True


def test_preparation_rejects_unlinked_result_before_docker(tmp_path, prepared_case, monkeypatch):
    baseline, selection, _, _ = prepared_case
    baseline['tool_dispatch'][0]['raw_result'] = {'exit_code': 9}
    monkeypatch.setattr(panel, 'DockerRuntime', lambda *a: pytest.fail('must not create actor'))
    with pytest.raises(ValueError, match='dispatch/result'):
        panel.prepare_panel(baseline, selection, tmp_path / 'prepare', 'E2-S')


def test_preparation_rejects_actual_full_prompt_token_mismatch(tmp_path, prepared_case, monkeypatch):
    baseline, selection, _, _ = prepared_case
    original = panel.TokenizerClient
    class WrongTokenizer(original):
        async def count_messages_details(self, messages, tools):
            result = await super().count_messages_details(messages, tools)
            if messages[-1]['role'] == 'user' and 'unreported' in messages[-1]['content']:
                result['token_count'] += 1
            return result
    monkeypatch.setattr(panel, 'TokenizerClient', WrongTokenizer)
    with pytest.raises(ValueError, match='full prompt costs differ'):
        panel.prepare_panel(baseline, selection, tmp_path / 'prepare', 'E2-S')


@pytest.fixture
def scheduler_case(tmp_path, prepared_case, monkeypatch):
    baseline, selection, _, replay = prepared_case
    prepared = panel.prepare_panel(baseline, selection, tmp_path / 'prepare', 'E2-S')
    calls = []
    class FixtureScheduler(ExperimentScheduler):
        # Only the real admission gate is replaced. Durable D5 logic and actual
        # task-bound registry signature/panel validation remain under test.
        def _validate(self, spec, contract):
            self.probe_authorizer(spec, contract)
    class Executor:
        def __init__(self, replay, root, baseline, manifest, *args):
            self.root = Path(root)
            self.root.mkdir()
        def __call__(self, spec):
            calls.append(spec)
            folder = self.root / spec.key
            folder.mkdir()
            completed = spec.probe_id != 'original'  # fresh Original is legitimately unstable
            flags = {'execution_valid': True, 'target_resolved': completed,
                     'regression_free': completed, 'task_completed': completed}
            result = {'outcome': flags, 'branch_key': spec.key, 'is_synthetic': False}
            cost = {'input_tokens': 100, 'output_tokens': 20, 'gpu_seconds': None,
                    'environment_seconds': 1.0, 'wall_seconds': 2.0}
            write(folder / 'result.json', result)
            write(folder / 'cost.json', cost)
            return ExecutionOutcome(result, cost)
    monkeypatch.setattr(panel, 'ExperimentScheduler', FixtureScheduler)
    monkeypatch.setattr(panel, 'RealBranchExecutor', Executor)
    manifest = {'tasks': {'task': {'input_files': [{'name': 'document.docx'}]}}}
    plan = {'agent_image': baseline['runtime']['image_digest'], 'grader_image': 'sha256:' + 'f' * 64,
            'exploration_seed': 2026, 'confirmation_seed': 2027,
            'candidate_order_sha256': '0' * 64, 'implementation_identity': '1' * 64,
            'physical_lock_path': str(tmp_path / 'shared.lock')}
    admission = {'replay': replay, 'authority': object()}
    return baseline, selection, manifest, plan, admission, prepared, calls


def test_eight_branch_panel_private_confirmation_logical_cache_and_instability(tmp_path, scheduler_case):
    baseline, selection, manifest, plan, admission, prepared, calls = scheduler_case
    root = tmp_path / 'run'
    summary = panel.run_success_panel(baseline, selection, manifest, plan, admission, prepared, root)
    assert len(calls) == 8
    assert [spec.phase for spec in calls[:4]] == ['exploration'] * 4
    assert len({spec.key for spec in calls}) == 8
    assert summary['physical_executions'] == 8 and summary['logical_charges'] == 5
    assert summary['unstable_success_baseline'] is True
    assert summary['physical_cost_total']['input_tokens'] == 800
    assert summary['physical_cost_total']['gpu_seconds'] is None
    assert summary['query_budget']['engineering-B-cache-audit']['spent'] == 1
    assert summary['query_budget']['engineering-A']['reserved'] == 0
    checks = {item['check']: item['passed'] for item in summary['checks']}
    assert checks['revoked_approval_denied_even_on_physical_cache'] is True
    assert checks['same_branch_one_physical_two_logical_charges'] is True
    assert checks['confirmation_absent_from_method_histories'] is True
    assert checks['confirmation_denied_while_B_open'] is True
    ledger = json.loads((root / 'scheduler/private/controller_ledger.json').read_text())
    assert len(ledger['physical']) == 8 and len(ledger['observations']) == 5
    assert all(row['charged'] == 1 for row in ledger['receipts'])
    assert len(ledger['predictions']) == 5
    for phase in ('exploration', 'confirmation'):
        index = json.loads((root / ('e2_s_' + phase) / 'index.json').read_text())
        assert len(index['branches']) == 4
        for ref in index['branches']:
            assert panel._sha(root / ref['result_path']) == ref['result_sha256']
    assert (root / 'scheduler/private/review.key').stat().st_mode & 0o777 == 0o600
    assert 'trusted_review_keys' not in (root / 'scheduler/branch_plan.json').read_text()
    assert summary['D6_D8'] == 'CLOSED'


def test_panel_rejects_preparation_drift_before_any_branch(tmp_path, scheduler_case):
    baseline, selection, manifest, plan, admission, prepared, calls = scheduler_case
    prepared['rendered_by_probe']['original']['rendered_text'] = 'answer material'
    with pytest.raises(ValueError, match='preparation review drift'):
        panel.run_success_panel(baseline, selection, manifest, plan, admission, prepared, tmp_path / 'run')
    assert calls == []


def test_panel_rejects_overlapping_phase_seed_domains(tmp_path, scheduler_case):
    baseline, selection, manifest, plan, admission, prepared, calls = scheduler_case
    plan['confirmation_seed'] = plan['exploration_seed']
    with pytest.raises(ValueError, match='disjoint phase seed'):
        panel.run_success_panel(baseline, selection, manifest, plan, admission, prepared, tmp_path / 'run')
    assert calls == []


def test_collector_workspace_mutation_rejected_before_rendering(tmp_path, prepared_case, monkeypatch):
    baseline, selection, _, _ = prepared_case
    collector = panel.collect_public_evidence
    def mutating_collector(actor, **kwargs):
        public = collector(actor, **kwargs)
        (actor.ws / 'document.docx').write_bytes(b'changed by collector')
        return public
    monkeypatch.setattr(panel, 'collect_public_evidence', mutating_collector)
    monkeypatch.setattr(panel, 'render_pair', lambda *args, **kwargs: pytest.fail('must not render changed workspace'))
    with pytest.raises(AssertionError):
        panel.prepare_panel(baseline, selection, tmp_path / 'prepare', 'E2-S')


def test_changed_frozen_source_rejected_before_branch_execution(tmp_path, scheduler_case):
    baseline, selection, manifest, plan, admission, prepared, calls = scheduler_case
    plan['implementation_lock'] = {'src/c3_sandbox/r1/r11_panel.py': '0' * 64}
    with pytest.raises(ValueError, match='frozen implementation changed'):
        panel.run_success_panel(baseline, selection, manifest, plan, admission, prepared, tmp_path / 'run')
    assert calls == []
