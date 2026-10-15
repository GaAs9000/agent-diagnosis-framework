"""Selected installed actors are execution dependencies, including registry identity."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from l3_sandbox import plan
from l3_sandbox.common import L3Error, read, sha_file, write
from test_plan_conditions import prepared, REPO


@pytest.fixture
def plugin_plan(prepared, tmp_path, monkeypatch):
    from c3_sandbox.l1.models.opencode import OpenCodeChatBackend
    path, spec, _ = prepared
    template = REPO / 'L1_sandbox/C3_Experimental_Sandbox/configs/l1/officeval.opencode-chat.json'
    spec['base_config'] = str(template)
    write(path, spec)
    entry = SimpleNamespace(name='opencode-chat', group='l1.models',
        value='c3_sandbox.l1.models.opencode:OpenCodeChatBackend',
        dist=SimpleNamespace(metadata={'Name': 'c3-experimental-sandbox'}, version='0.2.0'),
        load=lambda: OpenCodeChatBackend)
    original_entries = plan.metadata.entry_points
    monkeypatch.setattr(plan.metadata, 'entry_points', lambda *, group:
        [entry] if group == 'l1.models' else original_entries(group=group))
    directory = tmp_path / 'bound'
    frozen = plan.create_plan(path, directory)
    return directory, frozen, entry


def test_selected_plugin_freezes_actual_module_and_resolution(plugin_plan):
    directory, frozen, entry = plugin_plan
    actor = frozen['selected_actor_binding']
    assert actor['file'].endswith('/models/opencode.py')
    assert actor['file_sha256'] == sha_file(Path(actor['file']))
    assert frozen['source_sha256'][actor['file']] == actor['file_sha256']
    assert actor['module'] == 'c3_sandbox.l1.models.opencode'
    assert actor['qualname'] == 'OpenCodeChatBackend'
    assert actor['adapter'] == 'opencode-chat'
    assert actor['model'] == 'kimi-k2.6'
    assert actor['entry_point']['value'] == entry.value
    assert actor['entry_point']['distribution'] == {'name': 'c3-experimental-sandbox', 'version': '0.2.0'}
    assert plan.verify_plan(directory, verify_images=False) == frozen


@pytest.mark.parametrize('change', ['file', 'entrypoint', 'distribution', 'version', 'class', 'module'])
def test_selected_actor_drift_blocks_execution_and_allows_read_only_history(plugin_plan, monkeypatch, change):
    directory, frozen, entry = plugin_plan
    original_bytes = (directory / 'plan.json').read_bytes()
    actor_path = frozen['selected_actor_binding']['file']
    if change == 'file':
        original_hash = plan.sha_file
        monkeypatch.setattr(plan, 'sha_file', lambda path:
            'changed-actual-plugin-file' if str(path) == actor_path else original_hash(path))
    elif change == 'entrypoint':
        entry.value = 'alias.remapped:SameClass'
    elif change == 'distribution':
        entry.dist.metadata['Name'] = 'replacement-distribution'
    elif change == 'version':
        entry.dist.version = '99.0'
    elif change == 'class':
        from c3_sandbox.l1.models.opencode import OpenAIResponsesBackend
        entry.load = lambda: OpenAIResponsesBackend
    else:
        from c3_sandbox.l1.models.native import GeminiBackend
        entry.load = lambda: GeminiBackend
    with pytest.raises(L3Error, match='Selected actor binding changed'):
        plan.verify_plan(directory, verify_images=False)
    assert plan.verify_plan(directory, verify_images=False, verify_execution=False) == frozen
    assert (directory / 'plan.json').read_bytes() == original_bytes


@pytest.mark.parametrize('change', ['binding', 'legacy'])
def test_reuse_rejects_changed_or_missing_actor_binding(plugin_plan, change):
    _, source, _ = plugin_plan
    target = copy.deepcopy(source)
    job = source['jobs'][0]
    if change == 'binding':
        target['selected_actor_binding']['entry_point']['value'] = 'remapped:Backend'
    else:
        source.pop('selected_actor_binding')
    with pytest.raises(L3Error, match='Reused selected actor binding differs'):
        plan._verify_reuse_contract(source, target, job, job)
    if change == 'legacy':
        target.pop('selected_actor_binding')
        with pytest.raises(L3Error, match='Reused selected actor binding differs'):
            plan._verify_reuse_contract(source, target, job, job)


def test_legacy_opencode_plan_cannot_execute(plugin_plan):
    directory, frozen, _ = plugin_plan
    frozen.pop('selected_actor_binding')
    frozen['source_scope'] = 'execution-v1'
    write(directory / 'plan.json', frozen)
    (directory / 'plan.sha256').write_text(sha_file(directory / 'plan.json'))
    with pytest.raises(L3Error, match='requires a frozen selected actor binding'):
        plan.verify_plan(directory, verify_images=False)
    assert plan.verify_plan(directory, verify_images=False, verify_execution=False) == frozen


def test_native_gemini_ignores_unselected_opencode_and_analysis_changes(prepared, tmp_path, monkeypatch):
    path, _, _ = prepared
    # Use the real inventory to exercise the opencode exclusion, not the fixture's stub.
    monkeypatch.setattr(plan, 'source_inventory', lambda: plan._execution_sources(plan.provenance_inventory()))
    directory = tmp_path / 'native'
    frozen = plan.create_plan(path, directory)
    original = plan.provenance_inventory
    def revised():
        return {name: ('unselected-or-analysis-v2' if name.endswith(('/models/opencode.py', '/report.py',
                      '/evaluation/study.py')) else value) for name, value in original().items()}
    monkeypatch.setattr(plan, 'provenance_inventory', revised)
    assert plan.verify_plan(directory, verify_images=False) == frozen


@pytest.mark.parametrize('section,adapter', [('runtime', 'plugin-runtime'), ('evaluation', 'plugin-evaluator')])
def test_model_plugin_opt_in_cannot_select_external_execution_components(prepared, tmp_path, section, adapter):
    path, spec, _ = prepared
    from c3_sandbox.l1.core.config import resolve
    template = REPO / 'L1_sandbox/C3_Experimental_Sandbox/configs/l1/officeval.opencode-chat.json'
    config = resolve(template)
    config[section]['adapter'] = adapter
    local_template = tmp_path / 'template.json'
    write(local_template, config)
    spec['base_config'] = str(local_template)
    write(path, spec)
    with pytest.raises(L3Error, match='require native OfficeVal runtime/evaluation'):
        plan.create_plan(path, tmp_path / 'rejected')
