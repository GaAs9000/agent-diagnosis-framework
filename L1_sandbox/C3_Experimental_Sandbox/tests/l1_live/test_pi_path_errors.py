from __future__ import annotations

import copy
from pathlib import Path

import pytest

from c3_sandbox.l1.core.contracts import Call, ProtocolError
from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.runtime.tools import (
    execute_pi_edit,
    execute_pi_read,
    execute_pi_write,
    validate_call,
)
from .conftest import ScriptedGemini, gemini_call, gemini_done
from .test_pi_harness import PiFixtureRuntime


def arguments(tool, path):
    args = {'path': path}
    if tool == 'write':
        args['content'] = 'replacement'
    elif tool == 'edit':
        args['edits'] = [{'oldText': 'original', 'newText': 'replacement'}]
    return args


def execute(tool, path, workspace):
    call = Call('r0:c0', tool, arguments(tool, path), 'path-test')
    validate_call(call, harness='pi')
    if tool == 'read':
        return execute_pi_read(call, workspace, max_image_bytes=1024)
    if tool == 'write':
        return execute_pi_write(call, workspace)
    return execute_pi_edit(call, workspace)


@pytest.mark.parametrize('tool', ['read', 'write', 'edit'])
@pytest.mark.parametrize('path', [
    '/tmp/word/media/image1.png',
    '/tmp/docx_extract/word/document.xml',
    '/etc/passwd',
    '../outside.txt',
    '/workspace/../outside.txt',
    '/workspace',
    'link.txt',
    'linked-dir/outside.txt',
])
def test_rejected_paths_are_failed_observations_without_file_io(tmp_path, monkeypatch, tool, path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    outside = tmp_path / 'outside.txt'
    outside.write_text('original')
    (workspace / 'link.txt').symlink_to(outside)
    (workspace / 'linked-dir').symlink_to(tmp_path, target_is_directory=True)

    # Every rejected path must fail before any file content access or creation.
    def unexpected_io(*args, **kwargs):
        pytest.fail('rejected path reached file I/O')

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_bytes', unexpected_io)
        patch.setattr(Path, 'write_bytes', unexpected_io)
        patch.setattr(Path, 'mkdir', unexpected_io)
        result = execute(tool, path, workspace)

    assert result.observation['exit_code'] == 1
    assert result.observation['output_complete'] is True
    assert result.observation['stop_reason'] is None
    assert result.observation['stderr']
    assert result.stdout == b''
    assert result.stderr.decode() == result.observation['stderr']
    assert result.provenance['boundary'] == f'pi_workspace_{tool}'
    assert result.provenance['quiescent'] is True
    assert outside.read_text() == 'original'
    assert sorted(p.name for p in workspace.iterdir()) == ['link.txt', 'linked-dir']


@pytest.mark.parametrize('prefix', ['', '/workspace/'])
def test_workspace_operations_still_work_after_rejected_path(tmp_path, prefix):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    assert execute('write', '../outside.txt', workspace).observation['exit_code'] == 1
    path = prefix + 'nested/valid.txt'
    call = Call('r0:c1', 'write', {'path': path, 'content': 'original'}, 'valid-write')
    validate_call(call, harness='pi')
    assert execute_pi_write(call, workspace).observation['exit_code'] == 0
    assert execute('read', path, workspace).observation['output'] == 'original'
    assert execute('edit', path, workspace).observation['exit_code'] == 0
    assert execute('read', path, workspace).observation['output'] == 'replacement'


@pytest.mark.parametrize('edits', ['original', {'oldText': 'original', 'newText': 'replacement'}, []])
def test_malformed_edit_schema_remains_protocol_error(edits):
    call = Call('r0:c0', 'edit', {'path': '/tmp/outside.txt', 'edits': edits}, 'malformed-edit')
    with pytest.raises(ProtocolError):
        validate_call(call, harness='pi')


def test_episode_continues_after_rejected_pi_paths(tmp_path, config, components):
    config = copy.deepcopy(config)
    config['agent'] = {'name': 'pi-coding-agent', 'version': '0.73.1', 'harness': 'pi'}
    config['budget'].update(max_model_requests=5, max_tool_calls=4)
    responses = [
        gemini_call(tool, arguments(tool, '/tmp/word/media/image1.png'), native_id=f'bad-{tool}')
        for tool in ['read', 'write', 'edit']
    ] + [gemini_done()]
    model = ScriptedGemini(config['model'], responses)
    result = run(config, tmp_path / 'episode', injected={
        **components, 'model': model, 'runtime': PiFixtureRuntime(),
    })
    assert result['status']['termination'] == 'assistant_finished'
    assert result['verification']['verified'], result
    assert len(model.bodies) == 4
    for body in model.bodies[1:]:
        response = body['contents'][-1]['parts'][0]['functionResponse']['response']
        assert response['exit_code'] == 1
