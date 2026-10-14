"""No model/API use: compare cold compatibility to the pinned native daemon."""
from __future__ import annotations
import copy
import os
from pathlib import Path
import shutil
import subprocess
import pytest
from c3_sandbox.l1.core.config import DEFAULTS
from c3_sandbox.l1.core.contracts import Call
from c3_sandbox.l1.runtime.docker import DockerRuntime

BASE = 'sha256:b2a9f13cab4b017a58e2f9dd9abd1db475f52f17bae290e3dc464dfeb0668fb0'
COLD = 'l1-lhtb-sokoban-cold:0.2.1'
ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.docker, pytest.mark.skipif(
    os.environ.get('L1_DOCKER_TESTS') != '1', reason='Explicit real Docker opt-in required')]


def docker(*args):
    return subprocess.run(['docker', *args], capture_output=True, timeout=30)


def workspace(path):
    path.mkdir()
    os.chmod(path, 0o777)
    (path / 'input/bin').mkdir(parents=True)
    shutil.copy2(ROOT / 'benchmarks/lhtb/data/tasks/sokoban/bin/sokoban', path / 'input/bin/sokoban')
    return path


@pytest.fixture
def native_control(tmp_path):
    path = workspace(tmp_path / 'control')
    runtime = DockerRuntime(copy.deepcopy(DEFAULTS['runtime']))
    created = docker('create', *runtime._flags(), '--env', 'SOKOBAN_RUN_DIR=/workspace/runs',
                     '--env', 'SOKOBAN_DAEMON_LOG=/workspace/runs/daemon.log',
                     '--mount', f'type=bind,src={path},dst=/workspace',
                     '--workdir', '/workspace', '--entrypoint', 'sleep', BASE, '300')
    assert created.returncode == 0, created.stderr
    cid = created.stdout.decode().strip()
    assert docker('start', cid).returncode == 0
    try:
        yield path, cid
    finally:
        assert docker('rm', '-f', cid).returncode == 0


@pytest.fixture
def cold_runtime():
    config = copy.deepcopy(DEFAULTS['runtime'])
    config['image'] = COLD
    runtime = DockerRuntime(config)
    runtime.preflight()
    return runtime


def execute(runtime, path, command):
    result = runtime.execute(Call('native-test', 'run', {'command': command}, None), path, 30)
    assert result.provenance['quiescent'] and result.observation['output_complete']
    return result.observation['exit_code'], result.stdout, result.stderr


def assert_same(runtime, path, control, command):
    control_path, cid = control
    expected = docker('exec', cid, '/bin/sh', '-lc', command)
    actual = execute(runtime, path, command)
    assert actual == (expected.returncode, expected.stdout, expected.stderr)
    assert (path / 'runs/moves.log').read_bytes() == (control_path / 'runs/moves.log').read_bytes()
    assert (path / 'runs/meta.json').read_bytes() == (control_path / 'runs/meta.json').read_bytes()
    assert (path / 'submission/moves.log').read_bytes() == (control_path / 'runs/moves.log').read_bytes()


def test_cold_matches_native_cli_and_checkpoint(tmp_path, cold_runtime, native_control):
    path = workspace(tmp_path / 'cold')
    for command in ['sokoban status', '/workspace/input/bin/sokoban move D', 'sokoban move L',
                    '/workspace/input/bin/sokoban status', 'sokoban status']:
        assert_same(cold_runtime, path, native_control, command)
    checkpoint = tmp_path / 'checkpoint'
    shutil.copytree(path, checkpoint)
    branch = tmp_path / 'restored'
    shutil.copytree(checkpoint, branch)
    # Separate workspace copy must reconstruct changed state in a new daemon.
    for command in ['sokoban status', 'sokoban reset', '/workspace/input/bin/sokoban move D',
                    '/workspace/input/bin/sokoban move invalid',
                    '/workspace/input/bin/sokoban move ' + 'D' * 31,
                    '/workspace/input/bin/sokoban move ""', 'sokoban move',
                    'sokoban status', 'sokoban shutdown', 'sokoban status']:
        assert_same(cold_runtime, branch, native_control, command)
    assert (checkpoint / 'runs/moves.log').read_bytes() == b'D\nL\n'
    assert (path / 'runs/moves.log').read_bytes() == b'D\nL\n'


def test_wrapper_speculative_log_and_failed_preinvoke(tmp_path, cold_runtime):
    path = workspace(tmp_path / 'cold')
    assert execute(cold_runtime, path, 'sokoban move D')[0] == 0
    assert execute(cold_runtime, path, '/workspace/input/bin/sokoban move INVALID')[0] == 0
    assert (path / 'submission/moves.log').read_bytes() == b'D\n'
    # Simulate shell failure after wrapper-style preappend, before invoking CLI.
    assert execute(cold_runtime, path, 'printf "BAD\\n" >> submission/moves.log; false')[0] == 1
    assert (path / 'runs/moves.log').read_bytes() == b'D\n'
    assert (path / 'submission/moves.log').read_bytes() == b'D\nBAD\n'
    assert execute(cold_runtime, path, 'sokoban status')[0] == 0
    assert (path / 'submission/moves.log').read_bytes() == b'D\n'


def test_interrupted_replay_keeps_checkpoint_and_retries(tmp_path, cold_runtime, native_control):
    path = workspace(tmp_path / 'cold')
    control_path, cid = native_control
    # Accepted native commands only; no model, solver, or private scorer access.
    expected = docker('exec', cid, '/bin/sh', '-lc', 'for n in $(seq 1 10); do sokoban move DDDDDDDDDDDDDDDDDDDDDDDDDDDDDD >/dev/null; done; sokoban status')
    assert expected.returncode == 0
    (path / 'runs').mkdir()
    for name in ['moves.log', 'meta.json']:
        shutil.copy2(control_path / 'runs' / name, path / 'runs' / name)
    before = {name: (path / 'runs' / name).read_bytes() for name in ['moves.log', 'meta.json']}
    # Kill while daemon startup/replay is in progress. Persistent native files
    # must remain intact; transient native working files die with the container.
    interrupted = execute(cold_runtime, path, 'timeout 0.3 sokoban status')
    assert interrupted[0] == 124
    assert {name: (path / 'runs' / name).read_bytes() for name in before} == before
    assert_same(cold_runtime, path, native_control, 'sokoban status')
    snapshot = (path / 'runs/cold-state.json').read_bytes()
    interrupted = execute(cold_runtime, path, 'timeout 0.3 sokoban status')
    assert interrupted[0] == 124
    assert (path / 'runs/cold-state.json').read_bytes() == snapshot
    assert {name: (path / 'runs' / name).read_bytes() for name in before} == before
    assert_same(cold_runtime, path, native_control, 'sokoban status')



def test_same_container_interrupted_restore_rolls_back(tmp_path, cold_runtime, native_control):
    path = workspace(tmp_path / 'cold')
    control_path, cid = native_control
    seeded = docker('exec', cid, '/bin/sh', '-lc',
                    'for n in $(seq 1 10); do sokoban move DDDDDDDDDDDDDDDDDDDDDDDDDDDDDD >/dev/null; done')
    assert seeded.returncode == 0
    (path / 'runs').mkdir()
    for name in ['moves.log', 'meta.json']:
        shutil.copy2(control_path / 'runs' / name, path / 'runs' / name)
    expected = docker('exec', cid, 'sokoban', 'status')
    # timeout kills only the shim: the native daemon remains in this container.
    actual = execute(cold_runtime, path, 'timeout 0.3 sokoban status; sokoban status')
    assert actual == (expected.returncode, expected.stdout, expected.stderr)
    assert (path / 'runs/moves.log').read_bytes() == b'D\n' * 300
    assert_same(cold_runtime, path, native_control, 'sokoban status')
    # Also simulate a native command accepted before a killed shim could commit.
    # The unmodified CLI directly changes only the transient daemon working state.
    command = ('sokoban status >/dev/null; '
               'SOKOBAN_RUN_DIR=/tmp/sokoban-cold-native '
               'SOKOBAN_DAEMON_LOG=/tmp/sokoban-cold-native/daemon.log '
               '/usr/local/bin/sokoban-native move L >/dev/null; sokoban status')
    actual = execute(cold_runtime, path, command)
    assert actual == (expected.returncode, expected.stdout, expected.stderr)
    assert (path / 'runs/moves.log').read_bytes() == b'D\n' * 300
    assert (path / 'submission/moves.log').read_bytes() == b'D\n' * 300
