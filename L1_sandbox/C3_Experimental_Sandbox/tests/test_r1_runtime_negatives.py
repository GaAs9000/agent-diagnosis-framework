import json

import pytest

from c3_sandbox.r1 import runtime_negatives


class FakeRuntime:
    active = 0
    closed = []
    fail = None

    def __init__(self, image, run_dir):
        self.name = run_dir.name

    def __enter__(self):
        assert FakeRuntime.active == 0
        FakeRuntime.active += 1
        return self

    def __exit__(self, *args):
        FakeRuntime.active -= 1
        FakeRuntime.closed.append(self.name)

    def exec_python(self, code, timeout):
        assert timeout == 10
        if self.name == 'inaccessible_and_readonly':
            observed = {'reads': [{'read_succeeded': False, 'errno': 2}], 'rootfs_write': {'write_succeeded': False, 'errno': 30}}
        elif self.name == 'background_process':
            observed = {'spawned_pid': 42}
        else:
            observed = {'created': True}
        return {'exit_code': 0, 'stdout': json.dumps(observed), 'stderr': '', 'elapsed': 0, 'truncated': False, 'error': None}

    def export_workspace(self, destination):
        if FakeRuntime.fail != 'export':
            raise RuntimeError('workspace snapshot failed: symlink rejected')

    def audit_boundary(self):
        return {'qualified': self.name == 'inaccessible_and_readonly', 'effective_config_ok': FakeRuntime.fail != 'config',
                'unexpected_processes': [{'pid': 42}] if self.name == 'background_process' else [],
                'tmp_entries': ['/tmp/r1_transient_cache'] if self.name == 'tmp_cache' else []}


@pytest.fixture(autouse=True)
def fake_runtime(monkeypatch):
    FakeRuntime.active = 0
    FakeRuntime.closed = []
    FakeRuntime.fail = None
    monkeypatch.setattr(runtime_negatives, 'DockerRuntime', FakeRuntime)


def test_negative_acceptance_serial_cleanup_and_report(tmp_path):
    report = runtime_negatives.run_runtime_negatives('pinned-image', tmp_path)
    assert report['passed']
    assert len(report['cases']) == 4
    assert len(FakeRuntime.closed) == 4
    assert FakeRuntime.active == 0
    assert json.loads((tmp_path / 'runtime_negative_report.json').read_text()) == report
    assert (tmp_path / 'controller_private' / 'unmounted_canary.txt').stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('failure', ['export', 'config'])
def test_failures_persist_report_and_cleanup(tmp_path, failure):
    FakeRuntime.fail = failure
    with pytest.raises(RuntimeError, match='negative acceptance failed'):
        runtime_negatives.run_runtime_negatives('image', tmp_path)
    report = json.loads((tmp_path / 'runtime_negative_report.json').read_text())
    assert not report['passed']
    assert any(not case['passed'] for case in report['cases'])
    assert len(FakeRuntime.closed) == 4
    assert FakeRuntime.active == 0
