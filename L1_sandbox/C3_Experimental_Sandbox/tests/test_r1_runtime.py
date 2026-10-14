import copy
import os
from pathlib import Path

import pytest

from c3_sandbox.r1.docker_runtime import restore_snapshot, snapshot_workspace, validate_snapshot


def test_snapshot_roundtrip_hidden_empty_and_modes(tmp_path):
    source = tmp_path / 'source'
    source.mkdir(mode=0o750)
    (source / '.git').mkdir(mode=0o750)
    (source / '.git' / 'config').write_bytes(b'\x00\xff\n')
    (source / '.git' / 'config').chmod(0o640)
    (source / 'empty').mkdir(mode=0o700)
    snapshot = snapshot_workspace(source)
    destination = tmp_path / 'restored'
    restore_snapshot(snapshot, destination)
    assert snapshot_workspace(destination) == snapshot
    (destination / '.git' / 'config').unlink()
    assert snapshot_workspace(destination) != snapshot


@pytest.mark.parametrize('name', ['../escape', '/absolute', '.', 'a//b', 'a/../b', 'x\x00y'])
def test_reject_unsafe_paths(tmp_path, name):
    snapshot = {'version': 1, 'root_mode': 0o700, 'total_bytes': 0,
                'entries': [{'path': name, 'kind': 'directory', 'mode': 0o700}]}
    with pytest.raises(ValueError):
        restore_snapshot(snapshot, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_reject_symlinks_and_hardlinks(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'file').write_text('content')
    (root / 'link').symlink_to(root / 'file')
    with pytest.raises(ValueError):
        snapshot_workspace(root)
    (root / 'link').unlink()
    os.link(root / 'file', root / 'hardlink')
    with pytest.raises(ValueError):
        snapshot_workspace(root)


def test_reject_fifo(tmp_path):
    os.mkfifo(tmp_path / 'fifo')
    with pytest.raises(ValueError):
        snapshot_workspace(tmp_path)


def test_reject_invalid_snapshot_and_nonempty_destination(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'f').write_bytes(b'abc')
    original = snapshot_workspace(source)
    for mutate in [lambda s: s.update(total_bytes=0),
                   lambda s: s['entries'].append(copy.deepcopy(s['entries'][0])),
                   lambda s: s['entries'][0].update(data='!'),
                   lambda s: s['entries'][0].update(mode=-1),
                   lambda s: s['entries'][0].update(path='missing/f')]:
        broken = copy.deepcopy(original)
        mutate(broken)
        with pytest.raises(ValueError):
            validate_snapshot(broken)
    with pytest.raises(ValueError):
        restore_snapshot(original, source)


def test_bound_bytes_and_paths(tmp_path, monkeypatch):
    import c3_sandbox.r1.docker_runtime as runtime
    (tmp_path / 'f').write_bytes(b'1234')
    monkeypatch.setattr(runtime, 'MAX_WORKSPACE_BYTES', 3)
    with pytest.raises(ValueError, match='byte limit'):
        snapshot_workspace(tmp_path)
    monkeypatch.setattr(runtime, 'MAX_WORKSPACE_BYTES', 10)
    monkeypatch.setattr(runtime, 'MAX_PATHS', 0)
    with pytest.raises(ValueError, match='path limit'):
        snapshot_workspace(tmp_path)


def _mount_observation():
    inspected = {'HostConfig': {'Binds': None, 'Tmpfs': {'/workspace': 'intent', '/tmp': 'intent'}}, 'Mounts': []}
    state = {'mountinfo': '1 0 0:1 / /workspace rw,nosuid,nodev - tmpfs tmpfs rw,uid=1000,gid=1000\n2 0 0:2 / /tmp rw,nosuid,nodev,noexec - tmpfs tmpfs rw,uid=1000,gid=1000\n',
             'mount_stats': {p: {'uid': 1000, 'gid': 1000, 'mode': 0o700} for p in ('/workspace', '/tmp')}}
    return inspected, state


def test_kernel_mounts_required_despite_docker_declaration():
    from c3_sandbox.r1.docker_runtime import boundary_mounts_valid
    inspected, state = _mount_observation()
    assert boundary_mounts_valid(inspected, state)
    assert not boundary_mounts_valid(inspected, {'mountinfo': '', 'mount_stats': state['mount_stats']})
    state['mountinfo'] = state['mountinfo'].replace('- tmpfs tmpfs', '- ext4 /dev/fake', 1)
    assert not boundary_mounts_valid(inspected, state)


@pytest.mark.parametrize('missing', ['rw', 'nosuid', 'nodev', 'noexec'])
def test_mount_options_fail_closed(missing):
    from c3_sandbox.r1.docker_runtime import boundary_mounts_valid
    inspected, state = _mount_observation()
    state['mountinfo'] = state['mountinfo'].replace(missing, 'absent')
    assert not boundary_mounts_valid(inspected, state)


def test_mount_ownership_nested_mount_and_host_bind_rejected():
    from c3_sandbox.r1.docker_runtime import boundary_mounts_valid
    inspected, state = _mount_observation()
    for field, value in [('uid', 0), ('gid', 0), ('mode', 0o777)]:
        changed = copy.deepcopy(state)
        changed['mount_stats']['/tmp'][field] = value
        assert not boundary_mounts_valid(inspected, changed)
    changed = copy.deepcopy(state)
    changed['mountinfo'] += '3 1 0:3 / /workspace/subdir rw - tmpfs tmpfs rw\n'
    assert not boundary_mounts_valid(inspected, changed)
    inspected['Mounts'] = [{'Type': 'bind', 'Destination': '/workspace'}]
    assert not boundary_mounts_valid(inspected, state)
    inspected['Mounts'] = []
    inspected['HostConfig']['Binds'] = ['/host:/workspace']
    assert not boundary_mounts_valid(inspected, state)
