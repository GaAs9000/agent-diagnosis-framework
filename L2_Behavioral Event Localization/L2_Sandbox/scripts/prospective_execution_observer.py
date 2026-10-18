#!/usr/bin/env python3
"""Fail-closed prospective Docker observer; never launches a container or model.

Receipts attest consistency under the reviewed ordinary runner/OS/Docker trust
root. They do not turn historical provenance or statistical equivalence into PASS.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import importlib.util
import time
import urllib.parse
import shutil

L2 = Path(__file__).resolve().parents[1]
ROOT = L2.parents[1]
L1 = ROOT / 'L1_sandbox/C3_Experimental_Sandbox'
sys.path[:0] = [str(L1 / 'src')]
from c3_sandbox.l1.storage import canonical, digest

DEV = tuple('officeval_' + n for n in ('003', '004', '010', '033', '050', '060', '070', '085', '090'))
PROBE_SCRIPT = "import sys,json,importlib.metadata as m,hashlib,pathlib; p=pathlib.Path('/opt/l1/grade_worker.py'); print(json.dumps({'python':sys.version,'packages':{x.metadata['Name']:x.version for x in m.distributions()},'grader_worker_sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None,'has_app_git':pathlib.Path('/app/.git').exists()},sort_keys=True))"
RUNTIME_ENV = {'HOME': '/tmp', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
               'OMEGAUSE_OFFICEVAL_DISABLE_COM': '1'}
_HASH_LOCK = threading.RLock()
_HASH_CACHE = {}
_CAPTURE_CACHE = set()


def regular(path):
    path = Path(path).absolute()
    if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():
        raise ValueError('unaliased regular evidence file required: ' + str(path))
    return path


def sha(path):
    path = regular(path)
    def identity(st):
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
    with _HASH_LOCK:
        before = identity(path.stat())
        prior = _HASH_CACHE.get(str(path))
        # Do not cache newly written files within coarse timestamp resolution.
        # The large immutable archives are old by the time collection begins.
        stable_before = time.time_ns() - 2_000_000_000
        if prior is not None and prior[0] == before and max(before[3], before[4]) < stable_before:
            return prior[1]
        value = hashlib.sha256()
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'rb') as stream:
            if identity(os.fstat(stream.fileno())) != before:
                raise ValueError('evidence file changed before hashing')
            for chunk in iter(lambda: stream.read(1048576), b''):
                value.update(chunk)
            if identity(os.fstat(stream.fileno())) != before or identity(path.stat()) != before:
                raise ValueError('evidence file changed while hashing')
        result = value.hexdigest()
        _HASH_CACHE[str(path)] = before, result
        return result


def read(path):
    with regular(path).open('r', encoding='utf-8') as stream:
        return json.load(stream)


def operational_sources():
    paths = [Path(__file__), L2 / 'scripts/run_prospective_dev45.py', L2 / 'src/l2_sandbox/v3_execution.py']
    paths += [p for p in (L1 / 'src').rglob('*') if p.is_file()
              and p.suffix in ('.py', '.json', '.mjs', '.js')
              and '__pycache__' not in p.parts and 'node_modules' not in p.parts]
    paths += list((L2 / 'scripts').glob('*.py')) + list((ROOT / 'scripts').glob('*.py'))
    return {str(regular(p).relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def bound_reference(frozen, ref):
    if not isinstance(ref, dict) or not isinstance(ref.get('path'), str) or not Path(ref['path']).is_absolute():
        raise ValueError('absolute hash-bound evidence reference required')
    path = regular(ref['path'])
    expected = ref.get('sha256')
    if frozen['bound_files'].get(str(path)) != expected or sha(path) != expected:
        raise ValueError('nested evidence reference missing or changed: ' + str(path))
    return path


def nested_path(base, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('proof stage path escapes evidence root')
    return regular(base / relative)


def validate_artifact_graph(frozen, candidate_path, candidate, verification_path):
    """Require an exhaustive graph retained after independent CPU verification."""
    graph = read(bound_reference(frozen, frozen.get('verified_artifacts')))
    if (graph.get('schema') != 'prospective-verified-artifacts-v1'
            or graph.get('candidate_sha256') != sha(candidate_path)
            or graph.get('verification_sha256') != sha(verification_path)):
        raise ValueError('independent exhaustive artifact manifest binding differs')
    base = Path(graph.get('evidence_root', ''))
    if (not base.is_absolute() or not base.is_dir() or base.is_symlink()
            or any(p.is_symlink() for p in base.parents)):
        raise ValueError('absolute unaliased common proof root required')
    anchor_path = bound_reference(frozen, graph.get('external_anchor'))
    allowlist_path = bound_reference(frozen, graph.get('reviewed_allowlist'))
    anchor, allowlist = read(anchor_path), read(allowlist_path)
    if (anchor.get('approved') is not True or anchor.get('candidate_sha256') != sha(candidate_path)
            or anchor.get('reviewed_allowlist_sha256') != sha(allowlist_path)
            or anchor_path.is_relative_to(candidate_path.parent)
            or allowlist.get('approved') is not True):
        raise ValueError('external candidate approval and reviewed allowlist links differ')
    expected = {str(p): sha(p) for p in (candidate_path, verification_path, anchor_path, allowlist_path)}
    stages = []
    for ref in candidate.get('stages', []):
        path = nested_path(base, ref['path'])
        if sha(path) != ref.get('sha256'):
            raise ValueError('candidate stage manifest changed')
        stage = read(path)
        if stage.get('schema') != 'prospective-provenance-stage-v1' or stage.get('kind') != ref.get('kind'):
            raise ValueError('stage schema/kind differs from candidate')
        if not stage.get('files'):
            raise ValueError('stage artifact inventory is empty')
        expected[str(path)] = sha(path)
        for relative, expected_hash in stage['files'].items():
            item = nested_path(path.parent, relative)
            if sha(item) != expected_hash:
                raise ValueError('proof stage file differs: ' + str(item))
            expected[str(item)] = expected_hash
        stages.append((path, stage))
    required = {'controller_and_inputs', 'public_downloads', 'image_capture', 'image_inventory'}
    if not required <= {stage['kind'] for _, stage in stages}:
        raise ValueError('full controller/public/capture/inventory stages required')
    described = graph.get('stage_manifests', [])
    actual = [{'path': str(path), 'sha256': sha(path), 'kind': stage['kind']} for path, stage in stages]
    key = lambda row: (row['path'], row['sha256'], row['kind'])
    if (len({str(path) for path, _ in stages}) != len(stages)
            or sorted(map(key, described)) != sorted(map(key, actual))
            or graph.get('files') != expected
            or any(frozen['bound_files'].get(path) != value for path, value in expected.items())):
        raise ValueError('exhaustive artifact graph omits or changes nested proof files')
    preparation = [(p, s) for p, s in stages if s['kind'] == 'controller_and_inputs']
    downloads = [(p, s) for p, s in stages if s['kind'] == 'public_downloads']
    if len(preparation) != 1 or len(downloads) != 1:
        raise ValueError('exactly one controller/public proof stage required')
    prep_path, prep = preparation[0]
    if not {'controller.json', 'controller_source.tar', 'public_inputs.json', 'resolved_config.json'} <= set(prep['files']):
        raise ValueError('controller capture is incomplete')
    controller = read(prep_path.parent / 'controller.json')
    if (not controller.get('source_files')
            or controller.get('source_manifest_sha256') != digest(controller['source_files'])
            or any(frozen['source_files'].get(name) != value for name, value in controller['source_files'].items())):
        raise ValueError('archived controller identity differs from actual frozen sources')
    public_path = prep_path.parent / 'public_inputs.json'
    public = read(public_path)
    if set(public.get('tasks', {})) != set(DEV):
        raise ValueError('public proof must contain exact DEV9')
    download_path, download = downloads[0]
    if download.get('payload', {}).get('public_inputs_sha256') != sha(public_path):
        raise ValueError('public download proof uses another input lock')
    expected_downloads = {}
    for task, record in public['tasks'].items():
        prefix = public['dataset_repository'] + '/resolve/' + public['dataset_revision'] + '/'
        expected_downloads[prefix + 'tasks/' + task + '.json'] = record['task_sha256']
        for asset in record['assets']:
            expected_downloads[prefix + 'task_files/' + task + '/' + urllib.parse.quote(asset['name'], safe='')] = asset['sha256']
    if 'download_receipts.json' not in download['files']:
        raise ValueError('raw public download receipts missing')
    receipts = read(download_path.parent / 'download_receipts.json')
    if ({r['url']: r['sha256'] for r in receipts} != expected_downloads or len(receipts) != len(expected_downloads)
            or any(sha(nested_path(download_path.parent, r['file'])) != r['sha256'] for r in receipts)):
        raise ValueError('raw public bytes/URLs do not cover pinned DEV9 exactly')
    captures = {}
    for path, stage in stages:
        if stage['kind'] != 'image_capture':
            continue
        # Pure receipt validation once per unchanged stage. All proof files above
        # are independently hash/stat-checked on every callback before this cache.
        cache_key = (str(path), sha(path), digest(stage['files']))
        with _HASH_LOCK:
            if cache_key not in _CAPTURE_CACHE:
                spec = importlib.util.spec_from_file_location('prospective_capture_receipt_validator',
                        Path(__file__).with_name('build_prospective_provenance.py'))
                builder = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(builder)
                builder.validate_capture(path, stage, time.monotonic() + 120)
                _CAPTURE_CACHE.add(cache_key)
        captures[stage['payload']['image_id']] = (path, stage)
    observed_images = set()
    for path, stage in stages:
        if stage['kind'] != 'image_inventory':
            continue
        payload = stage.get('payload', {})
        image_id = payload.get('image_id')
        if image_id not in captures or not {'rootfs_inventory.json'} <= set(stage['files']):
            raise ValueError('image inventory lacks its actual capture')
        capture_path, captured = captures[image_id]
        approved = allowlist.get('images', {}).get(image_id, {})
        inventory_path = path.parent / 'rootfs_inventory.json'
        inventory = read(inventory_path)
        if (payload.get('capture_manifest_sha256') != sha(capture_path)
                or payload.get('public_inputs_sha256') != sha(public_path)
                or payload.get('rootfs_inventory_sha256') != sha(inventory_path)
                or inventory.get('archive_sha256') != captured['payload']['rootfs_archive_sha256']
                or approved.get('rootfs_inventory_sha256') != sha(inventory_path)
                or approved.get('all_entries_reviewed') is not True or not approved.get('content_policy')
                or (image_id == frozen['actor_image_id'] and inventory.get('known_private_or_task_file_matches') != [])):
            raise ValueError('reviewed full rootfs inventory/capture/public links differ')
        observed_images.add(image_id)
    verification = read(verification_path)
    if not set(verification.get('resolved_images', {}).values()) <= observed_images or frozen['actor_image_id'] not in observed_images:
        raise ValueError('actor/grader images lack reviewed capture and archive inventory')
    actor_capture_path = captures[frozen['actor_image_id']][0].parent / 'image_inspect.json'
    if str(actor_capture_path) != frozen['actor_image_inspect']['path'] or sha(actor_capture_path) != frozen['actor_image_inspect']['sha256']:
        raise ValueError('frozen Actor image inspect is outside the validated capture graph')


def env_dict(values):
    if not isinstance(values, list) or any(not isinstance(v, str) or '=' not in v for v in values):
        raise ValueError('Docker environment must be an explicit string list')
    result = {}
    for value in values:
        name, content = value.split('=', 1)
        if not name or name in result or '\x00' in value:
            raise ValueError('duplicate/invalid Docker environment name')
        result[name] = content
    return result


def host_user():
    return f'{os.getuid()}:{os.getgid()}' if os.getuid() != 0 else '65532:65532'


def validate_native_launch(frozen):
    native = frozen.get('native_execution') or {}
    selected_node = os.environ.get('C3_PI_NODE') or shutil.which('node')
    if not selected_node:
        raise ValueError('native Node launch selection missing')
    node = Path(shutil.which(selected_node) or selected_node).resolve(strict=True)
    bridge = Path(os.environ.get('C3_PI_NATIVE_DIR') or L1 / 'src/c3_sandbox/l1/runtime/pi_native').resolve(strict=True)
    actual_files = {p.name: sha(p) for p in sorted(bridge.iterdir()) if p.is_file()}
    expected = native.get('dependency_identity') or {}
    if (native.get('node_path') != str(node) or native.get('node_sha256') != sha(node)
            or native.get('bridge_path') != str(bridge) or native.get('bridge_files') != actual_files
            or expected.get('version') != '0.73.1' or expected.get('actor_requests') != 0
            or expected.get('approved_modules_verified') is not True or not expected.get('installed_tree_sha256')):
        raise ValueError('selected native launch differs from frozen Node/bridge/installed identity')
    selected_files = {str(node): sha(node), **{str(bridge / name): value for name, value in actual_files.items()}}
    if any(frozen['bound_files'].get(name) != value for name, value in selected_files.items()):
        raise ValueError('actual native launch files are not externally hash-bound')


def validate_freeze(freeze_path, expected_sha):
    path = regular(freeze_path)
    if sha(path) != expected_sha:
        raise ValueError('prospective freeze SHA mismatch')
    frozen = read(path)
    if (frozen.get('schema') != 'prospective-live-freeze-v1'
            or frozen.get('phase_authorized') is not True or frozen.get('internal_reviewed') is not True
            or frozen.get('d0_frozen') is not False or frozen.get('provenance_candidate_verified') is not True):
        raise ValueError('reviewed prospective execution freeze required; D0 remains unfrozen')
    if (set(frozen.get('task_allowed', [])) != set(DEV)
            or len(frozen.get('task_allowed', [])) != 9
            or set(frozen.get('task_config_sha256', {})) != set(DEV)):
        raise ValueError('exact DEV9 task/config lock required')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', frozen.get('actor_image_id', '')):
        raise ValueError('immutable Actor image ID required')
    if not re.fullmatch(r'\d+:\d+', frozen.get('actor_user', '')) or frozen['actor_user'] != host_user():
        raise ValueError('explicit frozen host UID:GID must match actual runtime host identity')
    env_dict(frozen.get('actor_image_env'))
    sources = frozen.get('source_files')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('operational source inventory missing')
    for name, expected in sources.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha(ROOT / relative) != expected:
            raise ValueError('frozen controller source changed: ' + name)
    required = operational_sources()
    if any(sources.get(name) != expected for name, expected in required.items()):
        raise ValueError('freeze omits actual driver/observer/L1 operational source')
    files = frozen.get('bound_files')
    if not isinstance(files, dict) or not files:
        raise ValueError('external evidence file lock missing')
    for name, expected in files.items():
        if not Path(name).is_absolute() or sha(name) != expected:
            raise ValueError('external evidence file changed: ' + name)
    validate_native_launch(frozen)
    image = read(bound_reference(frozen, frozen.get('actor_image_inspect')))
    if image.get('Id') != frozen['actor_image_id'] or image.get('Config', {}).get('Env') != frozen['actor_image_env']:
        raise ValueError('allowed Actor image environment differs from actual frozen image inspection')
    candidate_path = bound_reference(frozen, frozen.get('candidate'))
    candidate = read(candidate_path)
    if (candidate.get('schema') != 'prospective-provenance-candidate-v1'
            or candidate.get('actor_pool_frozen') is not False or candidate.get('g4_verified') is not False):
        raise ValueError('prospective candidate identity/state mismatch')
    verification_path = bound_reference(frozen, frozen.get('provenance_verification'))
    verification = read(verification_path)
    if (verification.get('preparation_links_validated') is not True
            or verification.get('candidate_sha256') != sha(candidate_path)
            or verification.get('g4_verified') is not False
            or verification.get('resolved_images', {}).get('actor') != frozen['actor_image_id']):
        raise ValueError('candidate preparation verification differs; future live links remain unverified')
    validate_artifact_graph(frozen, candidate_path, candidate, verification_path)
    mini = frozen.get('mechanism_mini') or {}
    if mini.get('reviewed') is not True:
        raise ValueError('mechanism mini review missing')
    mini_path = bound_reference(frozen, {'path': mini.get('result_path'), 'sha256': mini.get('result_sha256')})
    result = read(mini_path)
    rows = result.get('cases', [])
    if (result.get('schema') != 'recovery_mechanism_mini_result_v1'
            or result.get('actor_requests') != 0 or result.get('analyst_requests') != 0
            or result.get('tool_attempts') != 5 or len(rows) != 5
            or len({row.get('case') for row in rows}) != 5
            or any(row.get('restore_exact') is not True or row.get('request_exact') is not True
                   or row.get('unexplained_state_difference') is not False
                   or row.get('unexplained_tool_difference') is not False for row in rows)):
        raise ValueError('mechanism mini has missing coverage or unexplained differences')
    review_path = bound_reference(frozen, frozen.get('mini_review'))
    review = read(review_path)
    authorization = read(bound_reference(frozen, review.get('authorization')))
    if (authorization.get('authorization_source') != 'direct_user_reply'
            or not authorization.get('exact_text')
            or authorization.get('policy') != 'bounded-mechanism-contract-v1'
            or authorization.get('allow_unexplained_differences') is not False):
        raise ValueError('direct user bounded-contract authorization missing')
    variances = [row['case'] for row in rows if row.get('declared_runtime_variance') is True]
    if (review.get('approved') is not True or review.get('policy') != 'bounded-mechanism-contract-v1'
            or review.get('result_sha256') != sha(mini_path)
            or ('approved_declared_variance_cases' in review
                and set(review['approved_declared_variance_cases']) != set(variances))
            or (variances and review.get('allow_declared_runtime_variance') is not True)
            or review.get('unexplained_differences_allowed') is not False
            or review.get('full_tool_equivalence_claim') is not False
            or review.get('outcome_distribution_equivalence_claim') is not False
            or any(row.get('deterministic_tool_exact') is not True
                   and row.get('declared_runtime_variance') is not True for row in rows)):
        raise ValueError('explicit bounded mini contract review does not cover actual differences')
    return frozen


def write_exclusive(path, value):
    """Success returns only after the receipt and its directory are durable."""
    path = Path(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {'path': str(path), 'sha256': sha(path)}


class ProspectiveExecutionObserver:
    def __init__(self, freeze_path, expected_sha, external_dir):
        self.freeze_path = regular(freeze_path)
        self.expected_sha = expected_sha
        self.external_dir = Path(external_dir).absolute()
        if (self.external_dir.exists() or self.external_dir.is_symlink()
                or any(p.is_symlink() for p in self.external_dir.parents)
                or self.external_dir.is_relative_to(L1 / 'runs')
                or self.external_dir.is_relative_to(L1 / 'src')
                or self.external_dir.is_relative_to(L2 / 'scripts')
                or self.external_dir.is_relative_to(ROOT / '.git')
                or (self.external_dir.is_relative_to(L2 / 'validation')
                    and not self.external_dir.is_relative_to(L2 / 'validation/a_plus_i_execute_20261018'))
                or any((p / 'evidence_index.json').exists() or (p / 'trajectory.jsonl').exists()
                       for p in self.external_dir.parents)):
            raise ValueError('external receipt directory must be new, unaliased and outside execution/source trees')
        self.frozen = validate_freeze(self.freeze_path, expected_sha)
        self.config = None
        self.workspace = None
        self.episode_id = None
        self.seen_containers = set()
        self.native_verified = False

    def guard(self):
        checked = validate_freeze(self.freeze_path, self.expected_sha)
        if checked != self.frozen:
            raise ValueError('prospective freeze changed during execution')

    def validate(self, config):
        self.guard()
        if self.config is not None or self.episode_id is not None:
            raise ValueError('observer is one episode only')
        task = config.get('benchmark', {}).get('task_id')
        if task not in DEV or digest(config) != self.frozen['task_config_sha256'].get(task):
            raise ValueError('final resolved task configuration differs from freeze')
        runtime = config.get('runtime', {})
        if (runtime.get('adapter') != 'docker-cold' or runtime.get('memory') != '2g'
                or runtime.get('cpus') != 2 or runtime.get('pids_limit') != 128
                or runtime.get('read_only_rootfs', True) is not True
                or runtime.get('tmpfs_exec', False) is not False
                or runtime.get('options', {}).get('persist_app_repo', False) is not False
                or runtime.get('user') not in (None, self.frozen['actor_user'])):
            raise ValueError('resolved runtime violates frozen isolation/resources')
        self.config = copy.deepcopy(config)

    def bind_workspace(self, path):
        self.guard()
        value = Path(path).absolute()
        if (self.config is None or self.workspace is not None or value.is_symlink()
                or any(p.is_symlink() for p in value.parents) or not value.is_dir()
                or self.external_dir.is_relative_to(value) or value.is_relative_to(self.external_dir)):
            raise ValueError('one unaliased run workspace required, disjoint from receipts')
        self.workspace = value

    def bind_episode(self, episode_id, config_sha):
        self.guard()
        if (self.config is None or self.workspace is None or self.episode_id is not None
                or not re.fullmatch(r'ep_[a-zA-Z0-9_-]+', episode_id)
                or config_sha != digest(self.config)):
            raise ValueError('episode/config/workspace binding mismatch')
        self.external_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.episode_id = episode_id
        binding = {'schema': 'prospective-episode-binding-v1', 'episode_id': episode_id,
                   'task_id': self.config['benchmark']['task_id'], 'config_sha256': config_sha,
                   'freeze': {'path': str(self.freeze_path), 'sha256': self.expected_sha},
                   'workspace': str(self.workspace), 'external_dir': str(self.external_dir),
                   'assurance': 'ordinary trusted runner/OS/Docker; no hardware attestation', 'g4_verified': False}
        receipt = write_exclusive(self.external_dir / 'episode_binding.json', binding)
        return {**binding, 'external_receipt': receipt}

    def validate_native_identity(self, actual):
        self.guard()
        if (self.episode_id is None or self.native_verified
                or actual != self.frozen['native_execution']['dependency_identity']):
            raise ValueError('actual installed native validator differs from prospective freeze')
        value = {'schema': 'prospective-native-installed-v1', 'episode_id': self.episode_id,
                 'config_sha256': digest(self.config), 'freeze_sha256': self.expected_sha,
                 'dependency_identity': copy.deepcopy(actual), 'before_container_or_actor_send': True}
        receipt = write_exclusive(self.external_dir / 'native_identity.json', value)
        self.native_verified = True
        return receipt

    def check_inspection(self, cid, inspection):
        if not re.fullmatch(r'[a-f0-9]{64}', cid) or inspection.get('Id') != cid:
            raise ValueError('actual container ID mismatch')
        state, config, host = (inspection.get(k, {}) for k in ('State', 'Config', 'HostConfig'))
        if (inspection.get('Image') != self.frozen['actor_image_id']
                or state.get('Status') != 'created' or state.get('Running') is not False
                or state.get('StartedAt') not in ('0001-01-01T00:00:00Z', '0001-01-01T00:00:00.000000000Z')):
            raise ValueError('container must use frozen image and be inspected before its first start')
        network = inspection.get('NetworkSettings')
        if (not isinstance(network, dict) or not isinstance(network.get('Networks'), dict)
                or not set(network['Networks']) <= {'none'} or network.get('Ports') not in (None, {})):
            raise ValueError('actual Docker network attachment/port inventory differs')
        if (host.get('NetworkMode') != 'none' or host.get('ReadonlyRootfs') is not True
                or host.get('Privileged') is not False or host.get('CapDrop') != ['ALL']
                or host.get('CapAdd') not in (None, [])
                or host.get('SecurityOpt') not in (['no-new-privileges'], ['no-new-privileges:true'])
                or any(host.get(k) not in (None, [], {}) for k in (
                    'Binds', 'Devices', 'DeviceRequests', 'DeviceCgroupRules', 'VolumesFrom',
                    'Links', 'ExtraHosts', 'PortBindings'))
                or host.get('PublishAllPorts') not in (None, False)
                or any(host.get(k) not in ('', None, 'private') for k in ('PidMode', 'IpcMode', 'UTSMode', 'UsernsMode', 'CgroupnsMode'))):
            raise ValueError('actual container isolation/network/device configuration differs')
        if (host.get('Memory') != 2 * 1024**3 or host.get('NanoCpus') != 2_000_000_000
                or host.get('PidsLimit') != 128
                or any(host.get(k) not in (None, 0, '') for k in ('CpuQuota', 'CpuPeriod', 'CpusetCpus', 'CpusetMems'))):
            raise ValueError('actual resource limits differ')
        if host.get('Tmpfs') != {'/tmp': 'rw,nosuid,nodev,size=256m,mode=1777'}:
            raise ValueError('actual temporary filesystem differs')
        expected_env = {**env_dict(self.frozen['actor_image_env']), **RUNTIME_ENV}
        if env_dict(config.get('Env')) != expected_env or config.get('User') != self.frozen['actor_user']:
            raise ValueError('actual container environment/user differs from image plus approved runtime overrides')
        mounts = inspection.get('Mounts')
        if not isinstance(mounts, list):
            raise ValueError('actual mount inventory missing')
        if not mounts:
            if (config.get('WorkingDir') != '/tmp' or config.get('Entrypoint') != ['python']
                    or config.get('Cmd') != ['-I', '-c', PROBE_SCRIPT]
                    or host.get('Mounts') not in (None, [])):
                raise ValueError('zero-mount container must be the fixed environment probe')
            role = 'environment_probe'
        else:
            if len(mounts) != 1:
                raise ValueError('Actor must mount only its workspace')
            mount = mounts[0]
            if (mount.get('Type') != 'bind' or mount.get('Source') != str(self.workspace)
                    or mount.get('Destination') != '/workspace' or mount.get('RW') is not True
                    or mount.get('Propagation') not in ('rprivate', 'private')):
                raise ValueError('actual Actor mount differs from the unique run workspace')
            requested = host.get('Mounts')
            if (not isinstance(requested, list) or len(requested) != 1
                    or requested[0].get('Type') != 'bind'
                    or requested[0].get('Source') != str(self.workspace)
                    or requested[0].get('Target') != '/workspace'
                    or requested[0].get('ReadOnly', False) is not False):
                raise ValueError('HostConfig mount specification differs from actual workspace mount')
            if (config.get('WorkingDir') != '/workspace' or config.get('Entrypoint') not in (['/bin/bash'], ['/bin/sh'])
                    or not isinstance(config.get('Cmd'), list) or not config['Cmd']):
                raise ValueError('Actor command boundary differs from frozen shell runtime')
            role = 'actor_tool'
        if (inspection.get('Path') != config['Entrypoint'][0]
                or inspection.get('Args') != config['Cmd']):
            raise ValueError('actual process arguments differ from Config')
        return role

    def observe_container(self, cid, actualinspect):
        self.guard()
        if self.episode_id is None or self.workspace is None or not self.native_verified or cid in self.seen_containers:
            raise ValueError('unbound or duplicate container observation')
        role = self.check_inspection(cid, actualinspect)
        value = {'schema': 'prospective-container-before-start-v1', 'episode_id': self.episode_id,
                 'task_id': self.config['benchmark']['task_id'], 'config_sha256': digest(self.config),
                 'freeze_sha256': self.expected_sha, 'container_id': cid, 'role': role,
                 'inspection': copy.deepcopy(actualinspect), 'before_start': True, 'g4_verified': False}
        receipt = write_exclusive(self.external_dir / (cid + '.json'), value)
        self.seen_containers.add(cid)
        return {**receipt, 'episode_id': self.episode_id, 'container_id': cid, 'freeze_sha256': self.expected_sha}
