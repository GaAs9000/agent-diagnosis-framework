"""Prospective, single-episode observer for a sealed cross-actor branch.

The runner supplies the complete absolute source/reference inventory at freeze
time. Callbacks only hash bytes and inspect supplied Docker state; they never
execute the wrapper, dependency validator, Docker, or a provider request.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import re

from .v3_execution import digest, verify_seal


_SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/prospective_execution_observer.py'
_spec = importlib.util.spec_from_file_location('cross_actor_original_inspector', _SCRIPT)
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)


class CrossActorExecutionObserver(_original.ProspectiveExecutionObserver):
    """L1 observer hooks backed by one immutable per-branch contract.

    Required contract fields: artifact_sha256 (v3 seal), source_files and
    bound_files (absolute-path SHA maps), parent_index, checkpoint, wrapper,
    actor_image_inspect (bound path/sha256 references), expected_config_sha256,
    actor_image_id, actor_image_env, actor_user, native_dependency_identity.
    """

    def __init__(self, contract: dict, external_dir: Path):
        self.frozen = copy.deepcopy(contract)
        self.expected_sha = digest(self.frozen)
        self.external_dir = Path(external_dir).absolute()
        if (self.external_dir.exists() or self.external_dir.is_symlink()
                or any(p.is_symlink() for p in self.external_dir.parents)
                or any((p / 'evidence_index.json').exists() or (p / 'trajectory.jsonl').exists()
                       for p in self.external_dir.parents)):
            raise ValueError('external receipt directory must be new and unaliased outside evidence')
        self.config = None
        self.workspace = None
        self.episode_id = None
        self.seen_containers = set()
        self.native_verified = False
        self.guard()

    def guard(self):
        verify_seal(self.frozen)
        if digest(self.frozen) != self.expected_sha:
            raise ValueError('branch contract changed during execution')
        for key in ('source_files', 'bound_files'):
            inventory = self.frozen.get(key)
            if not isinstance(inventory, dict) or not inventory:
                raise ValueError(key + ' inventory required')
            for name, expected in inventory.items():
                if (not isinstance(name, str) or not Path(name).is_absolute()
                        or not isinstance(expected, str) or not re.fullmatch('[a-f0-9]{64}', expected)
                        or _original.sha(name) != expected):
                    raise ValueError(key + ' file changed: ' + str(name))
        # Reusing the original inspection code is safe only while those exact
        # bytes and this adapter remain in the runner's frozen source inventory.
        for path in (_SCRIPT, Path(__file__).resolve()):
            if self.frozen['source_files'].get(str(path)) != _original.sha(path):
                raise ValueError('observer source missing from source_files')
        for key in ('parent_index', 'checkpoint', 'wrapper', 'actor_image_inspect'):
            _original.bound_reference(self.frozen, self.frozen.get(key))
        image = _original.read(self.frozen['actor_image_inspect']['path'])
        if (not re.fullmatch(r'sha256:[a-f0-9]{64}', self.frozen.get('actor_image_id', ''))
                or image.get('Id') != self.frozen['actor_image_id']
                or image.get('Config', {}).get('Env') != self.frozen.get('actor_image_env')):
            raise ValueError('actor image identity/environment differs from bound inspection')
        _original.env_dict(self.frozen['actor_image_env'])
        if self.frozen.get('actor_user') != _original.host_user():
            raise ValueError('frozen actor_user differs from runtime host identity')
        if (not isinstance(self.frozen.get('native_dependency_identity'), dict)
                or not self.frozen['native_dependency_identity']
                or not re.fullmatch('[a-f0-9]{64}', self.frozen.get('expected_config_sha256', ''))):
            raise ValueError('native identity and final resolved config digest required')
        if self.config is not None and digest(self.config) != self.frozen['expected_config_sha256']:
            raise ValueError('bound configuration changed')

    def validate(self, config):
        self.guard()
        if self.config is not None or self.episode_id is not None:
            raise ValueError('observer is one episode only')
        if digest(config) != self.frozen['expected_config_sha256']:
            raise ValueError('final resolved configuration differs from branch contract')
        runtime = config.get('runtime', {})
        if (runtime.get('adapter') != 'docker-cold'
                or runtime.get('image') != self.frozen['actor_image_id']
                or runtime.get('memory') != '2g' or runtime.get('cpus') != 2
                or runtime.get('pids_limit') != 128
                or runtime.get('read_only_rootfs', True) is not True
                or runtime.get('tmpfs_exec', False) is not False
                or runtime.get('options', {}).get('persist_app_repo', False) is not False
                or runtime.get('user') not in (None, self.frozen['actor_user'])):
            raise ValueError('resolved runtime violates frozen isolation/resources')
        self.config = copy.deepcopy(config)

    # bind_workspace and check_inspection are deliberately inherited verbatim.

    def bind_episode(self, episode_id, config_sha):
        self.guard()
        if (self.config is None or self.workspace is None or self.episode_id is not None
                or not isinstance(episode_id, str) or not re.fullmatch(r'ep_[a-zA-Z0-9_-]+', episode_id)
                or config_sha != digest(self.config)):
            raise ValueError('episode/config/workspace binding mismatch')
        self.external_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        value = {'schema': 'cross-actor-episode-binding-v1', 'episode_id': episode_id,
                 'config_sha256': config_sha, 'config': copy.deepcopy(self.config),
                 'contract_sha256': self.expected_sha,
                 'contract_artifact_sha256': self.frozen['artifact_sha256'],
                 'workspace': str(self.workspace), 'external_dir': str(self.external_dir),
                 'before_container_or_actor_send': True, 'g4_verified': False}
        receipt = _original.write_exclusive(self.external_dir / 'episode_binding.json', value)
        self.episode_id = episode_id
        return {**value, 'external_receipt': receipt}

    def validate_native_identity(self, actual):
        self.guard()
        if (self.episode_id is None or self.native_verified
                or actual != self.frozen['native_dependency_identity']):
            raise ValueError('actual installed native validator differs from branch contract')
        value = {'schema': 'cross-actor-native-installed-v1', 'episode_id': self.episode_id,
                 'config_sha256': digest(self.config), 'contract_sha256': self.expected_sha,
                 'dependency_identity': copy.deepcopy(actual), 'before_container_or_actor_send': True}
        receipt = _original.write_exclusive(self.external_dir / 'native_identity.json', value)
        self.native_verified = True
        return receipt

    def observe_container(self, cid, actualinspect):
        self.guard()
        if self.episode_id is None or self.workspace is None or not self.native_verified or cid in self.seen_containers:
            raise ValueError('unbound or duplicate container observation')
        role = self.check_inspection(cid, actualinspect)
        value = {'schema': 'cross-actor-container-before-start-v1', 'episode_id': self.episode_id,
                 'config_sha256': digest(self.config), 'contract_sha256': self.expected_sha,
                 'container_id': cid, 'role': role, 'inspection': copy.deepcopy(actualinspect),
                 'before_start': True, 'g4_verified': False}
        receipt = _original.write_exclusive(self.external_dir / (cid + '.json'), value)
        self.seen_containers.add(cid)
        return {**receipt, 'episode_id': self.episode_id, 'container_id': cid,
                'contract_sha256': self.expected_sha}
