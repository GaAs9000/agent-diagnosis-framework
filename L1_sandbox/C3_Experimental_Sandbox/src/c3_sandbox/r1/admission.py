"""Controller-owned, evidence-bound admission for observed R1 cold boundaries.

The controller and its host are trusted. HMAC receipts prevent accidental or
agent-side substitution; they are not a proof against a malicious host.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import stat

from ..evidence._storage import directory, digest, encode, get, identity, local_path, put, read_regular
from .docker_runtime import boundary_mounts_valid


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _audit(audit, image):
    """Re-evaluate raw observations rather than trusting qualified flags."""
    data = audit['inspect']
    host, config = data['HostConfig'], data['Config']
    _require(data['Image'] == image, 'boundary image mismatch')
    _require(config['User'] == '1000:1000' and config['WorkingDir'] == '/workspace', 'boundary user/workdir mismatch')
    _require(data['State']['Running'] is True and type(data['State']['Pid']) is int and data['State']['Pid'] > 0, 'inactive boundary')
    expected = {'NetworkMode': 'none', 'ReadonlyRootfs': True, 'Privileged': False,
                'NanoCpus': 2000000000, 'Memory': 4294967296, 'MemorySwap': 4294967296,
                'PidsLimit': 128, 'IpcMode': 'none'}
    for key, value in expected.items():
        _require(type(host[key]) is type(value) and host[key] == value, 'unsafe boundary ' + key)
    _require(not host['Binds'] and not host['CapAdd'] and host['PidMode'] in ('', None), 'host access at boundary')
    _require(host['CapDrop'] == ['ALL'] and 'no-new-privileges:true' in host['SecurityOpt'], 'unsafe boundary capabilities')
    _require(boundary_mounts_valid(data, audit), 'unsafe boundary kernel mounts')
    tmpfs = host['Tmpfs']
    _require(set(tmpfs) == {'/workspace', '/tmp'}, 'unsafe tmpfs paths')
    for name, options in tmpfs.items():
        flags = set(options.split(','))
        needed = {'rw', 'nosuid', 'nodev', 'uid=1000', 'gid=1000', 'mode=0700', 'size=256m' if name == '/workspace' else 'size=64m'}
        if name == '/tmp':
            needed.add('noexec')
        _require(needed <= flags, 'unsafe tmpfs options')
    status = dict(line.split(':', 1) for line in audit['status'].splitlines() if ':' in line)
    _require(status.get('CapEff', '').strip() == '0000000000000000' and status.get('NoNewPrivs', '').strip() == '1', 'unsafe process privilege status')
    _require(audit['processes'] == [{'pid': 1, 'cmdline': 'sleep infinity'}] and audit['tmp_entries'] == [], 'persistent process or tmp state')
    _require(audit['qualified'] is True and audit['effective_config_ok'] is True, 'boundary audit failed')


def _validate(record, source, restored, report, model):
    _require(type(record['schema_version']) is int and record['schema_version'] == 1, 'unsupported checkpoint schema')
    _require(record['is_synthetic'] is False and record['snapshot_id'] is not None, 'real checkpoint required')
    from ..replay import ReplayQualification
    _require(not ReplayQualification._input_reasons(record) and not record['registration_reasons'], 'invalid checkpoint inputs')
    _require(digest(encode(record['messages'])) == record['message_identity'] and digest(encode(record['runtime'])) == record['runtime_identity'], 'checkpoint component identity mismatch')
    image = record['runtime']['image_digest']
    _audit(source, image)
    _audit(restored, image)
    _require(source['inspect']['State']['Pid'] != restored['inspect']['State']['Pid'], 'restore must use a fresh runtime')
    _require(report['verified'] is True and all(report[k] is True for k in ('messages_equal', 'runtime_equal', 'system_equal')), 'restore equality not established')
    _require(type(report['expected_paths']) is int and type(report['actual_paths']) is int and report['expected_paths'] >= 0 and report['expected_paths'] == report['actual_paths'], 'restore paths mismatch')
    for key in ('provider', 'model', 'tool_call_id', 'nonce'):
        _require(isinstance(model[key], str) and bool(model[key].strip()), 'missing native model qualification ' + key)
    for key in ('native_tool_calls', 'prompt_tokens', 'completion_tokens'):
        _require(type(model[key]) is int and model[key] > 0, 'missing native model qualification ' + key)
    identity(model['model_fingerprint'])
    registered = record['runtime']['model_config']
    _require(all(registered.get(key) == model[key] for key in ('provider', 'model', 'model_fingerprint')), 'model qualification identity mismatch')
    _require(model['tool_call_id'] == model['observation_tool_call_id'], 'tool observation mismatch')
    _require(model['nonce'] == model['observed_nonce'] == model['independent_nonce'], 'independent nonce mismatch')


class R1AdmissionAuthority:
    """Private local authority. Never mount this directory into an agent runtime."""

    def __init__(self, root: Path):
        self.root = local_path(root)
        directory(self.root)
        self.root.chmod(0o700)
        self.receipts = self.root / 'receipts'
        directory(self.receipts)
        self.receipts.chmod(0o700)
        self.key_path = self.root / 'authority.key'
        try:
            fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(secrets.token_bytes(32))
        self._key = read_regular(self.key_path)
        _require(len(self._key) == 32 and stat.S_IMODE(self.key_path.stat().st_mode) == 0o600, 'invalid private authority key')

    def _sign(self, payload):
        return hmac.new(self._key, encode(payload), hashlib.sha256).hexdigest()

    def issue(self, checkpoint_id, checkpoint_record, source_audit, restore_audit,
              restoration_report, model_qualification, evidence_paths):
        identity(checkpoint_id)
        _require(digest(encode(checkpoint_record)) == checkpoint_id, 'checkpoint content identity mismatch')
        _validate(checkpoint_record, source_audit, restore_audit, restoration_report, model_qualification)
        required = [checkpoint_record, source_audit, restore_audit, restoration_report, model_qualification]
        evidence, documents = [], []
        _require(isinstance(evidence_paths, (list, tuple)) and len(evidence_paths) >= 6, 'raw qualification evidence required')
        for supplied in evidence_paths:
            path = local_path(supplied)
            data = read_regular(path)
            _require(bool(data), 'empty qualification evidence')
            evidence.append({'path': str(path), 'sha256': digest(data)})
            try:
                documents.append(json.loads(data))
            except (ValueError, UnicodeDecodeError):
                pass
        _require(len({item['path'] for item in evidence}) == len(evidence), 'duplicate qualification evidence')
        _require(all(value in documents for value in required), 'audit/model/equality evidence documents missing')
        payload = {'schema_version': 1, 'checkpoint_identity': checkpoint_id,
                   'checkpoint_record': checkpoint_record, 'source_audit': source_audit,
                   'restore_audit': restore_audit, 'restoration_report': restoration_report,
                   'model_qualification': model_qualification, 'evidence': evidence}
        receipt = {'payload': payload, 'signature': self._sign(payload)}
        receipt_id = put(self.receipts, encode(receipt), '.json')
        # Exclusive immutable index: a checkpoint cannot be silently re-admitted.
        index = self.root / (checkpoint_id + '.receipt')
        try:
            with index.open('x') as stream:
                stream.write(receipt_id)
            index.chmod(0o400)
        except FileExistsError:
            _require(read_regular(index).decode() == receipt_id, 'checkpoint already admitted with different evidence')
        return receipt_id

    def verify(self, checkpoint_id, record):
        try:
            identity(checkpoint_id)
            _require(read_regular(self.key_path) == self._key and stat.S_IMODE(self.key_path.stat().st_mode) == 0o600, 'authority key changed')
            receipt_id = read_regular(self.root / (checkpoint_id + '.receipt')).decode()
            receipt = json.loads(get(self.receipts, receipt_id, '.json'))
            payload = receipt['payload']
            _require(hmac.compare_digest(receipt['signature'], self._sign(payload)), 'invalid admission signature')
            _require(payload['schema_version'] == 1 and payload['checkpoint_identity'] == checkpoint_id and digest(encode(record)) == checkpoint_id and encode(payload['checkpoint_record']) == encode(record), 'admission checkpoint mismatch')
            _validate(record, payload['source_audit'], payload['restore_audit'], payload['restoration_report'], payload['model_qualification'])
            for evidence in payload['evidence']:
                _require(digest(read_regular(local_path(evidence['path']))) == evidence['sha256'], 'admission evidence changed')
            return True
        except (ValueError, OSError, TypeError, KeyError, AttributeError):
            return False

    def receipt_identity(self, checkpoint_id, record):
        if not self.verify(checkpoint_id, record):
            raise ValueError('invalid admission receipt')
        return read_regular(self.root / (identity(checkpoint_id) + '.receipt')).decode()
