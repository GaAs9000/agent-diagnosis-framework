"""Read-only L1 package reader: events, checkpoints, snapshots, and grading projection (L2 v0.3)."""
from __future__ import annotations
import copy
import re
from pathlib import Path
from typing import Any
from ..io import IntegrityError, L2Error, digest, loads, read_json, read_jsonl, sha_file


def diff_snapshots(before: dict, after: dict) -> list[dict]:
    result = []
    for field in ('files', 'directories'):
        b = {x['path']: x for x in before.get(field, [])}
        a = {x['path']: x for x in after.get(field, [])}
        for p in sorted(set(a) | set(b)):
            if a.get(p) == b.get(p):
                continue
            change = (
                'directory_changed' if field == 'directories' else
                'created' if p not in b else
                'deleted' if p not in a else
                'modified'
            )
            result.append({'path': p, 'change': change, 'before': b.get(p), 'after': a.get(p)})
    return result


class L1Package:
    """Lightweight read-only view of an L1 episode package (root trajectory or resumed branch)."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        if not (self.root / 'manifest.json').is_file():
            raise L2Error(f'Missing manifest.json in L1 package: {self.root}')
        self._validate_declared_index()
        self.manifest = read_json(self.root / 'manifest.json')
        self.episode_id: str = self.manifest['episode_id']
        self.version: str = self.manifest.get('schema_version', '0.2.0')
        self.index_sha256: str = (
            sha_file(self.root / 'evidence_index.json')
            if (self.root / 'evidence_index.json').is_file()
            else sha_file(self.root / 'manifest.json')
        )
        raw_events = read_jsonl(self.root / 'trajectory.jsonl') if (self.root / 'trajectory.jsonl').is_file() else []
        self.events = [{**e, 'data': e.get('data', e.get('payload', {}))} for e in raw_events]
        self.events_by_id = {e['event_id']: e for e in self.events if 'event_id' in e}

        self.snapshots: dict[str, dict] = {}
        for e in self.events:
            if e.get('kind') == 'state.snapshot':
                snap_ref = (e['data'].get('snapshot') or {}).get('path')
                if snap_ref and (self.root / snap_ref).is_file():
                    s = self.json(snap_ref)
                    self.snapshots[s['snapshot_id']] = s

        # Turn offset: root runs start at 0 (so first model turn is 1);
        # branches resumed from cp{k:06d} start at k (so first resampled turn is k + 1).
        self.start_turn: int = self._resolve_start_turn()
        self.local_turns: int = sum(1 for e in self.events if e.get('kind') == 'model.request')
        self.turns: int = self.start_turn + self.local_turns

    def _validate_declared_index(self) -> None:
        index_path = self.root / 'evidence_index.json'
        if not index_path.is_file():
            if read_json(self.root / 'manifest.json').get('recording_mode') == 'live':
                raise IntegrityError('Live L1 package is missing evidence index')
            return  # Minimal offline fixtures have no declared seal to validate.
        index = read_json(index_path)
        files = index.get('files')
        if not isinstance(files, list):
            raise IntegrityError('L1 evidence index lacks file inventory')
        if index.get('episode_id') != read_json(self.root / 'manifest.json').get('episode_id'):
            raise IntegrityError('L1 evidence index episode mismatch')

    def _resolve_start_turn(self) -> int:
        lineage = self.manifest.get('lineage') or {}
        cp_id = lineage.get('parent_checkpoint_id')
        if isinstance(cp_id, str):
            m = re.fullmatch(r'cp([0-9]{6})', cp_id)
            if m:
                return int(m.group(1))
        for e in self.events:
            if e.get('kind') == 'episode.restored':
                rcp = (e.get('data') or {}).get('parent_checkpoint_id') or (e.get('data') or {}).get('checkpoint_id')
                if isinstance(rcp, str) and re.fullmatch(r'cp[0-9]{6}', rcp):
                    return int(rcp[2:])
        return 0

    def read(self, rel_path: str) -> bytes:
        p = (self.root / rel_path).resolve()
        if not p.is_relative_to(self.root) or not p.is_file():
            raise IntegrityError(f'Missing or unsafe L1 file: {rel_path}')
        return p.read_bytes()

    def json(self, rel_path: str) -> Any:
        return loads(self.read(rel_path))

    def jsonl(self, rel_path: str) -> list[dict]:
        p = (self.root / rel_path).resolve()
        if not p.is_relative_to(self.root) or not p.is_file():
            return []
        return read_jsonl(p)

    @property
    def benchmark(self) -> str:
        task = self.manifest.get('task') or {}
        if task.get('benchmark'):
            return str(task['benchmark']).lower()
        if (self.root / 'benchmark/public_task.json').is_file():
            pub = self.json('benchmark/public_task.json')
            if pub.get('benchmark'):
                return str(pub['benchmark']).lower()
        return 'officeval'

    @property
    def task_id(self) -> str:
        task = self.manifest.get('task') or {}
        if task.get('task_id'):
            return str(task['task_id'])
        if (self.root / 'benchmark/public_task.json').is_file():
            pub = self.json('benchmark/public_task.json')
            if pub.get('task_id'):
                return str(pub['task_id'])
        return 'unknown_task'

    @property
    def family_id(self) -> str:
        return f'{self.benchmark}:{self.task_id}'

    @property
    def is_boolean_endpoint(self) -> bool:
        return self.benchmark == 'lhtb'

    def checkpoints(self) -> dict[int, dict]:
        if not (self.root / 'checkpoints/index.json').is_file():
            return {}
        idx = self.json('checkpoints/index.json')
        out: dict[int, dict] = {}
        for row in idx.get('checkpoints', []):
            cp_id = row.get('checkpoint_id', '')
            if re.fullmatch(r'cp[0-9]{6}', cp_id) and (self.root / row['path']).is_file():
                k = int(cp_id[2:])
                cp = self.json(row['path'])
                out[k] = {
                    'k': k,
                    'checkpoint_id': cp_id,
                    'path': row['path'],
                    'sha256': sha_file(self.root / row['path']),
                    'eligible': (cp.get('qualification') or {}).get('eligible') is True,
                    'snapshot_id': cp.get('snapshot_id'),
                    'budget_used': cp.get('budget_used') or {},
                    'config_sha256': cp.get('config_sha256'),
                }
        return out

    def content_identity(self) -> str:
        # L1 owns sealing the inventory; L2 binds to that immutable seal.
        return self.index_sha256

    def validate_resume_checkpoint(self, k: int) -> dict:
        cp = self.checkpoints().get(k)
        if cp is None or not cp['eligible']:
            raise IntegrityError(f'Checkpoint {k} is absent or unqualified')
        cfg = self.json('raw/resolved_config.json')
        if cp['config_sha256'] != digest(cfg):
            raise IntegrityError(f'Checkpoint {k} config identity mismatch')
        obj = self.json(cp['path'])
        if obj.get('checkpoint_id') != cp['checkpoint_id']:
            raise IntegrityError(f'Checkpoint {k} identity mismatch')
        budget, used = cfg.get('budget', {}), cp['budget_used']
        for limit, counter, inclusive in (('max_model_requests', 'model_requests', True),
                                          ('max_tool_calls', 'tool_calls', False),
                                          ('wall_time_seconds', 'agent_seconds', True),
                                          ('max_total_tokens', 'known_total_tokens', True)):
            if budget.get(limit) is not None:
                val, cap = float(used.get(counter, 0)), float(budget[limit])
                if (val >= cap) if inclusive else (val > cap):
                    raise IntegrityError(f'Checkpoint {k} has no remaining {limit}')
        return cp

    def validate_branch(self, parent: 'L1Package', k: int) -> None:
        cp = parent.validate_resume_checkpoint(k)
        expected = {'parent_episode_id': parent.episode_id,
                    'parent_checkpoint_id': cp['checkpoint_id'],
                    'parent_index_sha256': parent.index_sha256,
                    'parent_checkpoint_sha256': cp['sha256']}
        lineage = self.manifest.get('lineage') or {}
        if any(lineage.get(key) != value for key, value in expected.items()):
            raise IntegrityError(f'Branch parent lineage mismatch: {self.root}')
        if self.json('raw/resolved_config.json') != parent.json('raw/resolved_config.json'):
            raise IntegrityError(f'Branch actor config mismatch: {self.root}')
        if self.start_turn != k or self.family_id != parent.family_id:
            raise IntegrityError(f'Branch checkpoint/task mismatch: {self.root}')

    def model_version(self) -> str | None:
        for e in self.events:
            if e.get('kind') == 'model.parsed':
                mv = e['data'].get('reported_model_version')
                if isinstance(mv, str) and mv:
                    return mv
        for e in self.events:
            if e.get('kind') == 'model.response':
                d = e['data']
                if isinstance(d.get('model_version'), str) and d['model_version']:
                    return d['model_version']
                body_path = (d.get('body') or {}).get('path')
                if body_path and (self.root / body_path).is_file():
                    try:
                        body = self.json(body_path)
                        if isinstance(body, dict):
                            mv = body.get('modelVersion') or body.get('model')
                            if isinstance(mv, str) and mv:
                                return mv
                    except Exception:
                        pass
        return (self.manifest.get('model') or {}).get('model_version')

    def termination(self) -> str | None:
        status = self.manifest.get('status')
        if isinstance(status, dict) and isinstance(status.get('termination'), str) and status['termination']:
            return status['termination']
        return None

    def outcome(self) -> dict[str, Any]:
        """Project the shared L1 research endpoint; preserve native grade fields."""
        try:
            from c3_sandbox.l1.evaluation.study import study_outcome
        except ModuleNotFoundError as exc:
            raise L2Error('Install the local L1 c3-experimental-sandbox package alongside L2 to read research outcomes') from exc

        study = study_outcome(
            self.root, allow_test_fixture=self.manifest.get('recording_mode') == 'test_fixture')
        return {
            **study,
            'valid': study['study_valid'],
            'y': study['study_outcome_y'],
            'reason': study['zero_reason'] or study['study_classification'],
            'termination': self.termination(),
            'modelVersion': self.model_version(),
        }

    def is_failed_trajectory(self) -> bool:
        """Section 3: Only failed trajectories are analyzed (OfficeVal dim1 failed or LHTB not passed, or y < 1.0)."""
        out = self.outcome()
        if not out['valid'] or out['y'] is None:
            return False
        if (self.root / 'official_grade.json').is_file():
            g = self.json('official_grade.json')
            proj = g.get('projection') if isinstance(g.get('projection'), dict) else {}
            dims = proj.get('official_dimensions') if isinstance(proj.get('official_dimensions'), dict) else {}
            raw = g.get('official_raw') if isinstance(g.get('official_raw'), dict) else {}
            if self.benchmark == 'lhtb':
                passed = dims.get('passed', proj.get('passed'))
                if isinstance(passed, bool):
                    return not passed
            else:
                dim1 = dims.get('dim1_pass', proj.get('dim1_pass', raw.get('dim1_pass')))
                if isinstance(dim1, bool):
                    return not dim1
        return float(out['y']) < 1.0


# Alias for compatibility
L1Reader = L1Package
