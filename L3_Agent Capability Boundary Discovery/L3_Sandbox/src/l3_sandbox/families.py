"""Validate the frozen operational semantic partition, without claiming lineage."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .common import L3Error, read, sha_file

DEFINITION = 'semantic_operational_group'


def load_family_manifest(path: Path) -> dict:
    """Require an exhaustive, nonoverlapping partition of the pinned task lock."""
    path = Path(path)
    manifest = read(path)
    if (manifest.get('schema_version') != 'officeval-family-v1'
            or manifest.get('status') != 'frozen'
            or manifest.get('family_definition') != DEFINITION
            or manifest.get('lineage_status') != 'unknown'):
        raise L3Error('Expected frozen semantic operational family manifest')
    lock_path = (path.parent / manifest['data_lock']).resolve()
    lock = read(lock_path)
    if (sha_file(lock_path) != manifest.get('data_lock_sha256')
            or lock.get('dataset_revision') != manifest.get('dataset_revision')):
        raise L3Error('Family manifest task lock identity mismatch')
    tasks = manifest.get('tasks', [])
    task_ids = [t['task_id'] for t in tasks]
    if len(set(task_ids)) != len(task_ids) or set(task_ids) != set(lock['tasks']):
        raise L3Error('Family manifest must cover each locked task exactly once')
    groups = {}
    for task in tasks:
        task_id, family_id = task['task_id'], task.get('family_id')
        if (not isinstance(family_id, str) or not family_id
                or family_id != task.get('provisional_block_id')
                or task.get('family_verified') is not True
                or task.get('family_definition') != DEFINITION
                or task.get('lineage_status') != 'unknown'
                or task.get('split') not in ('development', 'heldout')):
            raise L3Error('Invalid semantic family definition or membership')
        if task.get('heldout_eligible') != (task['split'] == 'heldout'):
            raise L3Error('Family heldout eligibility disagrees with split')
        if task['split'] == 'heldout' and (task.get('known_prior_exposure') or task.get('pilot_candidate')):
            raise L3Error('Explored or pilot tasks cannot be heldout')
        entry = lock['tasks'][task_id]
        if (task.get('task_metadata_sha256') != entry['task']
                or task.get('input_asset_sha256s') != entry['assets']):
            raise L3Error('Family task metadata or assets differ from task lock')
        group = groups.setdefault(family_id, {'family_id': family_id,
            'split': task['split'], 'task_ids': []})
        if group['split'] != task['split']:
            raise L3Error('A semantic family cannot span development and heldout')
        group['task_ids'].append(task_id)
    expected = [groups[k] for k in sorted(groups)]
    if manifest.get('families') != expected:
        raise L3Error('Family index disagrees with task membership')
    counts = Counter(t['split'] for t in tasks)
    family_counts = Counter(g['split'] for g in expected)
    actual = {'task_count': len(tasks), 'family_count': len(groups),
              'development_task_count': counts['development'], 'heldout_task_count': counts['heldout'],
              'development_family_count': family_counts['development'],
              'heldout_family_count': family_counts['heldout']}
    if any(manifest.get(k) != v for k, v in actual.items()):
        raise L3Error('Family partition counts disagree')
    return manifest


def validate_family_selection(manifest: dict, tasks: list[dict], purpose: str) -> None:
    """Bind plan metadata to the manifest; pilot plans stay on development families."""
    by_task = {t['task_id']: t for t in manifest['tasks']}
    for task in tasks:
        official = by_task.get(task['task_id'])
        if official is None or any(task.get(k) != official.get(k)
                                  for k in ('family_id', 'family_verified', 'split')):
            raise L3Error('Study task family metadata mismatches frozen family manifest')
        if purpose == 'pilot' and official['split'] != 'development':
            raise L3Error('Development pilot cannot use heldout families')
        if purpose == 'validation' and official['split'] != 'heldout':
            raise L3Error('Validation requires heldout families')
