"""Frozen semantic split is an operational definition, not a lineage claim."""
import copy
from collections import Counter
from pathlib import Path

import pytest

from l3_sandbox.common import L3Error, read, write
from l3_sandbox.families import load_family_manifest, validate_family_selection

SANDBOX = Path(__file__).resolve().parents[1]
MANIFEST = SANDBOX.parents[1] / 'L1_sandbox/C3_Experimental_Sandbox/benchmarks/officeval/family_split.json'


@pytest.fixture
def frozen_copy(tmp_path):
    m = read(MANIFEST)
    m['data_lock'] = str(MANIFEST.parent / m['data_lock'])
    p = tmp_path / 'family_split.json'
    write(p, m)
    return p, m


def test_canonical_family_partition_and_pilot_membership():
    m = load_family_manifest(MANIFEST)
    old = read(MANIFEST.with_name('family_split.draft.json'))
    assert old['status'] == 'draft_unfrozen'
    assert len(m['tasks']) == 91 and len(m['families']) == 28
    assert Counter(t['split'] for t in m['tasks']) == {'development': 55, 'heldout': 36}
    assert Counter(f['split'] for f in m['families']) == {'development': 9, 'heldout': 19}
    assert {t['task_id']: t['family_id'] for t in m['tasks']} == {
        t['task_id']: t['provisional_block_id'] for t in old['tasks']}
    for name in ('budget_pilot', 'structure_budget'):
        spec = read(SANDBOX / f'configs/{name}.json')
        assert (SANDBOX / 'configs' / spec['family_manifest']).resolve() == MANIFEST
        validate_family_selection(m, spec['tasks'], 'pilot')
        assert len(set(t['family_id'] for t in spec['tasks'])) == 4
    heldout = [t for t in m['tasks'] if t['split'] == 'heldout']
    assert Counter(t['application_stratum'] for t in heldout)['powerpoint'] == 19
    assert Counter(t['application_stratum'] for t in heldout)['excel'] == 3
    assert all(t['lineage_status'] == 'unknown' for t in m['tasks'])


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'family', 'split', 'index', 'lock', 'metadata', 'assets', 'exposed', 'definition', 'count'])
def test_manifest_rejects_leakage_or_partition_drift(frozen_copy, mutation):
    p, m = frozen_copy
    if mutation == 'missing': m['tasks'].pop()
    elif mutation == 'duplicate': m['tasks'].append(copy.deepcopy(m['tasks'][0]))
    elif mutation == 'family': m['tasks'][0]['family_id'] = 'different'
    elif mutation == 'split':
        m['tasks'][0].update(split='heldout', heldout_eligible=True)
    elif mutation == 'index': m['families'][0]['task_ids'].pop()
    elif mutation == 'lock': m['data_lock_sha256'] = '0' * 64
    elif mutation == 'metadata': m['tasks'][0]['task_metadata_sha256'] = '0' * 64
    elif mutation == 'assets': m['tasks'][0]['input_asset_sha256s'] = {}
    elif mutation == 'exposed':
        next(t for t in m['tasks'] if t['split'] == 'heldout')['known_prior_exposure'] = True
    elif mutation == 'definition': m['family_definition'] = 'certified_template_lineage'
    else: m['family_count'] = 27
    write(p, m)
    with pytest.raises(L3Error): load_family_manifest(p)


def test_selection_requires_canonical_membership_and_reserves_holdout():
    m = load_family_manifest(MANIFEST)
    dev = copy.deepcopy(next(t for t in m['tasks'] if t['split'] == 'development'))
    holdout = next(t for t in m['tasks'] if t['split'] == 'heldout')
    validate_family_selection(m, [holdout], 'validation')
    with pytest.raises(L3Error, match='Development pilot'):
        validate_family_selection(m, [holdout], 'pilot')
    with pytest.raises(L3Error, match='Validation requires'):
        validate_family_selection(m, [dev], 'validation')
    dev['family_id'] = 'wrong'
    with pytest.raises(L3Error, match='mismatches'):
        validate_family_selection(m, [dev], 'pilot')
