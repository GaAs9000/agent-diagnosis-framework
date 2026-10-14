"""Integrity failures and exact SDK prefix checks, without external execution."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('r1_evidence_verifier',
    Path(__file__).resolve().parents[1] / 'scripts/verify_r1_evidence.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_modified_evidence_hash_rejected(tmp_path):
    path = tmp_path / 'grade.json'
    path.write_bytes(b'original evidence')
    expected = verifier.sha(path)
    verifier.check_hash(path, expected)
    path.write_bytes(b'modified evidence')
    with pytest.raises(ValueError, match='file hash mismatch'):
        verifier.check_hash(path, expected)


def fixture_prefix():
    prefix = [{'role':'user','content':'Runtime-owned advisory: public observation presentation.\n甲\n'},
              {'role':'assistant','content':'','tool_calls':[{'id':'call-1','function':'workspace_python',
                  'arguments':{'code':'print("甲")\n'}}]},
              {'role':'tool','content':'甲\n','tool_call_id':'call-1'}]
    native = {'request':{'messages':[
        prefix[0].copy(),
        {'role':'assistant','content':'','tool_calls':[{'id':'call-1','function':{'name':'workspace_python',
            'arguments':'{"code":"print(\\"甲\\")\\n"}'},'type':'function'}]},
        prefix[2].copy()]}}
    return native, prefix


def test_actual_sdk_prefix_exact_bytes_and_argument_values():
    native, prefix = fixture_prefix()
    verifier.check_provider_prefix(native, prefix)


@pytest.mark.parametrize('change', ['advisory', 'tool_argument', 'observation', 'length'])
def test_actual_sdk_payload_substitution_rejected(change):
    native, prefix = fixture_prefix()
    native = copy.deepcopy(native)
    rows = native['request']['messages']
    if change == 'advisory':
        rows[0]['content'] = rows[0]['content'].rstrip('\n')
    elif change == 'tool_argument':
        rows[1]['tool_calls'][0]['function']['arguments'] = '{"code":"print(1)"}'
    elif change == 'observation':
        rows[2]['tool_call_id'] = 'another-call'
    else:
        rows.pop()
    with pytest.raises(ValueError, match='provider'):
        verifier.check_provider_prefix(native, prefix)


def test_workspace_bytes_and_modes_bound(tmp_path):
    path = tmp_path / 'artifact'
    path.write_bytes(b'original')
    original = verifier.workspace_hash(tmp_path)
    path.write_bytes(b'changed')
    assert verifier.workspace_hash(tmp_path) != original
    path.write_bytes(b'original')
    path.chmod(0o400)
    assert verifier.workspace_hash(tmp_path) != original
