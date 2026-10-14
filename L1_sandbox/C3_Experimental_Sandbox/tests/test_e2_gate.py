import json

import pytest

from c3_sandbox.foundation_gate import CHECKS, SUITES, source_identities, verify_foundation


def evidence(tmp_path):
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(json.dumps({"status": "passed", "is_synthetic": True,
                                      "checks": dict.fromkeys(CHECKS, True), "source_identities": source_identities()}))
    junit = tmp_path / "junit.xml"
    junit.write_text('<testsuites><testsuite>' + ''.join(f'<testcase classname="{s}" name="fixture"/>' for s in SUITES) + '</testsuite></testsuites>')
    return acceptance, junit


def test_gate_remains_scientifically_closed(tmp_path):
    report = verify_foundation(*evidence(tmp_path))
    assert report["may_develop_D6_D8"] is True
    assert report["formal_experiments_admitted"] is False
    assert report["real_replay_qualified"] is False


@pytest.mark.parametrize("bad", ["source", "checks", "synthetic"])
def test_gate_rejects_stale_or_incomplete_claim(tmp_path, bad):
    a, j = evidence(tmp_path)
    data = json.loads(a.read_text())
    if bad == "source":
        data["source_identities"].pop(next(iter(data["source_identities"])))
    elif bad == "checks":
        data["checks"][next(iter(CHECKS))] = False
    else:
        data["is_synthetic"] = False
    a.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        verify_foundation(a, j)


@pytest.mark.parametrize("tag", ["failure", "error", "skipped"])
def test_gate_rejects_nonpassing_tests(tmp_path, tag):
    a, j = evidence(tmp_path)
    text = j.read_text().replace('name="fixture"/>', f'name="fixture"><{tag}/></testcase>', 1)
    j.write_text(text)
    with pytest.raises(ValueError):
        verify_foundation(a, j)


def test_gate_requires_every_foundation_suite(tmp_path):
    a, j = evidence(tmp_path)
    j.write_text('<testsuites><testsuite><testcase classname="test_e2_data"/></testsuite></testsuites>')
    with pytest.raises(ValueError):
        verify_foundation(a, j)
