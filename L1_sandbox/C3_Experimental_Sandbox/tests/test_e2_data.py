import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest

from c3_sandbox.manifests import canonical_hash, system_identity, validate_family_manifest, validate_system_manifest
from c3_sandbox.trajectory import OUTCOME_FIELDS, import_legacy_episode, validate_trajectory

LEGACY = Path(__file__).resolve().parents[2] / "Agent_Latent_Capability_Audit_Kit/artifacts/officeval_pilot/llm_runs"


def trajectory(values=(None, None, None, None), status="unknown"):
    return {"schema_version": "1.0", "episode_id": "e", "task_id": "t", "task_family_id": "f", "system_id": "s", "split": "development", "status": status,
            "outcome": dict(zip(OUTCOME_FIELDS, values)), "events": [], "artifact_versions": [], "checkpoint_refs": [],
            "provenance": {"source": "test_fixture", "version": "1", "is_synthetic": True}}


@pytest.mark.parametrize("values,status", [((True,) * 4, "success"), ((True, False, False, False), "failure"), ((True, False, True, False), "partial"), ((None,) * 4, "unknown"), ((True, None, None, None), "unknown"), ((False, False, False, False), "unknown"), ((True, False, False, True), "success")])
def test_outcome_status(values, status):
    record = trajectory(values, status)
    assert validate_trajectory(record) == record
    record["status"] = "success" if status != "success" else "failure"
    with pytest.raises(ValueError, match="incompatible"):
        validate_trajectory(record)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), ("tuple",), {1: "nonstring key"}])
def test_finite_json(value):
    with pytest.raises(ValueError):
        canonical_hash(value)
    record = trajectory()
    record["extra"] = value
    with pytest.raises(ValueError):
        validate_trajectory(record)


def test_outcome_rejects_integer_boolean():
    record = trajectory()
    record["outcome"]["target_resolved"] = 1
    with pytest.raises(ValueError):
        validate_trajectory(record)


def test_real_legacy_import_byte_roundtrip():
    paths = sorted(LEGACY.glob("*/runs/*"))
    assert len(paths) == 12
    for path in paths:
        before = {name: (path / name).read_bytes() for name in ("summary.json", "events.jsonl", "artifacts.jsonl", "checkpoints.jsonl", "verifier_results.jsonl")}
        record = import_legacy_episode(path, task_family_id="officeval-source-" + json.loads(before["summary.json"])["task_id"], system_id=path.parent.parent.name)
        assert record["status"] == "unknown"
        assert all(value is None for value in record["outcome"].values())
        assert record["provenance"]["is_synthetic"] is False
        for name, data in before.items():
            raw = record["provenance"]["raw_files"][name]
            assert base64.b64decode(raw["base64"]) == data
            assert raw["sha256"] == hashlib.sha256(data).hexdigest()
            assert (path / name).read_bytes() == data
        assert record["events"] == [json.loads(line) for line in before["events.jsonl"].splitlines() if line.strip()]
        assert all(ref["replay_qualified"] is False for ref in record["checkpoint_refs"])


def legacy_fixture(tmp_path):
    files = {"summary.json": b'{"episode_id":"e","task_id":"t","status":"completed","safefix_pass":true,"unknown":{"x":2}}\r\n',
             "events.jsonl": b' {"event_id":"second","unknown":[1,2]}\r\n\n{"event_id":"first","stdout":"\\u4e2d"}',
             "artifacts.jsonl": b'{"unknown":"asset"}\n', "checkpoints.jsonl": b'{"checkpoint_id":"c","files":{}}\n', "verifier_results.jsonl": b'{"pass":true}\n'}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    return import_legacy_episode(tmp_path, task_family_id="f", system_id="s")


def test_unknown_fields_order_and_no_inferred_pass(tmp_path):
    record = legacy_fixture(tmp_path)
    assert [event["event_id"] for event in record["events"]] == ["second", "first"]
    assert record["events"][0]["unknown"] == [1, 2]
    assert record["provenance"]["raw_files"]["summary.json"]["records"]["unknown"] == {"x": 2}
    assert all(value is None for value in record["outcome"].values())
    with pytest.raises(ValueError, match="development-only"):
        import_legacy_episode(tmp_path, task_family_id="f", system_id="s", split="confirmation")


@pytest.mark.parametrize("mutation", ["hash", "records", "events", "checkpoint", "outcome"])
def test_retained_evidence_tamper_rejected(tmp_path, mutation):
    record = legacy_fixture(tmp_path)
    if mutation == "hash":
        record["provenance"]["raw_files"]["events.jsonl"]["sha256"] = "0" * 64
    elif mutation == "records":
        record["provenance"]["raw_files"]["events.jsonl"]["records"].reverse()
    elif mutation == "events":
        record["events"] = []
    elif mutation == "checkpoint":
        record["checkpoint_refs"][0]["replay_qualified"] = True
    else:
        record["outcome"] = dict.fromkeys(OUTCOME_FIELDS, True)
        record["status"] = "success"
    with pytest.raises(ValueError):
        validate_trajectory(record)


@pytest.mark.parametrize("data", [b'{"x":NaN}\n', b'{"x":1e999}\n', b'{"x":1,"x":2}\n', b'[]\n', b'{broken}\n'])
def test_legacy_rejects_ambiguous_json(tmp_path, data):
    legacy_fixture(tmp_path)
    (tmp_path / "events.jsonl").write_bytes(data)
    with pytest.raises(ValueError):
        import_legacy_episode(tmp_path, task_family_id="f", system_id="s")


def family(family_id="f", task="t", source="g", split="development", asset="a", digest="0" * 64):
    return {"task_family_id": family_id, "task_ids": [task], "source_group_ids": [source], "split": split, "asset_hashes": {asset: digest}}


def test_family_manifest_and_independent_copy():
    original = {"schema_version": "1.0", "families": [family(), family("f2", "t2", "g", asset="a2", digest="1" * 64)]}
    validated = validate_family_manifest(original)
    validated["families"][0]["task_ids"].append("new")
    assert original["families"][0]["task_ids"] == ["t"]


@pytest.mark.parametrize("second", [family(task="t2"), family("f2", "t"), family("f2", "t2", "g", "confirmation", "a2", "1" * 64), family("f2", "t2", "g2", asset="a", digest="1" * 64), family("f2", "t2", "g2", "confirmation", "a2")])
def test_split_leakage_duplicates_and_hash_conflicts(second):
    with pytest.raises(ValueError):
        validate_family_manifest({"schema_version": "1.0", "families": [family(), second]})


def system():
    config = {"model": {"id": "m", "revision": "r"}, "agent": {"version": "1"}, "tools": [], "prompt": "fixed prompt", "sampling": {"temperature": 0, "seed": 1}, "runtime": {"image": "sha256:fixed", "platform": "linux/arm64"}}
    return {"system_id": "s", "config": config, "config_sha256": canonical_hash(config)}


def test_system_hash_order_and_label():
    record = system()
    assert system_identity(record) == canonical_hash(record["config"])
    record["system_id"] = "other-label"
    assert system_identity(record) == record["config_sha256"]
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    assert validate_system_manifest({"schema_version": "1.0", "systems": [record]})["systems"] == [record]


@pytest.mark.parametrize("field", ["model", "agent", "tools", "prompt", "sampling", "runtime"])
def test_full_config_identity_changes(field):
    record = system()
    before = record["config_sha256"]
    record["config"][field] = {"changed": True}
    assert canonical_hash(record["config"]) != before
    with pytest.raises(ValueError, match="does not match"):
        system_identity(record)
    record["config_sha256"] = canonical_hash(record["config"])
    assert system_identity(record) != before


def test_missing_config_and_duplicate_system_id():
    record = system()
    del record["config"]["runtime"]
    with pytest.raises(ValueError):
        system_identity(record)
    with pytest.raises(ValueError, match="duplicate"):
        validate_system_manifest({"schema_version": "1.0", "systems": [system(), system()]})


def test_schema_matches_outcome_semantics():
    import itertools
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/trajectory_schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)
    for values in itertools.product((True, False, None), repeat=4):
        accepted = []
        for status in ("success", "failure", "partial", "unknown"):
            record = trajectory(values, status)
            schema_valid = validator.is_valid(record)
            try:
                validate_trajectory(record)
                python_valid = True
            except ValueError:
                python_valid = False
            assert schema_valid == python_valid, (values, status)
            if schema_valid:
                accepted.append(status)
        if values[0] is False and any(value is True for value in values[1:]):
            assert accepted == []
        elif values[0] is True and values[3] is True:
            assert accepted == ["success", "unknown"]
        elif values[0] is True and values[3] is False:
            assert accepted == ["failure", "partial", "unknown"]
        else:
            assert accepted == ["unknown"]


def test_failed_task_execution_valid_does_not_infer_partial():
    record = trajectory((True, False, True, False), "failure")
    assert validate_trajectory(record)["status"] == "failure"
    record["status"] = "partial"  # The scorer may explicitly distinguish progress.
    assert validate_trajectory(record)["status"] == "partial"


def test_native_qualified_checkpoint_requires_immutable_references():
    record = trajectory()
    checkpoint = {"checkpoint_id": "native-1", "replay_qualified": True}
    record["checkpoint_refs"] = [checkpoint]
    with pytest.raises(ValueError):
        validate_trajectory(record)
    checkpoint["checkpoint_identity"] = "1" * 64
    with pytest.raises(ValueError):
        validate_trajectory(record)
    checkpoint["qualification_report_ref"] = "2" * 64
    assert validate_trajectory(record)["checkpoint_refs"][0]["replay_qualified"] is True
    checkpoint["qualification_report_ref"] = "mutable-path.json"
    with pytest.raises(ValueError):
        validate_trajectory(record)


@pytest.mark.parametrize("field", ["target_resolved", "regression_free", "task_completed"])
def test_unknown_invalid_execution_cannot_claim_verified_success(field):
    record = trajectory((False, None, None, None), "unknown")
    record["outcome"][field] = True
    with pytest.raises(ValueError, match="invalid execution"):
        validate_trajectory(record)
