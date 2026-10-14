import json
from pathlib import Path
import importlib.util

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_integrated_delivery.py"
spec = importlib.util.spec_from_file_location("run_integrated_delivery", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def config():
    return module._load(Path(__file__).parents[1] / "configs/integrated_end_to_end.json")


def test_execute_preflights_once_and_records_target_reference_serially(tmp_path):
    preflight_calls, runner_calls = [], []

    def preflight(value):
        preflight_calls.append(value)
        return {"bundle": {}, "task_ids": value["task_ids"]}

    def runner(task_id, model):
        runner_calls.append((task_id, model))
        return {"status": "complete"}

    result = module.execute_delivery(config(), tmp_path, runner=runner, preflight=preflight)
    assert result["status"] == "complete"
    assert len(preflight_calls) == 1
    assert len(runner_calls) == 16
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert all(row["status"] == "complete" for row in manifest["runs"])
    assert (tmp_path / "config.lock").is_file()


def test_execute_setup_failure_is_global_and_writes_manifest(tmp_path):
    def preflight(_):
        raise FileNotFoundError("missing verifier")

    try:
        module.execute_delivery(config(), tmp_path, preflight=preflight)
    except RuntimeError as exc:
        assert "missing verifier" in str(exc)
    else:
        raise AssertionError("setup failure must abort execution")
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["status"] == "setup_failed"
    assert manifest["runs"] == []
    assert not (tmp_path / "config.lock").exists()
