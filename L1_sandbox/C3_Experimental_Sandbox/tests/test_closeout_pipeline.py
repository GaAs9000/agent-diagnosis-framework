import json
from pathlib import Path

from c3_sandbox.integrated.closeout import run_closeout


def test_closeout_writes_explicit_not_run_artifacts_for_invalid_original(tmp_path):
    result = run_closeout([{"task_id": "officeval_002", "model_key": "target",
                            "status": "setup_failed", "result": {"execution_valid": False}}],
                          {"diagnostic": {}}, tmp_path)
    assert result["predictions"] == 0 and result["branches"] == 1
    assert json.loads((tmp_path / "branch_results.jsonl").read_text().splitlines()[0])["status"] == "NOT_RUN"
    assert json.loads((tmp_path / "experiment_cards.jsonl").read_text().splitlines()[0])["outcome"]["status"] == "NOT_RUN"
    assert (tmp_path / "objects/relation_graph.json").is_file()
