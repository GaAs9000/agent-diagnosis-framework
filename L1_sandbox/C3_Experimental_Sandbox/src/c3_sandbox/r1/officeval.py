"""Pinned OfficeVal source bundle and fail-closed C3 operational adapter."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

SOURCE_COMMIT = "5f3da616ac6a0bc30ca35e9678f8f1eb255bd91a"
TASK_IDS = ("officeval_003", "officeval_002")
def _workspace_source_root(module_path: Path) -> Path | None:
    # The grader copies this module to /opt/grader, which has no source checkout.
    sandbox = next((parent for parent in module_path.resolve().parents
                    if parent.name == "C3_Experimental_Sandbox"), None)
    return (sandbox.parent / "Agent_Latent_Capability_Audit_Kit/data/raw/OmegaUse-OfficeVal"
            if sandbox else None)


SOURCE_ROOT = _workspace_source_root(Path(__file__))
ASSET_ROOT = SOURCE_ROOT.parents[1] / "cache/officeval_assets" if SOURCE_ROOT else None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _literal(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal(node.left) + _literal(node.right)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(n) for n in node.elts]
    return ast.literal_eval(node)


def expected_inventory(task_id: str, verifier: Path) -> list[dict]:
    """Extract only pinned static metadata; never execute a scorer to freeze rules."""
    if task_id == "officeval_002":
        for node in ast.parse(verifier.read_text()).body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "_DIM2_RULES":
                return [{"rule": rule, "max_delta": points if kind == "bonus" else -points}
                        for kind, rule, points, _ in _literal(node.value)]
        raise ValueError("missing pinned rule inventory")
    if task_id != "officeval_003":
        raise ValueError("task outside frozen dev selection")
    return [{"rule": rule, "max_delta": delta} for rule, delta in [
        ("文档表格中没有空白行", 5),
        ("文档表格中所有文本：下方都出现横线；横线长度与表格上边框长度一致；颜色为黑色，类型为细实线", 5),
        ("文档表格中所有横线：相邻两条横线之间的距离几乎相等，只允许有0.15cm的误差", 5),
        ("文中出现图片或者不可编辑的部分", -5),
        ("文档第一页顶部不满足其中任意一项", -5),
        ('"会议内容："下方第一行出现缩进', -1),
        ('文中没有出现"侯景澜：上月两项重点现场事项整改推进如下："文本内容', -1),
        ('"二月份作业偏差共发生若干件，具体情况如下："与"四、研究后续事项"之间的文本出现缩进', -1),
    ]]


def freeze_officeval(destination: Path) -> dict:
    """Create a new read-only bundle atomically, without downloads or target edits."""
    if SOURCE_ROOT is None or ASSET_ROOT is None:
        raise RuntimeError("source freezing requires the workspace OfficeVal checkout")
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(SOURCE_ROOT), *args], text=True).strip()
    if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("official source must be clean at pinned commit")
    versions = {"python": platform.python_version()}
    for name in ("python-docx", "lxml"):
        versions[name] = importlib.metadata.version(name)
    manifest = {"schema_version": 1, "selection": "E2-R1 existing dev selection; frozen before model outcomes",
                "bundle_root": str(destination), "source_commit": SOURCE_COMMIT,
                "source_tree": git("rev-parse", "HEAD^{tree}"), "package_versions": versions,
                "asset_dataset_revision": "cd6ba6d8fb83b3fb551e24eebc20e1fb0bd154a5",
                "tasks": {}, "files": {}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".officeval-freeze-", dir=destination.parent))
    try:
        def copy(source, relative):
            if source.is_symlink() or not source.is_file() or not source.stat().st_size:
                raise ValueError(f"missing, empty or symlink source: {source}")
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            manifest["files"][relative] = {"sha256": _sha(target), "size_bytes": target.stat().st_size}
        for name in ("LICENSE", "NOTICE", "requirements.txt", "pyproject.toml"):
            copy(SOURCE_ROOT / name, "source_metadata/" + name)
        for task_id in TASK_IDS:
            task = json.loads((SOURCE_ROOT / "data/tasks" / (task_id + ".json")).read_text())
            inputs = []
            for entry in task["origin_files"]:
                name = entry["dest"]
                if Path(name).name != name:
                    raise ValueError("unsafe input name")
                relative = f"inputs/{task_id}/{name}"
                copy(ASSET_ROOT / task_id / name, relative)
                inputs.append({"name": name, "path": relative, **manifest["files"][relative]})
            verifier = SOURCE_ROOT / "verifiers" / (task_id + "_verifier.py")
            rules = expected_inventory(task_id, verifier)
            copy(verifier, f"verifiers/{verifier.name}")
            for kind in ("tasks", "rubrics"):
                copy(SOURCE_ROOT / "data" / kind / (task_id + ".json"), f"{kind}/{task_id}.json")
            manifest["tasks"][task_id] = {
                "official_id": task_id.removeprefix("officeval_"),
                "task_prompt": task["instruction"], "input_files": inputs,
                "expected_output": next(i["name"] for i in inputs if i["name"].lower().endswith(".docx")),
                "expected_rules": rules,
                "unassessed_aliases": ([{"slot": 7, "alias": "未找到指定的章节标记，跳过检查",
                    "canonical_rule": rules[7]["rule"], "max_delta": -1,
                    "source_commit": SOURCE_COMMIT, "source_lines": "954-960"}]
                    if task_id == "officeval_003" else []),
                "max_score": sum(r["max_delta"] for r in rules if r["max_delta"] > 0),
                "task_path": f"tasks/{task_id}.json", "rubric_path": f"rubrics/{task_id}.json",
                "verifier_path": f"verifiers/{verifier.name}",
            }
        # Detect tracked-source changes during the copy as well.
        if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
            raise ValueError("official source changed during freeze")
        manifest["source_files_sha256"] = hashlib.sha256(
            json.dumps(manifest["files"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        for file in temporary.rglob("*"):
            if file.is_file():
                file.chmod(0o444)
        # Destination must be new. rename, rather than copying, publishes the complete bundle.
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def task_public_view(manifest: dict, task_id: str) -> dict:
    task = manifest["tasks"][task_id]
    return {"task_prompt": task["task_prompt"], "input_files": [dict(i) for i in task["input_files"]],
            "expected_output": task["expected_output"]}


def normalize_official(raw, task_id: str, task: dict) -> dict:
    """Keep official JSON intact; completion is a separately named C3 convention."""
    result = {"official_raw": raw, "validity": "infrastructure_invalid", "c3_operational_completed": False,
              "outcome": "infrastructure_invalid", "reason": "malformed_official_schema",
              "coverage_complete": None, "unassessed_rules": []}
    if not isinstance(raw, dict):
        return result
    if raw.get("status") == "error":
        result["reason"] = "official_verifier_error"
        return result
    if (raw.get("status") != "ok" or raw.get("id") != task["official_id"] or raw.get("error") is not None
            or "error" not in raw or type(raw.get("dim1_pass")) is not bool
            or not isinstance(raw.get("file_name"), str) or not isinstance(raw.get("dim1_reason"), str)
            or type(raw.get("total_score")) is not int or type(raw.get("max_score")) is not int
            or not isinstance(raw.get("dim2_items"), list)):
        return result
    official_maximum = task.get("official_max_score", task["max_score"])
    # Official scorers short-circuit dimension 1; empty dim2 is then legitimate.
    if raw["dim1_pass"] is False:
        if raw["dim2_items"] or raw["total_score"] != 0 or raw["max_score"] not in (0, official_maximum):
            return result
        result.update(validity="valid", outcome="task_failed", reason="dimension1_failed", coverage_complete=False)
        return result
    items = raw["dim2_items"]
    expected = Counter((r["rule"], r["max_delta"]) for r in task["expected_rules"])
    actual = []
    unassessed = list(task.get("required_unassessed_rules", []))
    aliases = {a["slot"]: a for a in task.get("unassessed_aliases", [])}
    failed_aliases = {(a["alias"], a["max_delta"]): a
                      for a in task.get("failed_rule_aliases", [])}
    for slot, item in enumerate(items):
        if (not isinstance(item, dict) or not isinstance(item.get("rule"), str)
                or type(item.get("max_delta")) is not int or type(item.get("delta")) is not int
                or type(item.get("hit")) is not bool or not isinstance(item.get("detail"), str)):
            return result
        if item["delta"] != (item["max_delta"] if item["hit"] else 0):
            result["reason"] = "inconsistent_rule_score"
            return result
        alias = aliases.get(slot)
        if alias and item["rule"] == alias["alias"] and item["max_delta"] == alias["max_delta"]:
            if item["hit"] is not False or item["delta"] != 0:
                result["reason"] = "invalid_unassessed_alias_score"
                return result
            actual.append((alias["canonical_rule"], alias["max_delta"]))
            unassessed.append(alias["canonical_rule"])
        else:
            failed_alias = failed_aliases.get((item["rule"], item["max_delta"]))
            if failed_alias:
                if item["hit"] is not False or item["delta"] != 0:
                    result["reason"] = "invalid_failed_alias_score"
                    return result
                actual.append((failed_alias["canonical_rule"], item["max_delta"]))
            else:
                actual.append((item["rule"], item["max_delta"]))
    if Counter(actual) != expected:
        result["reason"] = "unexpected_rule_inventory"
        return result
    if raw["total_score"] != sum(i["delta"] for i in items) or raw["max_score"] != official_maximum:
        result["reason"] = "inconsistent_total_score"
        return result
    known_failure = any(i["hit"] is not (i["max_delta"] > 0)
                        for i in items if i["max_delta"] != 0
                        and i["rule"] not in {a["alias"] for a in aliases.values()})
    complete = False if known_failure else (None if unassessed else True)
    result.update(validity="valid", c3_operational_completed=complete,
                  coverage_complete=not bool(unassessed), unassessed_rules=unassessed,
                  outcome="task_failed" if known_failure else ("unassessed" if unassessed else "completed"),
                  reason="requirements_unmet" if known_failure else ("incomplete_rule_coverage" if unassessed else "all_rules_met"))
    return result
