#!/usr/bin/env python3
"""Freeze the fixed eight-task development bundle for integrated delivery.

The bundle is new and immutable. Historical R1/R1.1 bundles are read-only
inputs; this script never changes them.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from c3_sandbox.r1.officeval_selection import static_inventory

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "Agent_Latent_Capability_Audit_Kit/data/raw/OmegaUse-OfficeVal"
ASSET_CACHE = ROOT.parent / "Agent_Latent_Capability_Audit_Kit/data/cache/officeval_assets"
HISTORICAL_ASSETS = ROOT / "runs/e2-r1-1/asset-downloads/task_files"
OLD_BUNDLE = ROOT / "runs/r1-build-context/frozen"
TASK_IDS = ("officeval_002", "officeval_003", "officeval_006", "officeval_012",
            "officeval_015", "officeval_018", "officeval_033", "officeval_037")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_file(src: Path, dst: Path, files: dict):
    if src.is_symlink() or not src.is_file() or src.stat().st_size == 0:
        raise ValueError(f"missing, empty or symlink source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    files[dst.relative_to(dst.parents[3] if len(dst.parents) > 3 else dst.parent).as_posix()] = {
        "sha256": sha(dst), "size_bytes": dst.stat().st_size}


def _asset(task_id: str, name: str) -> Path:
    candidates = [ASSET_CACHE / task_id / name, HISTORICAL_ASSETS / task_id / name]
    for path in candidates:
        if path.is_file() and not path.is_symlink():
            return path
    raise FileNotFoundError(f"asset not found for {task_id}: {name}")


def _task006_inventory(verifier: Path) -> dict:
    tree = ast.parse(verifier.read_text(encoding="utf-8-sig"))
    node = next((n for n in tree.body
                 if isinstance(n, (ast.Assign, ast.AnnAssign))
                 and any(isinstance(t, ast.Name) and t.id == "ALL_RULES"
                         for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))), None)
    if node is None:
        raise ValueError("officeval_006 missing static ALL_RULES")
    values = ast.literal_eval(node.value)
    rules = []
    for label, delta in values:
        if not isinstance(label, str) or type(delta) is not int or delta == 0:
            raise ValueError("officeval_006 has nonliteral rule")
        label = re.sub(r"^[+-]\d+\s*", "", label)
        rules.append({"rule": label, "max_delta": delta})
    return {"expected_rules": rules, "max_score": sum(r["max_delta"] for r in rules if r["max_delta"] > 0),
            "official_max_score": sum(r["max_delta"] for r in rules if r["max_delta"] > 0),
            "completion_threshold": sum(r["max_delta"] for r in rules if r["max_delta"] > 0),
            "failed_rule_aliases": [], "unassessed_aliases": [], "required_unassessed_rules": [],
            "inventory_derivation": "pinned AST ALL_RULES literal; scorer never executed",
            "verifier_sha256": sha(verifier)}


def build(destination: Path) -> dict:
    destination = destination.absolute()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".integrated-bundle-", dir=destination.parent))
    files = {}
    try:
        # Preserve pinned source metadata and make the complete source tree
        # available to the grader build context without modifying the checkout.
        for name in ("LICENSE", "NOTICE", "requirements.txt", "pyproject.toml"):
            src = SOURCE / name
            dst = temp / "source_metadata" / name
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
            files[str(dst.relative_to(temp))] = {"sha256": sha(dst), "size_bytes": dst.stat().st_size}
        old = json.loads((OLD_BUNDLE / "manifest.json").read_text())
        manifest = {"schema_version": 1, "selection": "C3-E2 integrated fixed development bundle",
                    "bundle_root": str(destination), "source_commit": old.get("source_commit"),
                    "source_tree": old.get("source_tree"), "asset_dataset_revision": old.get("asset_dataset_revision"),
                    "task_ids": list(TASK_IDS), "tasks": {}, "files": {}}
        for task_id in TASK_IDS:
            task_file = SOURCE / "data/tasks" / f"{task_id}.json"
            verifier = SOURCE / "verifiers" / f"{task_id}_verifier.py"
            task = json.loads(task_file.read_text())
            if task_id in {"officeval_002", "officeval_003"}:
                metadata = dict(old["tasks"][task_id])
                old_root = OLD_BUNDLE
                input_files = metadata["input_files"]
                for item in input_files:
                    src = old_root / item["path"]
                    rel = Path(item["path"])
                    dst = temp / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
                    files[str(rel)] = {"sha256": sha(dst), "size_bytes": dst.stat().st_size}
                for rel in (metadata["verifier_path"], metadata["task_path"], metadata["rubric_path"]):
                    src = old_root / rel; dst = temp / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
                    files[str(rel)] = {"sha256": sha(dst), "size_bytes": dst.stat().st_size}
                metadata["development"] = True
            else:
                origin = [item["dest"] for item in task["origin_files"]]
                inputs = []
                for name in origin:
                    src = _asset(task_id, name); rel = Path("inputs") / task_id / name
                    dst = temp / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
                    files[str(rel)] = {"sha256": sha(dst), "size_bytes": dst.stat().st_size}
                    inputs.append({"name": name, "path": str(rel), **files[str(rel)]})
                inventory = _task006_inventory(verifier) if task_id == "officeval_006" else static_inventory(task_id, verifier)
                if task_id in {"officeval_006", "officeval_012"}:
                    expected = task["origin_files"][1]["dest"]
                elif task_id in {"officeval_015", "officeval_018", "officeval_033"}:
                    expected = task["origin_files"][0]["dest"]
                else:
                    expected = "澜桥市高三英语试题.docx"
                metadata = {"official_id": task_id.removeprefix("officeval_"), "task_prompt": task["instruction"],
                            "input_files": inputs, "expected_output": expected, "expected_rules": inventory["expected_rules"],
                            "max_score": inventory["max_score"], "official_max_score": inventory.get("official_max_score", inventory["max_score"]),
                            "completion_threshold": inventory.get("completion_threshold", inventory["max_score"]),
                            "failed_rule_aliases": inventory.get("failed_rule_aliases", []),
                            "unassessed_aliases": inventory.get("unassessed_aliases", []),
                            "required_unassessed_rules": inventory.get("required_unassessed_rules", []),
                            "verifier_path": f"verifiers/{task_id}_verifier.py", "task_path": f"tasks/{task_id}.json",
                            "rubric_path": f"rubrics/{task_id}.json", "development": True,
                            "exact_output_name": True}
                for rel_src, rel_dst in [(task_file, Path(metadata["task_path"])),
                                         (SOURCE / "data/rubrics" / f"{task_id}.json", Path(metadata["rubric_path"])),
                                         (verifier, Path(metadata["verifier_path"]))]:
                    dst = temp / rel_dst; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(rel_src, dst)
                    files[str(rel_dst)] = {"sha256": sha(dst), "size_bytes": dst.stat().st_size}
            manifest["tasks"][task_id] = metadata
        manifest["files"] = files
        manifest["source_files_sha256"] = hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        files["manifest.json"] = {"sha256": sha(temp / "manifest.json"), "size_bytes": (temp / "manifest.json").stat().st_size}
        for p in temp.rglob("*"):
            if p.is_file(): p.chmod(0o444)
        os.rename(temp, destination)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True); raise
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("destination", type=Path)
    args = parser.parse_args(); print(json.dumps(build(args.destination), ensure_ascii=False, indent=2))
