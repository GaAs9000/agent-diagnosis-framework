"""Prospective Linux OfficeVal selection, using pinned AST metadata only.

No verifier is imported or executed here. Unsupported dependencies are rejected
before copying inputs; publication preserves the R1 grader manifest contract.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile

from .officeval import SOURCE_COMMIT, SOURCE_ROOT, _sha

SUPPORTED_TASK_IDS = tuple(f"officeval_{i}" for i in
                           ("010", "012", "015", "018", "021", "031", "033", "036", "037"))
ASSET_REVISION = "cd6ba6d8fb83b3fb551e24eebc20e1fb0bd154a5"
UNSUPPORTED = {
    "officeval_024": "required real pagination unavailable: info.com is always available=False",
    "officeval_035": "required +3 pagination check is constant False without rendering",
    "officeval_038": "full answer assessment cannot be distinguished from sample-only fallback in official JSON",
}
# Single designated output from the published instruction; references never enter grading.
OUTPUT_INDEX = {"officeval_012": 1, "officeval_021": 0, "officeval_031": 1,
                "officeval_036": 1}
OUTPUT_REASON = {
    "officeval_010": "edit the only supplied flowchart DOCX",
    "officeval_012": "edit the second DOCX thesis; first DOCX is the named formatting requirements",
    "officeval_015": "edit the first DOCX answer card; second DOCX supplies examination content",
    "officeval_018": "edit the first DOCX operations report; second DOCX supplies formatting requirements",
    "officeval_021": "edit the first DOCX; the second DOCX is explicitly labelled as template",
    "officeval_031": "place PDF questions into the supplied Word template, retaining its filename",
    "officeval_033": "edit the only supplied price catalogue DOCX",
    "officeval_036": "place PDF records into the supplied Word document, retaining its filename",
    "officeval_037": "convert the named target PDF to DOCX; the supplied DOCX is formatting reference",
}


def _value(node, names):
    """Restricted literal resolver, deliberately unable to call scorer code."""
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ValueError(f"nonliteral metadata: {node.id}")
        return names[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _value(node.left, names) + _value(node.right, names)
    if isinstance(node, ast.JoinedStr):
        return "".join(str(_value(x.value, names)) if isinstance(x, ast.FormattedValue)
                       else _value(x, names) for x in node.values)
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_value(x, names) for x in node.elts]
    return ast.literal_eval(node)


def _names(nodes, base=None):
    names = dict(base or {})
    for n in sorted(nodes, key=lambda n: getattr(n, "lineno", 0)):
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            try:
                value = _value(n.value, names)
            except (ValueError, TypeError, SyntaxError):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    names[target.id] = value
    return names


def _prefix(text):
    match = re.fullmatch(r"([+-]\d+)\s*[:：]?\s*(.+)", text, flags=re.S)
    if not match:
        raise ValueError(f"missing static score prefix: {text}")
    return match.group(2), int(match.group(1))


def static_inventory(task_id: str, verifier: Path) -> dict:
    """Derive exact active rules and provenance from the pinned verifier AST."""
    if task_id not in SUPPORTED_TASK_IDS:
        raise ValueError(UNSUPPORTED.get(task_id, "task not statically reviewed for R1.1"))
    tree = ast.parse(Path(verifier).read_text(encoding="utf-8-sig"))
    globals_ = _names(tree.body)
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    rules, aliases = [], []

    def add(rule, delta, node):
        if not isinstance(rule, str) or type(delta) is not int or delta == 0:
            raise ValueError("nonliteral or inactive rule metadata")
        rules.append({"rule": rule, "max_delta": delta,
                      "source_lines": f"{node.lineno}-{node.end_lineno}"})

    def calls(function, name):
        return sorted((n for n in ast.walk(function) if isinstance(n, ast.Call)
                       and ast.unparse(n.func) == name), key=lambda n: n.lineno)

    def unique(values):
        # Branches emit one item; repeated same-named items in 015 are separate.
        return list(dict.fromkeys(values))

    if task_id == "officeval_010":
        for n in calls(functions["evaluate_dimension2"], "self.item"):
            add(_value(n.args[1], globals_), _value(n.args[0], globals_), n)
    elif task_id == "officeval_018":
        for _, rule, delta, _ in globals_["DIM2_RULES"]:
            add(rule, delta, next(n for n in tree.body if isinstance(n, ast.Assign)
                                 and any(isinstance(t, ast.Name) and t.id == "DIM2_RULES" for t in n.targets)))
    elif task_id in {"officeval_015", "officeval_021"}:
        function = functions["check_dim2" if task_id.endswith("015") else "check_dimension2"]
        seen = set()
        for n in calls(function, "R.append" if task_id.endswith("015") else "score_items.append"):
            pair = (_value(n.args[0].elts[1], globals_), _value(n.args[0].elts[0], globals_))
            if task_id.endswith("015") or pair not in seen:
                add(*pair, n)
                seen.add(pair)
    elif task_id in {"officeval_012", "officeval_031"}:
        method = "_check_dimension2" if task_id.endswith("012") else "check_dimension2"
        for dispatch in sorted((n for n in ast.walk(functions[method]) if isinstance(n, ast.Call)
                                and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                                and n.func.value.id == "self"), key=lambda n: n.lineno):
            function = functions[dispatch.func.attr]
            names = _names(list(ast.walk(function)), globals_)
            pairs = []
            for n in calls(function, "self._add_result" if task_id.endswith("012") else "self.add_d2"):
                pair = (_prefix(_value(n.args[0], names)) if task_id.endswith("012")
                        else (_value(n.args[1], names), _value(n.args[2], names)))
                pairs.append(pair)
            pairs = unique(pairs)
            if len(pairs) != 1:
                raise ValueError(f"ambiguous static rule in {function.name}: {pairs}")
            add(*pairs[0], function)
    elif task_id == "officeval_033":
        function = functions["evaluate_positive_points"]
        names = _names(list(ast.walk(function)), globals_)
        for rule, delta, _ in names["groups"]:
            add(rule, delta, function)
        for function in (function, functions["evaluate_negative_points"]):
            for n in calls(function, "CheckResult"):
                values = {kw.arg: kw.value for kw in n.keywords}
                if isinstance(values["title"], ast.Name):
                    continue  # already enumerated by literal groups
                add(_value(values["title"], names), _value(values["score"], names), n)
    elif task_id == "officeval_036":
        seen = set()
        for n in sorted(ast.walk(tree), key=lambda n: getattr(n, "lineno", 0)):
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "ScorePoint":
                pair = (_value(n.args[0], globals_).removeprefix("扣分："), _value(n.args[1], globals_))
                if pair not in seen:
                    add(*pair, n)
                    seen.add(pair)
    elif task_id == "officeval_037":
        for dispatch in sorted((n for n in ast.walk(functions["dimension2_points"])
                                if isinstance(n, ast.Call)), key=lambda n: n.lineno):
            function = functions[dispatch.func.id]
            variants = unique([(_value(n.args[1], globals_), _value(n.args[0], globals_))
                               for n in calls(function, "PointResult")])
            canonical = variants[-1]
            add(*canonical, function)
            for rule, delta in variants[:-1]:
                if delta != canonical[1]:
                    raise ValueError("inconsistent branch score")
                aliases.append({"alias": rule, "max_delta": delta,
                                "canonical_rule": canonical[0], "must_hit": False,
                                "source_lines": f"{function.lineno}-{function.end_lineno}"})
    if not rules:
        raise ValueError("empty static inventory")
    positive = sum(r["max_delta"] for r in rules if r["max_delta"] > 0)
    # 015 evaluate sums signed scores; do not silently repair official arithmetic.
    official = sum(r["max_delta"] for r in rules) if task_id == "officeval_015" else positive
    if task_id == "officeval_031":
        maximum = functions["max_score"]
        returns = [n for n in maximum.body if isinstance(n, ast.Return)]
        if len(returns) != 1:
            raise ValueError("nonliteral official maximum")
        official = _value(returns[0].value, globals_)
        if positive != official:
            raise ValueError("static scorer maximum drift")
    return {"expected_rules": rules, "max_score": positive, "official_max_score": official,
            "completion_threshold": positive, "failed_rule_aliases": aliases,
            "unassessed_aliases": [], "required_unassessed_rules": [],
            "inventory_derivation": "pinned AST literals; scorer never executed",
            "verifier_sha256": _sha(verifier)}


def inspect_selection(task_ids, source_root=SOURCE_ROOT) -> dict:
    """Return all eligibility decisions before freezing or acquiring assets."""
    if source_root is None:
        raise RuntimeError("workspace source checkout required")
    source_root = Path(source_root)
    result = {}
    for task_id in task_ids:
        try:
            inventory = static_inventory(task_id, source_root / "verifiers" / f"{task_id}_verifier.py")
            task = json.loads((source_root / "data/tasks" / f"{task_id}.json").read_text())
            output = ("澜桥市高三英语试题.docx" if task_id == "officeval_037" else
                      task["origin_files"][OUTPUT_INDEX.get(task_id, 0)]["dest"])
            if not output.lower().endswith(".docx"):
                raise ValueError("designated deliverable must be one DOCX")
            result[task_id] = {"eligible": True, "expected_output": output,
                               "output_designation": {"basis": "published instruction and origin_files",
                                                      "reason": OUTPUT_REASON[task_id]}, **inventory}
        except (ValueError, KeyError, OSError, SyntaxError) as exc:
            result[task_id] = {"eligible": False, "reason": str(exc)}
    return result


def freeze_selection(destination: Path, task_ids, asset_root: Path, source_root=SOURCE_ROOT) -> dict:
    """Atomically publish a new immutable bundle with the existing R1 shape."""
    task_ids = tuple(task_ids)
    if not 1 <= len(task_ids) <= 12 or len(set(task_ids)) != len(task_ids):
        raise ValueError("one to twelve unique ordered tasks required")
    if source_root is None:
        raise RuntimeError("workspace source checkout required")
    source_root, asset_root = Path(source_root), Path(asset_root)
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source_root), *args], text=True).strip()
    def clean():
        if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
            raise ValueError("official source must be clean at pinned commit")
    clean()
    review = inspect_selection(task_ids, source_root)
    rejected = {k: v["reason"] for k, v in review.items() if not v["eligible"]}
    if rejected:
        raise ValueError(f"unsupported selection: {rejected}")
    versions = {"python": platform.python_version()}
    for name in ("python-docx", "lxml"):
        versions[name] = importlib.metadata.version(name)
    manifest = {"schema_version": 1, "selection": "E2-R1.1 prospective development pool; frozen before model outcomes",
                "bundle_root": str(destination), "source_commit": SOURCE_COMMIT,
                "source_tree": git("rev-parse", "HEAD^{tree}"), "package_versions": versions,
                "asset_dataset_revision": ASSET_REVISION, "tasks": {}, "files": {}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".officeval-selection-", dir=destination.parent))
    try:
        def copy(source, relative):
            if source.is_symlink() or not source.is_file() or not source.stat().st_size:
                raise ValueError(f"missing, empty or symlink source: {source}")
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            manifest["files"][relative] = {"sha256": _sha(target), "size_bytes": target.stat().st_size}
        for name in ("LICENSE", "NOTICE", "requirements.txt", "pyproject.toml"):
            copy(source_root / name, "source_metadata/" + name)
        for task_id in task_ids:
            task_path = source_root / "data/tasks" / f"{task_id}.json"
            task = json.loads(task_path.read_text())
            inputs = []
            for entry in task["origin_files"]:
                name = entry["dest"]
                if name in ("", ".", "..") or Path(name).name != name or "\\" in name:
                    raise ValueError("unsafe input name")
                relative = f"inputs/{task_id}/{name}"
                copy(asset_root / task_id / name, relative)
                inputs.append({"name": name, "path": relative, **manifest["files"][relative]})
            for kind in ("tasks", "rubrics"):
                copy(source_root / "data" / kind / f"{task_id}.json", f"{kind}/{task_id}.json")
            verifier_path = f"verifiers/{task_id}_verifier.py"
            copy(source_root / verifier_path, verifier_path)
            if manifest["files"][verifier_path]["sha256"] != review[task_id]["verifier_sha256"]:
                raise ValueError("verifier changed between static review and freeze")
            metadata = {k: v for k, v in review[task_id].items() if k != "eligible"}
            manifest["tasks"][task_id] = {"official_id": task_id.removeprefix("officeval_"),
                "task_prompt": task["instruction"], "input_files": inputs, **metadata,
                "source_group": task_id, "development": True, "exact_output_name": True,
                "dependencies": {"grader": ["python-docx", "lxml"], "platform": "linux",
                                 "com": False, "rendering": False,
                                 "actor_pdf_input": any(i["name"].lower().endswith(".pdf") for i in inputs)},
                "task_path": f"tasks/{task_id}.json", "rubric_path": f"rubrics/{task_id}.json",
                "verifier_path": verifier_path}
        clean()
        manifest["source_files_sha256"] = hashlib.sha256(json.dumps(
            manifest["files"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        for path in temporary.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest
