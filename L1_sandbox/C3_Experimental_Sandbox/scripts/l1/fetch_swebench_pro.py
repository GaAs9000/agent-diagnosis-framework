#!/usr/bin/env python3
"""Fetch and lock SWE-bench Pro V2 tasks from official scaleapi/SWE-bench_Pro-os repository."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

REPO_RAW = "https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/v2/tasks"
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "benchmarks/swebench_pro/data"
TASKS_DIR = DATA_DIR / "tasks"

TARGET_TASKS = [
    "instance_ansible__ansible-0ea40e09d1b35bcb69ff4d9cecf3d0defa4b36e8-v30a923fb5c164d6cd18280c02422f75e611e8fb2",
    "instance_ansible__ansible-0fd88717c953b92ed8a50495d55e630eb5d59166-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5",
]

TASK_FILES = [
    "instruction.md",
    "task.toml",
    "tests/config.json",
    "tests/parser.py",
    "tests/run_script.sh",
    "tests/test.sh",
    "tests/test_patch.patch",
    "solution/gold_patch.diff",
    "solution/solve.sh",
    "environment/Dockerfile",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def fetch_file(task_id: str, rel_path: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{REPO_RAW}/{task_id}/{rel_path}"
    req = urllib.request.Request(url, headers={"User-Agent": "c3-l1-preflight"})
    with urllib.request.urlopen(req, timeout=15) as r:
        content = r.read()
    dest.write_bytes(content)
    print(f"  [FETCHED] {rel_path} ({len(content)} bytes)")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock_data = {
        "version": "1.0",
        "benchmark": "swebench_pro",
        "source_repo": "scaleapi/SWE-bench_Pro-os",
        "split": "v2",
        "tasks": {}
    }

    for task_id in TARGET_TASKS:
        print(f"Staging task: {task_id}")
        t_dir = TASKS_DIR / task_id
        t_dir.mkdir(parents=True, exist_ok=True)
        task_hashes = {}
        for rel in TASK_FILES:
            dest = t_dir / rel
            if not dest.exists():
                fetch_file(task_id, rel, dest)
            task_hashes[rel] = {
                "sha256": sha256_file(dest),
                "size_bytes": dest.stat().st_size
            }
        lock_data["tasks"][task_id] = task_hashes

    lock_file = DATA_DIR / "data.lock.json"
    lock_file.write_text(json.dumps(lock_data, indent=2, sort_keys=True) + "\n")
    print(f"Locked {len(TARGET_TASKS)} tasks into {lock_file}")


if __name__ == "__main__":
    main()
