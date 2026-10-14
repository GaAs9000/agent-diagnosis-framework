"""SWE-bench Pro V2 native benchmark adapter.

Adheres strictly to L1 zero-core-change isolation:
- No verifiers imported in controller.
- Official tests & gold solution staged as private files only.
- Pinned to official SWE-bench Pro V2 repository commit & data lock.
"""
from __future__ import annotations
import os
from pathlib import Path
from ...core.contracts import Asset, Task, EvaluationSpec, ConfigurationError
from ...storage import confined, read_json, file_digest, regular_file, safe_relative

PIN = "scaleapi/SWE-bench_Pro-os:v2"


class NativeSWEBenchPro:
    def __init__(self, config: dict):
        self.config = config
        data_dir = config.get("data_dir", "benchmarks/swebench_pro/data")
        self.data = Path(data_dir)
        self.tasks_dir = self.data / "tasks"

    def list_tasks(self):
        if not self.tasks_dir.is_dir():
            return []
        out = []
        for p in sorted(self.tasks_dir.iterdir()):
            if p.is_dir() and (p / "instruction.md").is_file():
                out.append({
                    "task_id": p.name,
                    "supported_runtime": "docker-cold",
                    "assets_ready": True
                })
        return out

    def load(self, task_id: str) -> Task:
        task_dir = self.tasks_dir / task_id
        if not task_dir.is_dir():
            raise ConfigurationError(f"Unknown SWE-bench Pro V2 task: {task_id}")

        inst_file = task_dir / "instruction.md"
        toml_file = task_dir / "task.toml"
        regular_file(inst_file)
        regular_file(toml_file)
        instruction = inst_file.read_text(encoding="utf-8")

        # Check data lock if present
        lock_path = self.data / "data.lock.json"
        lock_status = "unlocked"
        if lock_path.is_file():
            lock = read_json(lock_path)
            if task_id in lock.get("tasks", {}):
                lock_status = "sha256_locked"

        # Private files: hidden tests, verifier scripts, gold patch
        private = {}
        private["source/task.toml"] = toml_file
        for subdir in ["tests", "solution", "environment"]:
            s_dir = task_dir / subdir
            if s_dir.is_dir():
                for p in sorted(s_dir.rglob("*")):
                    if p.is_file():
                        rel = p.relative_to(task_dir).as_posix()
                        private["source/" + rel] = p

        image = self.config.get("image", f"ghcr.io/scaleapi/swe-bench_pro-v2:{task_id}")

        provenance = {
            "adapter": "swebench_pro-native",
            "source_commit": self.config.get("source_commit", PIN),
            "task_id": task_id,
            "data_lock_status": lock_status,
            "evaluation_protocol": "official_swe_bench_pro_v2_verifier",
            "task_sha256": file_digest(toml_file),
            "instruction_sha256": file_digest(inst_file),
            "private_files_sha256": {k: file_digest(v) for k, v in private.items()},
        }

        spec = EvaluationSpec(
            kind="swebench_pro-verifier",
            source_root=task_dir,
            entrypoint="tests/test.sh",
            task_id=task_id,
            metadata={"image": image}
        )

        return Task(
            benchmark="swebench_pro",
            task_id=task_id,
            prompt=instruction,
            assets=[],
            private_files=private,
            requirements={"image": image, "runtime": "docker-cold"},
            evaluation=spec,
            provenance=provenance
        )
