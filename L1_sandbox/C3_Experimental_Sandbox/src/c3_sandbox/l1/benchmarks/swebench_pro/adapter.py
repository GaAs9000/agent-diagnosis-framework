"""SWE-bench Pro V2 native benchmark adapter.

Adheres strictly to L1 zero-core-change isolation:
- No verifiers imported in controller.
- Official tests & gold solution staged as private files only.
- Pinned to official SWE-bench Pro V2 repository commit & data lock.
"""
from __future__ import annotations
from pathlib import Path
from ...core.contracts import Task, EvaluationSpec, ConfigurationError
from ...storage import confined, read_json, file_digest, regular_file

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
        task_dir = confined(self.tasks_dir, task_id)
        if not task_dir.is_dir():
            raise ConfigurationError(f"Unknown SWE-bench Pro V2 task: {task_id}")

        inst_file = task_dir / "instruction.md"
        toml_file = task_dir / "task.toml"
        regular_file(inst_file)
        regular_file(toml_file)
        instruction = inst_file.read_text(encoding="utf-8")

        # A locked provenance claim covers the complete independent file set.
        lock_path = self.data / "data.lock.json"
        lock_status = "unlocked"
        if lock_path.exists() or lock_path.is_symlink():
            try:
                regular_file(lock_path)
                lock = read_json(lock_path)
                entries = lock.get("tasks", {}).get(task_id)
                if not isinstance(entries, dict) or not entries:
                    raise ValueError("task missing from data lock")
                actual = set()
                for path in task_dir.rglob("*"):
                    if "__pycache__" in path.parts or path.name == ".DS_Store" or path.suffix == ".pyc":
                        continue
                    if path.is_symlink():
                        raise ValueError("symlink task asset")
                    if not path.is_dir():
                        regular_file(path)
                        actual.add(path.relative_to(task_dir).as_posix())
                if actual != set(entries):
                    raise ValueError("task file set differs from data lock")
                for relative, expected in entries.items():
                    path = confined(task_dir, relative)
                    regular_file(path)
                    if (not isinstance(expected, dict)
                            or file_digest(path) != expected.get("sha256")
                            or type(expected.get("size_bytes")) is not int
                            or path.stat().st_size != expected["size_bytes"]):
                        raise ValueError(f"locked file mismatch: {relative}")
                lock_status = "sha256_locked"
            except (ValueError, OSError, TypeError, AttributeError) as exc:
                raise ConfigurationError(f"Invalid SWE-bench Pro data lock: {exc}") from exc

        # Private files: hidden tests, verifier scripts, gold patch
        private = {}
        private["source/task.toml"] = toml_file
        for subdir in ["tests", "solution", "environment"]:
            s_dir = task_dir / subdir
            if s_dir.is_dir():
                for p in sorted(s_dir.rglob("*")):
                    if "__pycache__" in p.parts or p.name == ".DS_Store" or p.suffix == ".pyc":
                        continue
                    if p.is_file():
                        rel = p.relative_to(task_dir).as_posix()
                        confined(task_dir, rel)
                        regular_file(p)
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
