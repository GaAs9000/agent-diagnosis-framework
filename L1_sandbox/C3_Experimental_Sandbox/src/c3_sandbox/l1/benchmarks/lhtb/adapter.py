"""LHTB (Long-Horizon Terminal-Bench) native benchmark adapter.

Follows L1 benchmark isolation: no verifiers imported in controller,
hidden tests are private files only, strictly pinned source commit.
"""
from __future__ import annotations
from pathlib import Path
from ...core.contracts import Asset, Task, EvaluationSpec, ConfigurationError
from ...storage import confined, read_json, file_digest, regular_file, safe_relative

PIN = '7ca0cacea9bf6a0694eaad7306bed794eaaaabce'


class NativeLHTB:
    def __init__(self, config: dict):
        self.config = config
        data_dir = config.get('data_dir', 'benchmarks/lhtb/data')
        self.data = Path(data_dir)
        self.tasks_dir = self.data / 'tasks'

    def list_tasks(self):
        if not self.tasks_dir.is_dir():
            return []
        out = []
        for p in sorted(self.tasks_dir.iterdir()):
            if p.is_dir() and (p / 'instruction.md').is_file():
                out.append({
                    'task_id': p.name,
                    'supported_runtime': 'docker-cold',
                    'assets_ready': (p / 'bin/sokoban').is_file() if p.name == 'sokoban' else True
                })
        return out

    def load(self, task_id: str) -> Task:
        task_dir = confined(self.tasks_dir, task_id)
        if not task_dir.is_dir():
            raise ConfigurationError(f'Unknown LHTB task: {task_id}')

        inst_file = task_dir / 'instruction.md'
        toml_file = task_dir / 'task.toml'
        regular_file(inst_file)
        regular_file(toml_file)
        instruction = inst_file.read_text(encoding='utf-8')

        assets = []
        bin_dir = task_dir / 'bin'
        if bin_dir.is_dir():
            for p in sorted(bin_dir.iterdir()):
                if p.is_file():
                    safe_relative(p.name)
                    regular_file(p)
                    # Asset path relative to /workspace/input
                    assets.append(Asset(
                        name=f'bin/{p.name}',
                        path=p,
                        sha256=file_digest(p),
                        size_bytes=p.stat().st_size
                    ))

        # A lock claim requires checking every declared file and all task files.
        lock_path = self.data / 'data.lock.json'
        lock_status = 'unlocked'
        if lock_path.is_file():
            try:
                lock = read_json(lock_path)
                if lock.get('source_commit') != self.config.get('source_commit', PIN):
                    raise ValueError('source_commit mismatch')
                entries = lock.get('tasks', {}).get(task_id)
                if not isinstance(entries, dict) or not entries:
                    raise ValueError('task missing from data lock')
                actual = {p.relative_to(task_dir).as_posix()
                          for p in task_dir.rglob('*') if p.is_file()}
                if actual != set(entries):
                    raise ValueError('task file set differs from data lock')
                for relative, expected in entries.items():
                    path = confined(task_dir, relative)
                    regular_file(path)
                    if (not isinstance(expected, dict)
                            or file_digest(path) != expected.get('sha256')
                            or path.stat().st_size != expected.get('size_bytes')):
                        raise ValueError(f'locked file mismatch: {relative}')
                lock_status = 'sha256_locked'
            except (ValueError, OSError, TypeError, AttributeError) as exc:
                raise ConfigurationError(f'Invalid LHTB data lock: {exc}') from exc

        # Private files: hidden tests, verifier scripts, task config
        private = {}
        private['source/task.toml'] = toml_file
        tests_dir = task_dir / 'tests'
        if tests_dir.is_dir():
            for p in sorted(tests_dir.rglob('*')):
                if p.is_file():
                    rel = p.relative_to(task_dir).as_posix()
                    private['source/' + rel] = p

        image = self.config.get('image', f'zli12321/lhtb-{task_id}:20260615')

        provenance = {
            'adapter': 'lhtb-native',
            'source_commit': self.config.get('source_commit', PIN),
            'task_id': task_id,
            'data_lock_status': lock_status,
            'evaluation_protocol': 'official_terminal_bench_verifier',
            'task_sha256': file_digest(toml_file),
            'instruction_sha256': file_digest(inst_file),
            'private_files_sha256': {k: file_digest(v) for k, v in private.items()},
        }

        prompt = (
            f"{instruction}\n\n"
            f"[SANDBOX ENVIRONMENT & HELPER]\n"
            f"- A pre-configured CLI helper is available at /workspace/input/bin/sokoban.\n"
            f"- Run `/workspace/input/bin/sokoban status` or `/workspace/input/bin/sokoban move <SEQ>`.\n"
            f"- Moves are automatically saved and mirrored to /workspace/submission/moves.log for verification.\n"
        )

        spec = EvaluationSpec(
            kind='lhtb-verifier',
            source_root=task_dir,
            entrypoint='tests/test_outputs.py',
            task_id=task_id,
            metadata={'image': image, 'source_commit': provenance['source_commit'],
                      'data_lock_status': lock_status}
        )

        return Task(
            benchmark='lhtb',
            task_id=task_id,
            prompt=prompt,
            assets=tuple(assets),
            provenance=provenance,
            private_files=private,
            evaluation=spec,
            requirements={'runtime': 'docker-cold', 'image': image}
        )
