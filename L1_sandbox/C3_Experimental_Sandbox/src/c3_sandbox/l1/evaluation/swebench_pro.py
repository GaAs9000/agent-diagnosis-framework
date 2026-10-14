"""SWE-bench Pro V2 official verifier execution in isolated Docker container."""
from __future__ import annotations
import os
import json
from pathlib import Path
from ..core.contracts import CapabilityError
from ..runtime.docker import DockerRuntime
from ..storage import read_json, write_bytes, write_json, file_digest, regular_file


def interpret_swebench_pro(raw: dict, task_id: str, policy: str = "official_dimensions") -> dict:
    out = {
        "evaluation_validity": "invalid",
        "task_outcome": "unknown",
        "outcome_policy": policy,
        "official_dimensions": None,
        "completion_rate": None,
        "rubric_coverage": "fail_to_pass_and_pass_to_pass",
        "reason": "malformed_official_result"
    }
    if not isinstance(raw, dict):
        return out

    passed = bool(raw.get("passed", False))
    reward = float(raw.get("reward", 1.0 if passed else 0.0))

    out.update(
        evaluation_validity="valid",
        reason="official_result_retained",
        completion_rate=1.0 if passed else 0.0,
        official_dimensions={
            "passed": passed,
            "reward": reward,
            "score": reward,
            "status": "ok" if passed else "failed"
        }
    )
    out["task_outcome"] = "success" if passed else "failure"
    return out


class SWEBenchProEvaluator:
    def __init__(self, config: dict):
        self.config = config

    def evaluate(self, spec, submission: Path, output: Path) -> dict:
        if spec.kind != "swebench_pro-verifier":
            raise CapabilityError("SWEBenchProEvaluator requires swebench_pro-verifier EvaluationSpec")

        output.mkdir(parents=True, exist_ok=False)
        image = spec.metadata.get("image", self.config.get("image", f"ghcr.io/scaleapi/swe-bench_pro-v2:{spec.task_id}"))

        runtime = DockerRuntime({
            "image": image,
            "memory": self.config.get("memory", "4g"),
            "cpus": self.config.get("cpus", 2),
            "pids_limit": self.config.get("pids_limit", 256),
            "output_limit_bytes": self.config.get("output_limit_bytes", 8388608),
            "read_only_rootfs": False,
            "user": "root"
        })
        provenance = runtime.preflight()

        grade_dir = output / "worker-output"
        grade_dir.mkdir(mode=0o777)
        if hasattr(os, "getuid") and os.getuid() == 0:
            os.chown(grade_dir, 65532, 65532)

        tests_dir = spec.source_root / "tests"
        if not tests_dir.is_dir():
            tests_dir = spec.source_root

        command = [
            "/bin/bash", "-c",
            (
                "set -o pipefail\n"
                "cd /app 2>/dev/null || cd /testbed 2>/dev/null\n"
                "# Fast clean reset of test code\n"
                "python3 - <<'PYRESET' > /tmp/reset_list.txt\n"
                "import subprocess, re\n"
                "CODE_DIRS = {'tests', 'test', '__tests__', 'spec', 'specs'}\n"
                "DATA_DIRS = {'testdata', 'test_data', 'fixtures', 'fixture', 'mocks', 'mock', '__snapshots__'}\n"
                "def is_test_code(p):\n"
                "    segs = p.split('/')\n"
                "    if any(s in DATA_DIRS for s in segs[:-1]): return False\n"
                "    return any(s in CODE_DIRS for s in segs[:-1])\n"
                "def git(a): return subprocess.run(['git'] + a, capture_output=True, text=True).stdout.splitlines()\n"
                "for p in git(['ls-files']):\n"
                "    if is_test_code(p): print('T ' + p)\n"
                "for p in git(['ls-files', '--others', '--exclude-standard']):\n"
                "    if is_test_code(p): print('U ' + p)\n"
                "PYRESET\n"
                "grep '^T ' /tmp/reset_list.txt | cut -c3- | xargs -r git checkout HEAD -- 2>/dev/null || true\n"
                "grep '^U ' /tmp/reset_list.txt | cut -c3- | xargs -r rm -f -- 2>/dev/null || true\n"
                "\n"
                "# Look for patch in submission\n"
                "APPLIED=0\n"
                "for p in /submission/patch.diff /submission/model.patch /submission/gold_patch.diff; do\n"
                "  if [ -f \"$p\" ] && [ -s \"$p\" ]; then\n"
                "    echo \"[VERIFIER] Applying patch from $p...\"\n"
                "    git apply -v \"$p\" || git apply -v --reject \"$p\" || true\n"
                "    APPLIED=1\n"
                "    break\n"
                "  fi\n"
                "done\n"
                "if [ \"$APPLIED\" -eq 0 ]; then\n"
                "  echo \"[VERIFIER] No patch file provided or empty patch (negative baseline).\"\n"
                "fi\n"
                "\n"
                "# Apply test patch\n"
                "echo \"[VERIFIER] Applying test patch...\"\n"
                "git apply -v /tests/test_patch.patch\n"
                "\n"
                "# Run selected tests\n"
                "export PATH=/app/bin:$PATH\n"
                "TEST_FILES=$(python3 -c \"\n"
                "import json\n"
                "with open('/tests/config.json') as f:\n"
                "    config = json.load(f)\n"
                "    test_files_raw = config.get('selected_test_files_to_run', [])\n"
                "    if isinstance(test_files_raw, str):\n"
                "        test_files = json.loads(test_files_raw)\n"
                "    elif isinstance(test_files_raw, list):\n"
                "        test_files = test_files_raw\n"
                "    else:\n"
                "        test_files = []\n"
                "    print(','.join(test_files))\n"
                "\" 2>/dev/null || echo \"\")\n"
                "\n"
                "set +e\n"
                "if [ -n \"$TEST_FILES\" ]; then\n"
                "    bash /tests/run_script.sh \"$TEST_FILES\" > /tmp/stdout.log 2> /tmp/stderr.log\n"
                "else\n"
                "    bash /tests/run_script.sh > /tmp/stdout.log 2> /tmp/stderr.log\n"
                "fi\n"
                "set -e\n"
                "\n"
                "# Run parser\n"
                "python3 /tests/parser.py /tmp/stdout.log /tmp/stderr.log /tmp/output.json\n"
                "cp /tmp/output.json /logs/verifier/output.json 2>/dev/null || true\n"
                "\n"
                "# Check results against config.json\n"
                "python3 << 'EVAL_EOF'\n"
                "import json, sys, ast\n"
                "results = json.load(open('/tmp/output.json'))\n"
                "config = json.load(open('/tests/config.json'))\n"
                "fail_to_pass_raw = config.get('fail_to_pass', [])\n"
                "pass_to_pass_raw = config.get('pass_to_pass', [])\n"
                "def parse_tests(td):\n"
                "    if isinstance(td, str):\n"
                "        try: return json.loads(td)\n"
                "        except Exception: pass\n"
                "        try: return ast.literal_eval(td)\n"
                "        except Exception: return []\n"
                "    elif isinstance(td, list): return td\n"
                "    return []\n"
                "fail_to_pass = set(parse_tests(fail_to_pass_raw))\n"
                "pass_to_pass = set(parse_tests(pass_to_pass_raw))\n"
                "passed_tests = {t.get('name', '') for t in results.get('tests', []) if t.get('status') == 'PASSED'}\n"
                "all_required = fail_to_pass | pass_to_pass\n"
                "success = all_required <= passed_tests\n"
                "print(f'Required: {len(all_required)}, Passed: {len(passed_tests)}')\n"
                "missing = all_required - passed_tests\n"
                "if missing:\n"
                "    print(f'Missing: {list(missing)}')\n"
                "with open('/logs/verifier/reward.txt', 'w') as f:\n"
                "    f.write('1\\n' if success else '0\\n')\n"
                "sys.exit(0 if success else 1)\n"
                "EVAL_EOF\n"
                "echo $? > /logs/verifier/test_exit_code.txt\n"
            )
        ]

        mounts = [
            (submission, "/submission", True),
            (tests_dir, "/tests", True),
            (grade_dir, "/logs/verifier", False)
        ]

        timeout = self.config.get("timeout_seconds", 300)
        process = runtime.container(mounts, command, timeout, cwd="/tmp")

        write_bytes(output, "stdout.bin", process.pop("stdout"))
        write_bytes(output, "stderr.bin", process.pop("stderr"))
        write_json(output, "process.json", process)
        write_json(output, "runtime.json", provenance)

        # Check reward.txt generated by eval logic
        reward_path = grade_dir / "reward.txt"
        passed = False
        reward = 0.0

        if reward_path.is_file():
            try:
                val = reward_path.read_text().strip()
                if val == "1":
                    passed = True
                    reward = 1.0
                elif val == "0":
                    passed = False
                    reward = 0.0
            except Exception:
                pass
        else:
            exit_code_path = grade_dir / "test_exit_code.txt"
            if exit_code_path.is_file():
                try:
                    c = int(exit_code_path.read_text().strip())
                    passed = (c == 0)
                    reward = 1.0 if passed else 0.0
                except Exception:
                    pass

        raw_result = {
            "task_id": spec.task_id,
            "passed": passed,
            "reward": reward,
            "score": reward,
            "status": "ok" if passed else "failed",
            "exit_code": 0 if passed else 1
        }

        test_output_json = grade_dir / "output.json"
        if test_output_json.is_file():
            try:
                raw_result["test_results"] = json.loads(test_output_json.read_text())
            except Exception:
                pass

        write_json(grade_dir, "official.json", raw_result)
        raw_sha = file_digest(grade_dir / "official.json")

        policy = self.config.get("outcome_policy", "official_dimensions")
        projection = interpret_swebench_pro(raw_result, spec.task_id, policy)

        out_grade = {
            "official_raw": raw_result,
            "official_raw_sha256": raw_sha,
            "projection": projection,
            "process": process,
            "runtime": provenance
        }
        write_json(output, "grade.json", out_grade)
        return out_grade
