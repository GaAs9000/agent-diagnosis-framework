"""SWE-bench Pro V2 official verifier execution in isolated Docker container."""
from __future__ import annotations
import os
import ast
import math
import shlex
from pathlib import Path
from ..core.contracts import CapabilityError
from ..runtime.docker import DockerRuntime
from ..storage import read_json, write_bytes, write_json, file_digest, regular_file


# Run inside the worker: /tmp is a bounded tmpfs; host exports share one cap.
# Keep this script standalone so its actual copy behavior is testable offline.
_BOUNDED_EXPORT = """
import json, os, stat, sys
from pathlib import Path
source, target, limit = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
remaining = limit
receipt = {'limit_bytes': limit, 'truncated': False, 'files': {}}
for src, dst in [('stdout.log', 'test_stdout.log'), ('stderr.log', 'test_stderr.log'), ('output.json', 'output.json')]:
    fd = os.open(source / src, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('nonregular worker output')
        data = stream.read(remaining)
        size = os.fstat(stream.fileno()).st_size
    with open(target / dst, 'xb') as stream:
        stream.write(data)
    remaining -= len(data)
    receipt['files'][dst] = {'source_size_bytes': size, 'exported_size_bytes': len(data)}
    receipt['truncated'] |= size > len(data)
with open(target / 'output_export.json', 'x') as stream:
    json.dump(receipt, stream)
"""


def _required_tests(config: dict) -> set[str]:
    required = set()
    for key in ("fail_to_pass", "pass_to_pass"):
        value = config[key]
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
            raise ValueError("malformed required tests")
        required.update(value)
    if not required:
        raise ValueError("empty required tests")
    return required


def _passed_tests(results: dict) -> set[str]:
    if not isinstance(results, dict) or not isinstance(results.get("tests"), list):
        raise ValueError("missing or malformed parsed tests")
    names, passed = set(), set()
    for test in results["tests"]:
        if (not isinstance(test, dict) or not isinstance(test.get("name"), str)
                or not test["name"] or test["name"] in names
                or test.get("status") not in ("PASSED", "FAILED", "SKIPPED", "ERROR")):
            raise ValueError("malformed parsed test")
        names.add(test["name"])
        if test["status"] == "PASSED":
            passed.add(test["name"])
    return passed


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
    if not isinstance(raw, dict) or raw.get("task_id") != task_id or raw.get("error"):
        return out
    passed, reward = raw.get("passed"), raw.get("reward")
    if (type(passed) is not bool or type(reward) not in (int, float)
            or reward not in (0, 1) or not math.isfinite(reward)
            or reward != int(passed) or type(raw.get("score")) not in (int, float)
            or raw["score"] != reward or raw.get("status") != ("ok" if passed else "failed")
            or type(raw.get("exit_code")) is not int or raw["exit_code"] != (0 if passed else 1)):
        return out
    protocol = raw.get("protocol")
    if (not isinstance(protocol, dict)
            or any(not isinstance(protocol.get(key), str) for key in ("reward_text", "test_exit_text", "test_run_exit_text"))
            or protocol["reward_text"].strip() != str(int(passed))
            or protocol.get("test_exit_text", "").strip() != str(raw["exit_code"])
            or protocol.get("test_run_exit_text", "").strip() not in ("0", "1")
            or not isinstance(protocol.get("output_export"), dict)
            or protocol["output_export"].get("truncated") is not False):
        return out
    try:
        required = _required_tests(raw.get("test_config", {}))
        if (required <= _passed_tests(raw.get("test_results"))) != passed:
            return out
    except (ValueError, TypeError, KeyError, SyntaxError):
        return out

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

        output_limit = self.config.get("output_limit_bytes", 8388608)
        if type(output_limit) is not int or output_limit <= 0:
            raise CapabilityError("output_limit_bytes must be a positive integer")
        runtime = DockerRuntime({
            "image": image,
            "memory": self.config.get("memory", "4g"),
            "cpus": self.config.get("cpus", 2),
            "pids_limit": self.config.get("pids_limit", 256),
            "output_limit_bytes": output_limit,
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
                "set -e -o pipefail\n"
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
                "echo $? > /logs/verifier/test_run_exit.txt\n"
                "set -e\n"
                "\n"
                "# Run parser\n"
                "python3 /tests/parser.py /tmp/stdout.log /tmp/stderr.log /tmp/output.json\n"
                f"python3 -c {shlex.quote(_BOUNDED_EXPORT)} /tmp /logs/verifier {output_limit}\n"
                "\n"
                "# Check results against config.json\n"
                "set +e\n"
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
                "code=$?\n"
                "echo \"$code\" > /logs/verifier/test_exit_code.txt\n"
                "exit \"$code\"\n"
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

        # Exit codes alone never establish an official score. Keep every artifact
        # even when a partial worker output cannot support a valid projection.
        texts = {}
        error = None
        for name, filename in (("reward_text", "reward.txt"),
                               ("test_exit_text", "test_exit_code.txt"),
                               ("test_run_exit_text", "test_run_exit.txt")):
            try:
                path = grade_dir / filename
                regular_file(path)
                texts[name] = path.read_text(encoding="utf-8")
            except (OSError, ValueError, UnicodeError):
                texts[name] = None
        reward_text = texts["reward_text"]
        reward = float(reward_text.strip()) if isinstance(reward_text, str) and reward_text.strip() in ("0", "1") else None
        raw_result = {
            "task_id": spec.task_id,
            "passed": reward == 1 if reward is not None else None,
            "reward": reward, "score": reward,
            "status": "ok" if reward == 1 else "failed",
            "exit_code": process.get("container_exit_code"),
            "protocol": texts,
        }
        export_error = None
        try:
            regular_file(grade_dir / "output_export.json")
            export = read_json(grade_dir / "output_export.json")
            texts["output_export"] = export
            if (not isinstance(export, dict) or type(export.get("limit_bytes")) is not int
                    or export["limit_bytes"] != output_limit or type(export.get("truncated")) is not bool):
                export_error = "missing_or_malformed_output_export"
            elif export["truncated"]:
                export_error = "verifier_output_limit"
        except (OSError, ValueError, TypeError):
            export_error = "missing_or_malformed_output_export"
        try:
            regular_file(grade_dir / "output.json")
            raw_result["test_results"] = read_json(grade_dir / "output.json")
            raw_result["test_config"] = read_json(tests_dir / "config.json")
            required = _required_tests(raw_result["test_config"])
            expected_pass = required <= _passed_tests(raw_result["test_results"])
        except (OSError, ValueError, TypeError, KeyError, SyntaxError):
            error = "missing_or_malformed_official_result"
            expected_pass = None

        if process.get("stop_reason") is not None:
            error = process["stop_reason"]
        elif process.get("oom_killed") is not False or process.get("output_complete") is not True:
            error = "incomplete_verifier_execution"
        elif export_error:
            error = export_error
        elif reward is None:
            error = "missing_or_malformed_official_reward"
        elif error is None:
            expected_exit = "0" if expected_pass else "1"
            if (texts["test_exit_text"] is None or texts["test_exit_text"].strip() != expected_exit
                    or texts["test_run_exit_text"] is None or texts["test_run_exit_text"].strip() not in ("0", "1")
                    or type(process.get("container_exit_code")) is not int
                    or process["container_exit_code"] != int(expected_exit)
                    or (reward == 1) != expected_pass):
                error = "inconsistent_official_protocol"
        if error:
            raw_result.update(status="error", error=error)

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
        if error:
            projection["reason"] = error
            out_grade.update(status="infra_error", error_kind="infrastructure", error=error)
        write_json(output, "grade.json", out_grade)
        return out_grade
