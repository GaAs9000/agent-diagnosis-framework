"""Run counterfactual continuation verification on SWE-bench Pro cohort."""
import sys
import os
import time
from pathlib import Path
import json

base_dir = Path("/Users/jiashen/Desktop/问题诊断")
l1_src = base_dir / "L1_sandbox" / "C3_Experimental_Sandbox" / "src"
l2_src = base_dir / "L2_Behavioral Event Localization" / "L2_Sandbox" / "src"

sys.path.insert(0, str(l1_src))
sys.path.insert(0, str(l2_src))

from l2_sandbox.report import verify_episode

l1_run = base_dir / "L1_sandbox" / "C3_Experimental_Sandbox" / "runs" / "swebench_pro_qwen36_cohort" / "run_task1_s42"
l2_dir = base_dir / "L2_Behavioral Event Localization" / "L2_Sandbox" / "diagnoses" / "swebench_pro_qwen36" / "run_task1_s42"

req_text = "Fix the Ansible issue by implementing __or__ and __ror__ in VarsWithSources and save the patch to /workspace/submission/patch.diff"
k_target = 4
n_branches = 4

print(f"=== Starting Continuation Verification ===")
print(f"Parent L1 run: {l1_run}")
print(f"Diagnosis dir: {l2_dir}")
print(f"Checkpoint k: {k_target}")
print(f"Arms: plain, rule (n={n_branches} each, total 8 branches)")
print(f"Requirement: {req_text}")

t0 = time.monotonic()
res = verify_episode(
    l1_run,
    output_dir=l2_dir,
    mode="intervention",
    k=k_target,
    arms=["plain", "rule"],
    n=n_branches,
    outcome="total",
    requirement=req_text,
    execute=True,
    max_concurrency=4,
)
elapsed = time.monotonic() - t0
print(f"\nVerification completed in {elapsed:.2f}s")
print(json.dumps(res, indent=2, default=str))
