"""Run multi-turn diagnosis across Analyst Groups A, B, C for SWE-bench Pro cohort."""
import sys
import time
from pathlib import Path
import json

base_dir = Path("/Users/jiashen/Desktop/问题诊断")
l1_src = base_dir / "L1_sandbox" / "C3_Experimental_Sandbox" / "src"
l2_src = base_dir / "L2_Behavioral Event Localization" / "L2_Sandbox" / "src"

sys.path.insert(0, str(l1_src))
sys.path.insert(0, str(l2_src))

from l2_sandbox.report import diagnose_episode

cohort_dir = base_dir / "L1_sandbox" / "C3_Experimental_Sandbox" / "runs" / "swebench_pro_qwen36_cohort"
cache_dir = base_dir / "L2_Behavioral Event Localization/L2_Sandbox/cache"
out_base = base_dir / "L2_Behavioral Event Localization/L2_Sandbox/diagnoses/swebench_pro_qwen36"
cfg_path = "configs/analyst.qwen36.json"

analysts = [
    "rules",
    f"llm_raw:{cfg_path}",
    f"llm:{cfg_path}",
]

runs = ["run_task1_s42", "run_task1_s101", "run_task2_s42"]
summary = {}

# Ensure running from L2_Sandbox dir for relative config path resolution
l2_sandbox_dir = base_dir / "L2_Behavioral Event Localization/L2_Sandbox"
import os
os.chdir(str(l2_sandbox_dir))

for r in runs:
    print(f"\n==========================================")
    print(f"Diagnosing {r}...")
    print(f"==========================================")
    t0 = time.monotonic()
    l1_run_path = cohort_dir / r
    output_dir = out_base / r
    
    res = diagnose_episode(
        l1_run_path,
        output_dir=output_dir,
        analysts=analysts,
        execute=True,
        cohort_dir=cohort_dir,
        cache_dir=cache_dir,
    )
    elapsed = time.monotonic() - t0
    print(f"Diagnosed {r} in {elapsed:.2f}s")
    summary[r] = {
        "elapsed_s": elapsed,
        "episode_id": res.get("episode_id"),
        "defects": res.get("defects"),
        "analysts": list((res.get("scores") or {}).get("analysts", {}).keys()),
        "scores": (res.get("scores") or {}).get("analysts"),
    }

print("\n=== DIAGNOSIS SUMMARY ===")
print(json.dumps(summary, indent=2))
