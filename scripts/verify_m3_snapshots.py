"""Snapshot scoring & Y3 consistency verification for SWE-bench Pro cohort."""
import sys
from pathlib import Path
import json

base_dir = Path("/Users/jiashen/Desktop/问题诊断")
l1_src = base_dir / "L1_sandbox" / "C3_Experimental_Sandbox" / "src"
l2_src = base_dir / "L2_Behavioral Event Localization" / "L2_Sandbox" / "src"

sys.path.insert(0, str(l1_src))
sys.path.insert(0, str(l2_src))

from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.grades import grade_episode_snapshots
from l2_sandbox.defects import build_defect_table

cohort_dir = base_dir / "L1_sandbox" / "C3_Experimental_Sandbox" / "runs" / "swebench_pro_qwen36_cohort"
cache_dir = base_dir / "L2_Behavioral Event Localization" / "L2_Sandbox" / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

runs = ["run_task1_s42", "run_task1_s101", "run_task2_s42"]
pkgs = [L1Package(cohort_dir / r) for r in runs]

print("=== Starting Snapshot Scoring on Baseline Runs ===")
results = {}

for r_name, pkg in zip(runs, pkgs):
    print(f"\nProcessing {r_name} (Episode: {pkg.episode_id}, Task: {pkg.task_id})...")
    extra = [p for p in pkgs if p.episode_id != pkg.episode_id]
    
    grade_rows = grade_episode_snapshots(pkg, cache_dir=cache_dir, extra_runs=extra)
    print(f"Total grade rows: {len(grade_rows)}")
    
    term_row = grade_rows[0]
    assert term_row["kind"] == "terminal_submission"
    official_y = pkg.outcome()["y"]
    
    cp_rows = [r for r in grade_rows if r["kind"] == "checkpoint"]
    last_cp_row = cp_rows[-1]
    s_T = last_cp_row["score"]
    
    print(f"Official Y: {official_y}, Terminal Snapshot s(T) [cp{last_cp_row['k']:06d}]: {s_T}")
    y3_match = (s_T == official_y)
    print(f"Y3 Consistency: {'MATCH' if y3_match else 'MISMATCH'}")
    
    # Calculate delivery gap g = max_k s(k) - Y
    scores_k = [r["score"] for r in cp_rows]
    max_s = max(scores_k)
    g = max_s - official_y
    print(f"max_k s(k): {max_s}, Delivery Gap g: {g}")
    
    defects_doc = build_defect_table(pkg, grade_rows, [], extra_runs=extra)
    defects = defects_doc.get("defects", [])
    print(f"Defects ({len(defects)}):")
    for d in defects:
        print(f"  - Defect: {d['defect']}, Kind: {d['kind']}, Deduction: {d['deduction']}, Turn: {d.get('turn')}")
        
    results[r_name] = {
        "episode_id": pkg.episode_id,
        "task_id": pkg.task_id,
        "official_y": official_y,
        "s_T": s_T,
        "y3_match": y3_match,
        "max_s": max_s,
        "delivery_gap_g": g,
        "checkpoint_count": len(cp_rows),
        "defects": defects,
        "k_star": defects_doc.get("k_star"),
    }

print("\n=== SUMMARY RESULTS ===")
print(json.dumps(results, indent=2))
