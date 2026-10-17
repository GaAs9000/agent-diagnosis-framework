#!/usr/bin/env python3
"""R6 Audit Script: Side-by-Side `study.py` Rule Version (`v1.0` vs `v1.1`) Sensitivity Analysis.

Scans all L1 run cohorts in `L1_sandbox/C3_Experimental_Sandbox/runs` without modifying any raw
`official_grade.json` or `manifest.json`, computes `study_outcome_by_version(pkg)`, lists every
affected episode/attempt where `v1.0 != v1.1`, and writes JSON + Markdown reports.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
L1_ROOT = REPO_ROOT / "L1_sandbox" / "C3_Experimental_Sandbox"
L2_ROOT = REPO_ROOT / "L2_Behavioral Event Localization" / "L2_Sandbox"
sys.path.insert(0, str(L1_ROOT / "src"))

from c3_sandbox.l1.evaluation.study import (  # noqa: E402
    DELIVERY_ERRORS_V1_0,
    DELIVERY_ERRORS_V1_1,
    RULE_VERSION,
    SUPPORTED_RULE_VERSIONS,
    study_outcome_by_version,
)

OUT_JSON = L2_ROOT / "reports" / "v3_r6_study_outcome_version_audit_20261017.json"
OUT_MD = L2_ROOT / "reports" / "V3_R6_STUDY_OUTCOME_VERSION_AUDIT_20261017.md"

COHORTS = [
    ("qwen36_pi_p1_pilot_20261017/episodes", L1_ROOT / "runs" / "qwen36_pi_p1_pilot_20261017" / "episodes"),
    (
        "qwen36_pi_p1_pilot_20261017/v3_g1_cold_resume",
        L1_ROOT / "runs" / "qwen36_pi_p1_pilot_20261017" / "v3_g1_cold_resume",
    ),
    ("v3_qwen36_pi_dev45_20261017/episodes", L1_ROOT / "runs" / "v3_qwen36_pi_dev45_20261017" / "episodes"),
    (
        "v3_qwen36_pi_dev45_20261017/failed_attempts",
        L1_ROOT / "runs" / "v3_qwen36_pi_dev45_20261017" / "failed_attempts",
    ),
    (
        "v3_qwen36_pi_dev45_20261017/v3_g2_c0_resumes",
        L1_ROOT / "runs" / "v3_qwen36_pi_dev45_20261017" / "v3_g2_c0_resumes",
    ),
    ("deepseek_dev45_20261015/runs", L1_ROOT / "runs" / "deepseek_dev45_20261015" / "runs"),
    ("deepseek_holdout45_20261017/runs", L1_ROOT / "runs" / "deepseek_holdout45_20261017" / "runs"),
    ("swebench_pro_f27_live_20261017", L1_ROOT / "runs" / "swebench_pro_f27_live_20261017"),
    ("lhtb_sokoban_qwen36_cohort", L1_ROOT / "runs" / "lhtb_sokoban_qwen36_cohort"),
]


def _scan_cohort(cohort_id: str, cohort_dir: Path) -> dict[str, Any]:
    if not cohort_dir.exists():
        return {
            "cohort_id": cohort_id,
            "exists": False,
            "total_packages": 0,
            "v1_0": {"valid_count": 0, "valid_zero_count": 0, "invalid_count": 0, "mean_score": None},
            "v1_1": {"valid_count": 0, "valid_zero_count": 0, "invalid_count": 0, "mean_score": None},
            "affected_count": 0,
            "affected_episodes": [],
        }

    pkg_dirs = [
        p for p in sorted(cohort_dir.iterdir())
        if p.is_dir() and (p / "manifest.json").exists()
    ]

    v10_scores: list[float] = []
    v11_scores: list[float] = []
    v10_valid_zero = 0
    v11_valid_zero = 0
    v10_invalid = 0
    v11_invalid = 0
    affected: list[dict[str, Any]] = []

    for pkg in pkg_dirs:
        both = study_outcome_by_version(pkg)
        o10 = both["v1.0"]
        o11 = both["v1.1"]

        if o10.get("study_valid") and o10.get("study_outcome_y") is not None:
            v10_scores.append(float(o10["study_outcome_y"]))
            if o10.get("study_classification") == "valid_zero":
                v10_valid_zero += 1
        else:
            v10_invalid += 1

        if o11.get("study_valid") and o11.get("study_outcome_y") is not None:
            v11_scores.append(float(o11["study_outcome_y"]))
            if o11.get("study_classification") == "valid_zero":
                v11_valid_zero += 1
        else:
            v11_invalid += 1

        if both.get("changed"):
            affected.append({
                "episode_id": pkg.name,
                "path": str(pkg.relative_to(REPO_ROOT)),
                "native_error": o11.get("native_error"),
                "v1_0": o10,
                "v1_1": o11,
            })

    return {
        "cohort_id": cohort_id,
        "exists": True,
        "total_packages": len(pkg_dirs),
        "v1_0": {
            "valid_count": len(v10_scores),
            "valid_zero_count": v10_valid_zero,
            "invalid_count": v10_invalid,
            "mean_score": round(sum(v10_scores) / len(v10_scores), 6) if v10_scores else None,
        },
        "v1_1": {
            "valid_count": len(v11_scores),
            "valid_zero_count": v11_valid_zero,
            "invalid_count": v11_invalid,
            "mean_score": round(sum(v11_scores) / len(v11_scores), 6) if v11_scores else None,
        },
        "affected_count": len(affected),
        "affected_episodes": affected,
    }


def main() -> int:
    cohort_results = [_scan_cohort(cid, cdir) for cid, cdir in COHORTS]
    all_affected = [
        {"cohort_id": c["cohort_id"], **ep}
        for c in cohort_results
        for ep in c.get("affected_episodes", [])
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "default_rule_version": RULE_VERSION,
        "supported_rule_versions": list(SUPPORTED_RULE_VERSIONS),
        "added_in_v1_1": [x for x in DELIVERY_ERRORS_V1_1 if x not in DELIVERY_ERRORS_V1_0],
        "raw_official_grade_files_modified": False,
        "total_affected_packages_across_cohorts": len(all_affected),
        "cohorts": cohort_results,
        "affected_packages": all_affected,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# R6 研究终点规则版本化与新旧敏感性双算报告（2026-10-17）",
        "",
        f"- **生成时间**：`{payload['generated_at']}`",
        f"- **支持规则版本**：`{SUPPORTED_RULE_VERSIONS}`（当前默认 `RULE_VERSION = {RULE_VERSION!r}`）",
        f"- **`v1.1` 新增条目**：`{payload['added_in_v1_1']}`",
        "- **原始官方评分文件是否改动**：`False`（所有 `official_grade.json` 与 `manifest.json` 原件保持只读，不改原始官方分）",
        "",
        "## 1. 规则版本定义对比",
        "",
        "| 规则版本 | 涵盖为 `valid_zero` (`missing_or_damaged_delivery`) 的损坏交付异常 | 触发背景 |",
        "|---|---|---|",
        "| `v1.0` | `FileNotFoundError`, `BadZipFile`, `PackageNotFoundError`, `InvalidFileException`, `XMLSyntaxError`, `ParseError`, `'未找到'`, `'不存在'`, `'无法打开'`, `'不是 .docx'`, `'损坏'`, `'corrupt'`, `'not a zip file'` | 初始归档/空文件损坏交付白名单 |",
        "| `v1.1` | 在 `v1.0` 基础上显式新增：OpenXML 属性缺失异常 `\"AttributeError: 'NoneType' object has no attribute 'val'\"` | `P1` 探针 `officeval_070.r02`（以及早期单条探针 `070.r00`）：产出的 `.docx` 缺少必要 XML `.val` 属性导致评测器抛 `AttributeError`，实为损坏交付而非评测容器故障 |",
        "",
        "## 2. 各批次 `v1.0` vs `v1.1` 并排敏感性汇总表",
        "",
        "| 批次 (`cohort_id`) | 扫描包总数 | `v1.0` 有效数 / 有效零分数 / 无效数 | `v1.0` 有效均分 | `v1.1` 有效数 / 有效零分数 / 无效数 | `v1.1` 有效均分 | 受影响包数 (`v1.0 != v1.1`) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for c in cohort_results:
        v10 = c["v1_0"]
        v11 = c["v1_1"]
        m10 = f"{v10['mean_score']:.4f}" if v10["mean_score"] is not None else "N/A"
        m11 = f"{v11['mean_score']:.4f}" if v11["mean_score"] is not None else "N/A"
        md_lines.append(
            f"| `{c['cohort_id']}` | {c['total_packages']} | "
            f"{v10['valid_count']} / {v10['valid_zero_count']} / {v10['invalid_count']} | `{m10}` | "
            f"{v11['valid_count']} / {v11['valid_zero_count']} / {v11['invalid_count']} | `{m11}` | "
            f"**{c.get('affected_count', 0)}** |"
        )

    md_lines.extend([
        "",
        "## 3. 受影响运行（`v1.0 != v1.1`）逐条明细清单",
        "",
        "| 批次 | 运行/尝试 ID | `official_grade.json` 原始 `error` | `v1.0` 判定 (`classification / valid / score`) | `v1.1` 判定 (`classification / valid / score`) |",
        "|---|---|---|---|---|",
    ])
    if not all_affected:
        md_lines.append("| 无 | - | - | - | - |")
    else:
        for ep in all_affected:
            v10 = ep["v1_0"]
            v11 = ep["v1_1"]
            err_txt = str(ep.get("native_error") or "").replace("\n", " ").replace("|", "\\|")
            if len(err_txt) > 120:
                err_txt = err_txt[:117] + "..."
            md_lines.append(
                f"| `{ep['cohort_id']}` | `{ep['episode_id']}` | `{err_txt}` | "
                f"`{v10['study_classification']} / {v10['study_valid']} / {v10['study_outcome_y']}` | "
                f"`{v11['study_classification']} / {v11['study_valid']} / {v11['study_outcome_y']}` |"
            )

    OUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[R6 AUDIT] Wrote {OUT_JSON} and {OUT_MD} (affected_packages={len(all_affected)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
