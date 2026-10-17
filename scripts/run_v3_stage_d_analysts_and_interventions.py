#!/usr/bin/env python3
"""DEPRECATED / PAUSED: Monolithic V3 Stage D script replaced by R15–R18 split entrypoints.

Per V3 Revision Task Order (db13afd1-9d9a-4e09-bc97-9475157a32d2, R15–R18):
- Stage D is strictly split into three independent plan-first entrypoints:
  1. `scripts/v3_d1_ladder.py`   (R15, R18: D1 3D stratified ladder + single-item R + R12 passing harm check; default --plan-only)
  2. `scripts/v3_d2_ceiling.py`  (R16, R18: D2 2-fold cross-fitted diagnostic value ceiling V3-P3 at 0 new Actor requests)
  3. `scripts/v3_d3_analysts.py` (R17, R18: D3 real LLM Analyst B/C evaluation, gated on D2 V3-P3 passing; default --plan-only)
"""
from __future__ import annotations

import sys


def main() -> int:
    sys.stderr.write(
        "ERROR [R15-R18]: Monolithic `scripts/run_v3_stage_d_analysts_and_interventions.py` is PAUSED and DEPRECATED.\n"
        "Use the split V3 Stage D plan-first entrypoints instead:\n"
        "  - D1 (Intervention Ladder & Harm Check): python scripts/v3_d1_ladder.py --plan-only\n"
        "  - D2 (0-Request Cross-Fitted Ceiling):   python scripts/v3_d2_ceiling.py\n"
        "  - D3 (Analyst M/A/B/C Evaluation):       python scripts/v3_d3_analysts.py --plan-only\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
