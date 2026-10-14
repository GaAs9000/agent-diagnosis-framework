# Integrated delivery freeze

The integrated delivery is a bounded comparison of the fixed OfficeVal task set
002, 003, 006, 012, 015, 018, 033, 037 (two original episodes per task,
maximum 16). It freezes seed 2026, target Qwen3-Coder-30B-A3B, reference
Qwen3-4B, at most three error revisions and 100 execution fragments.

scripts/run_integrated_delivery.py is analysis-only by default. It reads saved
JSON runs, records explicit unknown/no-data states, and writes separate
software delivery, runtime coverage, natural failure, natural success, method
comparison, and scientific claim fields. Missing runs are never inferred or
replaced by another task. Source, task, asset, and verifier hash slots are
included in the frozen manifest.

Real execution is an injected serial hook (serial_execute); tests and the
analysis CLI do not start models, Docker, or verifiers. Existing R1/R1.1/R1.2
artifacts are read-only inputs and are not rewritten.
