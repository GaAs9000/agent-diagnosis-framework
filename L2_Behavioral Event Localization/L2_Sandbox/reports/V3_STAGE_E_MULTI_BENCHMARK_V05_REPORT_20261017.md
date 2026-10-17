# V3 Stage E: Multi-Benchmark Minimal Deliverable Report (`L2 v0.5`)

- **Generated At**: `2026-10-17T11:41:45Z`
- **Unified Actor**: `Qwen/Qwen3.6-27B (revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9)`
- **Protocol**: `L2 v0.5` (`DiagnosticUnitV05`, `E1Observation` + `E2ResponseVector`, Notion V3 §3.6 & §4.1)

---

## 1. Cross-Benchmark Category Mapping & Minimal Deliverable Summary (V3 §3.6 & §4.1)

| Benchmark | Native Harness | `V3-G4` Snapshot-Terminal Consistency | `broken` (E1) | `undelivered` (E1) | `never_succeeded_on_disk` (E1) | Benchmark-Specific Boundary Notes |
|---|---|---:|---:|---:|---:|---|
| **SWE-bench Pro V2 (Post-`F27`)** | `native-tool-agent` + writable `/workspace/.swe_app_upper` bind-mounted at `/app` | `4/4 (100% across 3 pre-F27 + 1 post-F27 episodes)` | **`3`** (`pass_to_pass` regressions at step 11) | **`4`** (`4/4` `fail_to_pass` tests passed in `/app` at `cp000011..12` but unsubmitted to `submission/patch.diff`) | `0` | Fixing `F27` (writable `/app` persisted across `docker-cold` steps) turned candidate extraction from `0/63` checkpoints (pre-`F27`) to `2/13` active patch checkpoints, turning **`4/4` (`100%`) `fail_to_pass` tests green** while localizing `3` `pass_to_pass` regressions (`broken`). |
| **LHTB Sokoban** | `native-tool-agent` + stateful `/workspace/.lhtb_state` | `3/3 (100%)` (`63` checkpoints) | **`2`** (`irreversible_state_damage` via zero-cost box-corner deadlock detection) | `N/A` (`not_applicable`) | `1` | Sokoban has no separate delivery channel (`undelivered = not_applicable`) and no hidden rubric criteria (`R_rule = not_applicable`); zero-cost deadlock detection identifies irreversible state damage (`broken`) at turn 9 (`run_seed42`). |

---

## 2. SWE-bench Pro V2 (`F27` Writable `/app` Verification & `L2 v0.5` Decomposition)

- **Episode**: `task2_s42_f27` (`swebench_pro:instance_ansible__ansible-0fd88717c953...`)
- **`c3-l1 verify`**: `valid`
- **Before vs. After `F27` Fix**:
  - **Pre-`F27`**: `0/63 (read-only /app blocked in-place edits)`
  - **Post-`F27`**: `2/13 (cp000011 & cp000012 captured 2,038B git diff in /workspace/.swe_app_upper)`
- **Checkpoint `cp000011` & `cp000012` Official Pytest Snapshot Evaluation (`27/30` = `90.0%` tests passed)**:
  - **`fail_to_pass` (`4/4` = `100%` PASSED in `/app`, `undelivered` at terminal)**:
    - `test/units/plugins/lookup/test_password.py::TestParseContent::test` (`deduction = 0.0` at `cp000011..12`, `operational_label = "undelivered"`)
    - `test/units/plugins/lookup/test_password.py::TestParseContent::test_empty_password_file` (`deduction = 0.0` at `cp000011..12`, `operational_label = "undelivered"`)
    - `test/units/plugins/lookup/test_password.py::TestParseContent::test_with_salt` (`deduction = 0.0` at `cp000011..12`, `operational_label = "undelivered"`)
    - `test/units/plugins/lookup/test_password.py::TestParseContent::test_with_salt_and_ident` (`deduction = 0.0` at `cp000011..12`, `operational_label = "undelivered"`)
  - **`pass_to_pass` (`23/26` PASSED, `3` Regressions Localized as `broken` at Step 11)**:
    - `test/units/plugins/lookup/test_password.py::TestLookupModuleWithPasslib::test_encrypt` (`operational_label = "broken"`, `source_turn = 11`)
    - `test/units/plugins/lookup/test_password.py::TestLookupModuleWithoutPasslib::test_no_encrypt` (`operational_label = "broken"`, `source_turn = 11`)
    - `test/units/plugins/lookup/test_password.py::TestLookupModuleWithoutPasslib::test_only_a` (`operational_label = "broken"`, `source_turn = 11`)

---

## 3. LHTB Sokoban (`L2 v0.5` Deadlock & Boundary Decomposition)

| Episode | `k_star` | `first_deadlock_turn` | `e1_class` | `candidate_substate` | `operational_label` | `evidence_status` |
|---|---:|---:|---|---|---|---|
| `run_seed42` | `0` | `15` | `broken` | `irreversible_state_damage` | `broken` | `adjudicated` |
| `run_seed101` | `0` | `None` | `never_succeeded_on_disk` | `no_candidate` | `None` | `not_applicable` |
| `run_seed2026` | `0` | `4` | `broken` | `irreversible_state_damage` | `broken` | `adjudicated` |
