# Agent Behavioral Diagnosis & Capability Boundary Discovery Framework (L1–L3)

> **Core Thesis**: **L1** executes agents in isolated sandboxes on real benchmarks to produce cryptographically sealed execution evidence; **L2** localizes critical behavioral events and predicts bounded interventions from that evidence; **L3** aggregates cross-task findings and validates conditional capability boundaries.

---

## Repository Architecture

```text
.
├── RESEARCH_FRAMEWORK.md                        # Full L1-L3 architecture & research protocol
├── L1_sandbox/
│   └── C3_Experimental_Sandbox/                 # L1: Agent Execution Sandbox & Sealed Evidence Producer (c3-l1)
│       ├── src/c3_sandbox/l1/                   # Core orchestrator, contracts, docker runtime, adapters & evaluators
│       ├── configs/l1/                          # Benchmark & model configurations (OfficeVal, LHTB, SWE-bench Pro V2)
│       ├── benchmarks/                          # SHA-256 data lock manifests (data.lock.json)
│       ├── docker/                              # Isolated container Dockerfiles & grader workers
│       ├── schemas/                             # Trajectory & evidence JSON schemas
│       ├── scripts/l1/                          # Local acceptance, stateful resume & portable archiving tools
│       └── tests/                               # 204 unit & live integration tests (tests/l1, tests/l1_live)
├── L2_Behavioral Event Localization/
│   └── L2_Sandbox/                              # L2: Evidence-grounded Behavioral Event Localization (c3-l2)
│       ├── src/l2_sandbox/                      # L1 reader, fact builder, analyst agent, intervention planner & controller
│       ├── configs/                             # Analyst model & rule configurations
│       ├── docs/                                # L2 v0.1 design contract, protocol & security specs
│       ├── scripts/                             # Schema builder & verification utilities
│       └── tests/                               # 96 unit & integration tests
└── L3_Agent Capability Boundary Discovery/
    └── README.md                                # L3: Cross-task Capability Boundary Discovery & Validation layer
```

---

## Three-Layer Division of Labor

| Layer | Core Question | Responsibility | Primary Deliverable |
| :--- | :--- | :--- | :--- |
| **L1: Execution Sandbox** (`c3-l1`) | *What did the agent actually do?* | Ingests mature benchmarks (`OfficeVal`, `LHTB`, `SWE-bench Pro V2`), runs native tool-calling agents in isolated containers, supports stateful checkpoint cold-resume, and runs official verifiers. | **Sealed Execution Evidence Package** |
| **L2: Behavioral Event Localization** (`c3-l2`) | *Which behavioral events caused the outcome?* | Verifies L1 evidence hashes, extracts objective `FactRecord`s, localizes anomalous steps (`EventLocalization`), formulates falsifiable `Hypothesis` records, and bridges bounded checkpoint interventions (`ExperimentReceipt`). | **Verifiable Diagnosis Package** |
| **L3: Capability Boundary Discovery** | *Under what conditions does the agent succeed or fail?* | Synthesizes cross-task & cross-model patterns, proposes conditional boundary hypotheses, and validates them via out-of-task and counterfactual experiments. | **Capability Boundary Report** |

---

## Quickstart (Framework Only, Zero Datasets Bundled)

This repository tracks **only the L1–L3 agent framework source code, schemas, configurations, and test suites** (all heavy benchmark datasets, third-party repositories, model weights, and run deliverables are excluded via `.gitignore`).

### 1. Install L1 & L2 in a Shared Virtual Environment

```bash
cd L1_sandbox/C3_Experimental_Sandbox
python3 -m venv .venv
source .venv/bin/activate

pip install -e .
pip install -e "../../L2_Behavioral Event Localization/L2_Sandbox"
```

### 2. Run Framework Verification & Unit Tests

```bash
# L1 core unit tests
pytest tests/l1

# L2 behavioral localization unit tests
pytest "../../L2_Behavioral Event Localization/L2_Sandbox/tests"
```

For full architectural details and design red lines, see [RESEARCH_FRAMEWORK.md](./RESEARCH_FRAMEWORK.md).
