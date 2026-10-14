"""Offline D6--D8 integrated delivery helpers.

This package consumes already saved evidence dictionaries.  It never runs a
model, tool, sandbox, or verifier and does not mutate source evidence.
"""
from .delivery import (
    EvidenceIndex, build_evidence_index, extract_object_changes, DeterministicPredictor,
    select_probe, seal_prediction, generate_report,
    OWN_TEXT, ALL_TEXT, C3_STATE,
)

__all__ = [
    "EvidenceIndex", "build_evidence_index", "extract_object_changes", "DeterministicPredictor",
    "select_probe", "seal_prediction", "generate_report",
    "OWN_TEXT", "ALL_TEXT", "C3_STATE",
]
from .plan import (TASK_IDS, TARGET_MODEL, REFERENCE_MODEL, SEED, FrozenPlan,
                   build_plan, serial_execute, analyze_saved_runs, sha256_file)
from .office_runner import build_task_bundle, run_episode, run_episode_sync
from .d7_backend import DiagnosticBackend, DiagnosticPrediction, FixturePredictor, build_evidence_view, select_condition
__all__ += ["DiagnosticBackend", "DiagnosticPrediction", "FixturePredictor", "build_evidence_view", "select_condition"]
from .d7_backend import OpenAICompatibleBackend
__all__ += ["OpenAICompatibleBackend"]
# Closeout graph/metrics API (additive to the original D6-D8 helpers).
from .delivery import (
    GraphNode, GraphRelation, RelationGraph, build_relation_graph,
    build_typed_relation_graph, build_evidence_graph, compute_d8_metrics,
    compute_branch_metrics, calculate_d8_metrics,
)
__all__ += [
    "GraphNode", "GraphRelation", "RelationGraph", "build_relation_graph",
    "build_typed_relation_graph", "build_evidence_graph", "compute_d8_metrics",
    "compute_branch_metrics", "calculate_d8_metrics",
]

from .experiment_card import *
from .gemini_api import GeminiNativeToolAgent, GeminiRecord, GeminiToolCall, MODEL as GEMINI_MODEL
from .gemini_episode import run_gemini_episode, run_gemini_episode_sync
__all__ += ["run_gemini_episode", "run_gemini_episode_sync"]

from ..gateway import (
    BackendError,
    CostRecord,
    ModelBackend,
    ModelIdentity,
    ModelResponse,
    ModelToolCall,
    UnifiedMessage,
    ApiBackend,
    LocalBackend,
    SavedReferenceBackend,
    BackendRegistry,
    ModelGateway,
)
__all__ += [
    "BackendError",
    "CostRecord",
    "ModelBackend",
    "ModelIdentity",
    "ModelResponse",
    "ModelToolCall",
    "UnifiedMessage",
    "ApiBackend",
    "LocalBackend",
    "SavedReferenceBackend",
    "BackendRegistry",
    "ModelGateway",
]
