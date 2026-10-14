"""C3 Unified Model Gateway Package."""
from .backend import (
    BackendError,
    CostRecord,
    ModelBackend,
    ModelIdentity,
    ModelResponse,
    ModelToolCall,
    UnifiedMessage,
)
from .api_backend import ApiBackend
from .local_backend import LocalBackend, SavedReferenceBackend
from .registry import BackendRegistry
from .gateway import ModelGateway
from .diagnostician import GatewayDiagnosticBackend

__all__ = [
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
    "GatewayDiagnosticBackend",
]
