"""BackendRegistry: Config-driven resolution of ModelBackend instances.

Decouples application code (Sandbox, Diagnostics, Experiment Cards) from
concrete model providers and execution engines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Type

from .backend import BackendError, ModelBackend, ModelIdentity
from .api_backend import ApiBackend
from .local_backend import LocalBackend, SavedReferenceBackend


class BackendRegistry:
    """Registry and factory for constructing ModelBackend instances from config."""

    _BACKENDS: dict[str, Type[ModelBackend]] = {
        "api": ApiBackend,
        "local": LocalBackend,
        "saved_reference": SavedReferenceBackend,
    }

    @classmethod
    def register(cls, backend_type: str, backend_cls: Type[ModelBackend]) -> None:
        cls._BACKENDS[backend_type.lower()] = backend_cls

    @classmethod
    def create(
        cls,
        model_config: Mapping[str, Any],
        *,
        root: str | Path | None = None,
        transport: Callable[..., Any] | None = None,
    ) -> ModelBackend:
        """Create a ModelBackend from a dictionary configuration.
        
        Supports both modern (`backend: "api" | "local"`) and legacy formats
        (`provider: "gemini" | "mlx" | "saved_reference"`).
        """
        cfg = dict(model_config)
        root_path = Path(root).resolve() if root else Path.cwd()

        # Determine backend_type
        backend_type = cfg.get("backend")
        provider = (cfg.get("provider") or cfg.get("engine") or "").lower()

        if not backend_type:
            if provider in {"gemini", "google", "openai", "anthropic", "openai-compatible"}:
                backend_type = "api"
            elif provider in {"mlx", "vllm", "ollama", "local"}:
                backend_type = "local"
            elif provider == "saved_reference":
                backend_type = "saved_reference"
            else:
                backend_type = "api"  # default fallback

        backend_type = str(backend_type).lower()
        if backend_type not in cls._BACKENDS:
            raise BackendError(f"Unknown model backend type: {backend_type}. Registered: {list(cls._BACKENDS.keys())}")

        # Model name & identity
        model_name = cfg.get("model") or cfg.get("name") or cfg.get("repository") or "unknown"
        fingerprint = cfg.get("fingerprint") or cfg.get("model_fingerprint") or model_name
        tokenizer_fingerprint = cfg.get("tokenizer_fingerprint")
        revision = cfg.get("revision")
        endpoint = cfg.get("endpoint") or cfg.get("model_endpoint")

        # Resolve paths in metadata if provided
        metadata = dict(cfg)
        if "model_path" in metadata:
            mp = Path(metadata["model_path"])
            if not mp.is_absolute() and root_path:
                metadata["model_path"] = str(root_path / mp)

        identity = ModelIdentity(
            backend=backend_type,
            provider=provider or ("gemini" if backend_type == "api" else "mlx"),
            model_name=str(model_name),
            fingerprint=str(fingerprint),
            tokenizer_fingerprint=str(tokenizer_fingerprint) if tokenizer_fingerprint else None,
            revision=str(revision) if revision else None,
            endpoint=str(endpoint) if endpoint else None,
            metadata=metadata,
        )

        backend_cls = cls._BACKENDS[backend_type]
        return backend_cls(identity, config=cfg, transport=transport)
