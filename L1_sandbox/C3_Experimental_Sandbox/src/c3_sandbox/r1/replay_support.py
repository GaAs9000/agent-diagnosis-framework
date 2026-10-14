"""Small replay helpers shared by checkpoint and remote-model paths.

This module intentionally has no dependency on ``NativeAgent`` or Inspect.
The replay recorder and the independent Gemini runner can therefore be loaded
in environments that only provide Docker and the pinned grader.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ..evidence._storage import encode


SYSTEM_PROMPT = (
    "Complete the user's document task using the native workspace_python tool. "
    "Use the installed document libraries to inspect and edit the supplied files. "
    "Save the deliverable with its designated original filename. Preserve existing "
    "content and formatting unless the task requires a change. Tool observations "
    "are actual execution results. Explain briefly when finished."
)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path: Path, value) -> None:
    with Path(path).open("xb") as stream:
        stream.write(encode(value))


def _qualification(receipt: dict) -> dict:
    """Validate a previously saved real native qualification receipt."""
    if (receipt.get("status") != "passed" or receipt.get("is_synthetic") is not False
            or receipt.get("provider") != "openai-api/mlx/default_model"
            or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("model_fingerprint", "")))):
        raise ValueError("matching passed real native model qualification required")
    for key in ("native_tool_calls", "prompt_tokens", "completion_tokens", "model_calls"):
        if type(receipt.get(key)) is not int or receipt[key] <= 0:
            raise ValueError("positive native qualification usage required")
    if (not receipt.get("tool_call_id") or receipt["tool_call_id"] != receipt.get("observation_tool_call_id")
            or not receipt.get("nonce") or not receipt["nonce"] == receipt.get("observed_nonce") == receipt.get("independent_nonce")):
        raise ValueError("native qualification observation mismatch")
    return dict(receipt)
