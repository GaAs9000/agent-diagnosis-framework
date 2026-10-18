"""Pinned Pi validation via a persistent, execution-disabled native agent loop.

The child consumes recorded tool calls, never invokes a model, and returns native
validation errors verbatim. Missing dependencies and bridge failures are fatal
infrastructure errors rather than recoverable tool feedback.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from typing import Any


class NativePiInfrastructureError(RuntimeError):
    """The native validator cannot safely provide a result."""


class NativePiValidator:
    """Requests and responses are bounded to at most 1 MiB each."""
    def __init__(
        self,
        node_executable: str | None = None,
        bridge_dir: str | Path | None = None,
        timeout_seconds: float = 15,
        max_message_bytes: int = 1024 * 1024,
    ) -> None:
        if timeout_seconds <= 0 or not 256 <= max_message_bytes <= 1024 * 1024:
            raise ValueError("Invalid native bridge limits")
        self.node_executable = node_executable or os.environ.get("C3_PI_NODE") or shutil.which("node")
        selected_dir = bridge_dir or os.environ.get("C3_PI_NATIVE_DIR")
        self.bridge_dir = Path(selected_dir) if selected_dir is not None else Path(__file__).with_name("pi_native")
        self.timeout_seconds = timeout_seconds
        self.max_message_bytes = max_message_bytes
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.RLock()
        self._responses: queue.Queue[Any] = queue.Queue(maxsize=2)
        self._provenance: dict[str, Any] | None = None
        self._closed = False

    def __enter__(self) -> NativePiValidator:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _deadline(self, deadline_monotonic: float | None) -> float:
        local_limit = time.monotonic() + self.timeout_seconds
        return min(local_limit, deadline_monotonic) if deadline_monotonic is not None else local_limit

    @contextmanager
    def _locked_until(self, deadline: float):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._lock.acquire(timeout=remaining):
            raise NativePiInfrastructureError("Native Pi bridge deadline expired waiting for lock")
        try:
            yield
        finally:
            self._lock.release()

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self._fail("Native Pi bridge deadline expired")
        return remaining

    def start(self, deadline_monotonic: float | None = None) -> None:
        deadline = self._deadline(deadline_monotonic)
        with self._locked_until(deadline):
            if self._closed:
                raise NativePiInfrastructureError("Native Pi validator is closed")
            if self._process is not None:
                return
            if not self.node_executable:
                raise NativePiInfrastructureError("Node.js is required for native Pi validation")
            for filename in ("bridge.mjs", "package.json", "package-lock.json", "provenance.json"):
                if not (self.bridge_dir / filename).is_file():
                    raise NativePiInfrastructureError(f"Missing native Pi asset: {filename}")
            self._remaining(deadline)
            try:
                self._process = subprocess.Popen(
                    [self.node_executable, str(self.bridge_dir / "bridge.mjs")],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    cwd=self.bridge_dir,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "NODE_OPTIONS": ""},
                )
            except OSError as error:
                raise NativePiInfrastructureError("Cannot start native Pi bridge") from error
            threading.Thread(target=self._read_responses, daemon=True).start()
            ready = self._receive(deadline)
            if ready.get("ready") is not True or not isinstance(ready.get("provenance"), dict):
                self._fail("Native Pi bridge did not provide its dependency identity")
            self._provenance = ready["provenance"]

    def _read_responses(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            while True:
                line = self._process.stdout.readline(self.max_message_bytes + 1)
                if not line or len(line) > self.max_message_bytes or not line.endswith(b"\n"):
                    self._responses.put(NativePiInfrastructureError("Native Pi bridge closed or exceeded response limit"))
                    return
                self._responses.put(json.loads(line))
        except (OSError, ValueError) as error:
            self._responses.put(NativePiInfrastructureError(f"Invalid native Pi bridge response: {type(error).__name__}"))

    def _receive(self, deadline: float) -> dict[str, Any]:
        try:
            result = self._responses.get(timeout=self._remaining(deadline))
        except queue.Empty:
            self._fail("Native Pi bridge response timeout")
        if isinstance(result, Exception):
            self.close()
            raise result
        if not isinstance(result, dict) or "infrastructure_error" in result:
            self._fail("Native Pi bridge infrastructure failure")
        self._remaining(deadline)
        return result

    def _fail(self, message: str) -> None:
        self.close()
        raise NativePiInfrastructureError(message)

    def _request(self, request: dict[str, Any], deadline_monotonic: float | None = None) -> dict[str, Any]:
        deadline = self._deadline(deadline_monotonic)
        with self._locked_until(deadline):
            self.start(deadline)
            try:
                payload = (json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            except (TypeError, ValueError) as error:
                raise NativePiInfrastructureError("Tool arguments must be finite JSON values") from error
            if len(payload) > self.max_message_bytes:
                raise NativePiInfrastructureError("Native Pi request exceeds byte limit")
            self._remaining(deadline)
            assert self._process is not None and self._process.stdin is not None
            # A stopped child can fill a pipe even before the response timeout.
            written: queue.Queue[Any] = queue.Queue(maxsize=1)

            def write() -> None:
                try:
                    self._process.stdin.write(payload)
                    self._process.stdin.flush()
                    written.put(None)
                except (OSError, ValueError) as error:
                    written.put(error)

            threading.Thread(target=write, daemon=True).start()
            try:
                error = written.get(timeout=self._remaining(deadline))
            except queue.Empty:
                self._fail("Native Pi bridge write timeout")
            if error is not None:
                self._fail("Native Pi bridge write failed")
            result = self._receive(deadline)
            if result.get("provenance") != self._provenance:
                self._fail("Native Pi dependency identity changed")
            return result

    def validate(self, name: str, arguments: Any, call_id: str = "validation", *, deadline_monotonic: float | None = None) -> dict[str, Any]:
        if not isinstance(name, str) or not isinstance(call_id, str):
            raise NativePiInfrastructureError("Tool name and call ID must be strings")
        result = self._request({"op": "validate", "name": name, "arguments": arguments, "call_id": call_id}, deadline_monotonic)
        if not isinstance(result.get("ok"), bool):
            self._fail("Native Pi bridge omitted validation outcome")
        return result

    def validate_batch(self, calls: list[dict[str, Any]], *, deadline_monotonic: float | None = None) -> list[dict[str, Any]]:
        """Validate an ordered recorded batch in one native assistant turn."""
        result = self._request({"op": "validate_batch", "calls": calls}, deadline_monotonic)
        if not isinstance(result.get("results"), list):
            self._fail("Native Pi bridge omitted batch outcomes")
        return result["results"]

    def native_tool_schemas(self) -> list[dict[str, Any]]:
        """Export schemas for inspection only; validation retains native symbols."""
        return self._request({"op": "schemas"})["schemas"]

    def dependency_identity(self) -> dict[str, Any]:
        with self._lock:
            self.start()
            assert self._provenance is not None
            return json.loads(json.dumps(self._provenance))

    def close(self) -> None:
        with self._lock:
            self._closed = True
            process = self._process
            if process is None:
                return
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            for pipe in (process.stdin, process.stdout):
                if pipe is not None:
                    pipe.close()


def validate_pi_call(name: str, arguments: Any, call_id: str = "validation", *, deadline_monotonic: float | None = None, **options: Any) -> dict[str, Any]:
    validator = NativePiValidator(**options)
    try:
        return validator.validate(name, arguments, call_id, deadline_monotonic=deadline_monotonic)
    finally:
        validator.close()
