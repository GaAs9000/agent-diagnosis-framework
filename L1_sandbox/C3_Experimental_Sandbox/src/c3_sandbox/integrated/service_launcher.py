"""Serial local MLX service launcher for the integrated delivery.

It uses absolute argv entries, no shell interpolation, and keeps model and
tokenizer processes bound to one configured model at a time.  The caller owns
the host and must stop the returned manager when the run ends.
"""
from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import time
from urllib import request as urlrequest


class ServiceLaunchError(RuntimeError):
    pass


class SerialServiceManager:
    def __init__(self, *, root: str | Path, config: dict, output_dir: str | Path):
        self.root = Path(root).absolute()
        self.config = config
        self.server = config["identities"]["server"]
        self.output_dir = Path(output_dir).absolute() / "services"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_process = None
        self.tokenizer_process = None
        self.model_key = None
        self.model_pid = None

    def _model_path(self, model_key: str) -> Path:
        value = self.config["models"][model_key]["model_path"]
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        if not path.is_dir():
            raise ServiceLaunchError(f"model path unavailable: {path}")
        return path

    def _stop_process(self, process):
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)

    def stop(self):
        self._stop_process(self.tokenizer_process)
        self._stop_process(self.model_process)
        self.tokenizer_process = self.model_process = None
        self.model_key = self.model_pid = None

    def _wait_port(self, port: int, timeout: float = 30.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((self.server["host"], port), timeout=1):
                    return
            except OSError:
                time.sleep(.25)
        raise ServiceLaunchError(f"service port did not open: {port}")

    def _wait_health(self, timeout: float = 30.0):
        endpoint = self.config["services"]["model_endpoint"].rstrip("/") + "/health"
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            try:
                with urlrequest.urlopen(endpoint, timeout=2) as response:
                    if response.status == 200:
                        return
            except Exception as exc:
                last = exc
            time.sleep(.5)
        raise ServiceLaunchError(f"model health check failed: {last}")

    def ensure(self, model_key: str) -> int:
        if self.model_key == model_key and self.model_process and self.model_process.poll() is None:
            return int(self.model_process.pid)
        self.stop()
        model = self.config["models"][model_key]
        python = Path(self.server["python"])
        if not python.is_absolute() or not python.is_file():
            raise ServiceLaunchError("approved model Python must be an absolute existing file")
        model_path = self._model_path(model_key)
        host, port = self.server["host"], int(self.server["model_port"])
        max_tokens = int(model.get("max_tokens", 4096))
        model_log = (self.output_dir / f"{model_key}-model.log").open("w")
        tokenizer_log = (self.output_dir / f"{model_key}-tokenizer.log").open("w")
        model_argv = [str(python), "-m", "mlx_lm.server", "--model", str(model_path),
                      "--host", host, "--port", str(port), "--prompt-cache-size", "0",
                      "--decode-concurrency", "1", "--prompt-concurrency", "1",
                      "--max-tokens", str(max_tokens), "--temp", "0", "--log-level", "INFO"]
        tokenizer_script = self.root / "src/c3_sandbox/r1/tokenizer_server.py"
        tokenizer_argv = [str(python), str(tokenizer_script), "--model-path", str(model_path)]
        if model.get("revision"):
            tokenizer_argv += ["--revision", str(model["revision"])]
        if model.get("tool_parser") == "mlx_lm.tool_parsers.qwen3_coder.parse_tool_call":
            tokenizer_argv += ["--expected-parser", "qwen3_coder"]
        try:
            self.model_process = subprocess.Popen(model_argv, cwd=str(self.root), stdout=model_log,
                                                   stderr=subprocess.STDOUT)
            self.tokenizer_process = subprocess.Popen(tokenizer_argv, cwd=str(self.root), stdout=tokenizer_log,
                                                       stderr=subprocess.STDOUT)
            self._wait_port(port)
            self._wait_port(int(self.server["tokenizer_port"]))
            self._wait_health()
        except Exception:
            self.stop(); raise
        self.model_key = model_key
        self.model_pid = int(self.model_process.pid)
        (self.output_dir / f"{model_key}-argv.json").write_text(json.dumps({"model": model_argv, "tokenizer": tokenizer_argv}, indent=2) + "\n")
        return self.model_pid

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
