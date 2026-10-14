"""Actual native-tool qualification against a random file inside Docker."""
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import secrets
import subprocess

from ..evidence._storage import encode
from ..manifests import canonical_hash
from .agent import NativeAgent
from .docker_runtime import DockerRuntime
from .tokenizer_client import TokenizerClient

ROOT = Path(__file__).resolve().parents[3]
MODEL_REPO = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MODEL_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
TOOL_SCHEMAS = [{"type": "function", "function": {
    "name": "workspace_python", "description": "Execute Python as a standalone script in the isolated task workspace, not as a Notebook. Use print() to return observations; a bare expression produces no output. Installed: python-docx, lxml, PyMuPDF (fitz), pdfplumber, Pillow. Returns actual stdout, stderr and exit code.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"], "additionalProperties": False},
}}]


def image_identity(tag):
    data = json.loads(subprocess.check_output(["docker", "image", "inspect", tag]))[0]
    return data["Id"]


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def model_fingerprint():
    folder = ROOT / "models/Qwen3-4B-Instruct-2507-4bit"
    files = {str(p.relative_to(folder)): file_hash(p) for p in sorted(folder.iterdir()) if p.is_file()}
    descriptor = {
        "repository": MODEL_REPO, "revision": MODEL_REVISION, "files": files,
        "server_packages_lock": file_hash(ROOT / "requirements.model.local-arm64.lock.txt"),
        "inspect_version": importlib.metadata.version("inspect-ai"),
        "prompt_cache_size": 0, "prompt_concurrency": 1, "decode_concurrency": 1,
        "tokenizer": TokenizerClient().text_details("fingerprint")["tokenizer_fingerprint"],
    }
    return {**descriptor, "sha256": canonical_hash(descriptor)}


def delivered_tool_result(_name, _arguments, raw):
    """Bound the model context; complete raw tool output remains in event records."""
    visible = {key: raw.get(key) for key in ("exit_code", "error", "truncated", "stdout_bytes", "stderr_bytes")}
    visible["stdout"] = raw.get("stdout", "")[:6000]
    visible["stderr"] = raw.get("stderr", "")[:2000]
    visible["context_output_truncated"] = len(raw.get("stdout", "")) > 6000 or len(raw.get("stderr", "")) > 2000
    return json.dumps(visible, ensure_ascii=False)


async def qualify_model(root: Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    source = root / "input"
    source.mkdir()
    source.chmod(0o700)
    # This checks native tool/observation plumbing, not long identifier copying.
    # A real tool call and matching result ID remain mandatory.
    nonce = str(100000 + secrets.randbelow(900000))
    (source / "nonce.txt").write_text(nonce)
    fingerprint = model_fingerprint()
    (root / "model_fingerprint.json").write_bytes(encode(fingerprint))
    with DockerRuntime(image_identity("c3-r1-agent:locked"), root / "runtime") as runtime:
        runtime.seed_workspace(source)
        audit = runtime.audit_boundary()
        (root / "audit.json").write_bytes(encode(audit))
        if not audit["qualified"]:
            raise RuntimeError("runtime boundary not qualified")
        agent = NativeAgent({"seed": 2026, "generation_timeout_seconds": 20, "token_count": TokenizerClient().count_messages,
                             "tool_result_transform": delivered_tool_result,
                             "identity_context": {"model_fingerprint": fingerprint["sha256"],
                                "tokenizer_fingerprint": fingerprint["tokenizer"],
                                "provenance": {"stage": "native-tool-qualification", "source": "local-pinned-model"}}}, root / "agent")
        result = await agent.run([
            {"role": "system", "content": "Use native tools when filesystem information is needed. Do not guess file contents."},
            {"role": "user", "content": "Use workspace_python to read nonce.txt and print its contents to stdout. Then reply with exactly those contents."},
        ], {"workspace_python": lambda code: runtime.exec_python(code)}, TOOL_SCHEMAS, max_turns=3)
        (root / "result.json").write_bytes(encode(result))
        messages = result["messages"]
        calls = [c for m in messages if m["role"] == "assistant" for c in (m.get("tool_calls") or [])]
        tools = [m for m in messages if m["role"] == "tool"]
        final = messages[-1].get("content", "")
        final_text = final if isinstance(final, str) else json.dumps(final, ensure_ascii=False)
        match = next((m for m in tools if nonce in str(m.get("content", ""))), None)
        native = next((c for c in calls if match and c["id"] == match["tool_call_id"] and c["function"] == "workspace_python"), None)
        if not native or nonce not in final_text or result["tokens"]["output_tokens"] <= 0:
            raise RuntimeError("model failed actual native-call/nonce qualification")
        receipt = {
            "provider": "openai-api/mlx/default_model", "model": MODEL_REPO,
            "model_fingerprint": fingerprint["sha256"], "native_tool_calls": len(calls),
            "tokenizer_fingerprint": fingerprint["tokenizer"],
            "prompt_tokens": result["tokens"]["full_prompt_tokens"], "completion_tokens": result["tokens"]["output_tokens"],
            "tool_call_id": native["id"], "observation_tool_call_id": match["tool_call_id"],
            "nonce": nonce, "observed_nonce": nonce, "independent_nonce": nonce,
            "is_synthetic": False, "status": "passed", "model_calls": result["modelcall_count"],
            "model_trace_sha256": file_hash(root / "agent/model_calls.jsonl"),
        }
        (root / "qualification.json").write_bytes(encode(receipt))
        return receipt


if __name__ == "__main__":
    import sys
    print(json.dumps(asyncio.run(qualify_model(Path(sys.argv[1]))), ensure_ascii=False, indent=2))
