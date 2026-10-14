"""Controller-side exact Inspect conversion and loopback tokenizer callbacks."""
from __future__ import annotations

import asyncio
import http.client
import json

ENDPOINT = "http://127.0.0.1:18081"
MAX_BODY_BYTES = 4 * 1024 * 1024


class TokenizerClient:
    def __init__(self, endpoint: str = ENDPOINT, *, timeout: float = 30):
        if endpoint != ENDPOINT:
            raise ValueError("Tokenizer endpoint must be the private R1 loopback endpoint")
        self.timeout = timeout
        self._provider = None

    def _post(self, route: str, body: dict) -> dict:
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(data) > MAX_BODY_BYTES:
            raise ValueError("Tokenizer request exceeds body limit; no truncation is allowed")
        # HTTPConnection bypasses ambient HTTP proxies and never follows redirects.
        connection = http.client.HTTPConnection("127.0.0.1", 18081, timeout=self.timeout)
        try:
            connection.request("POST", route, body=data, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            raw = response.read(65537)
            if response.status != 200 or len(raw) > 65536:
                raise RuntimeError("Local tokenizer request failed")
            result = json.loads(raw)
            if (type(result) is not dict or type(result.get("token_count")) is not int
                    or result["token_count"] < 0
                    or not isinstance(result.get("tokenizer_fingerprint"), dict)
                    or not isinstance(result.get("prompt_token_id_sha256"), str)):
                raise RuntimeError("Invalid local tokenizer response")
            return result
        finally:
            connection.close()

    def text_details(self, text: str) -> dict:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self._post("/text", {"text": text})

    def text_count(self, text: str) -> int:
        return self.text_details(text)["token_count"]

    async def count_messages_details(self, messages: list[dict], tools: list[dict]) -> dict:
        from inspect_ai.model import ChatMessage, get_model
        from inspect_ai.tool import ToolInfo
        from pydantic import TypeAdapter

        # Reuse the same pinned compatible-provider methods as NativeAgent.
        # Construction performs no generation or HTTP request.
        if self._provider is None:
            self._provider = get_model(
                "openai-api/mlx/default_model", base_url="http://127.0.0.1:18080/v1",
                api_key="local-unused", emulate_tools=False, responses_api=False,
                stream=False, strict_tools=False, memoize=False).api
        provider = self._provider
        native = TypeAdapter(list[ChatMessage]).validate_python(messages)
        infos = []
        for schema in tools:
            definition = schema.get("function", schema)
            infos.append(ToolInfo(name=definition["name"], description=definition.get("description", ""),
                                  parameters=definition.get("parameters", {"type": "object", "properties": {}})))
        wire_messages = await provider.messages_to_openai(native)
        wire_tools = provider.tools_to_openai(infos) if infos else None
        return await asyncio.to_thread(self._post, "/count", {"messages": wire_messages, "tools": wire_tools})

    async def count_messages(self, messages: list[dict], tools: list[dict]) -> int:
        return (await self.count_messages_details(messages, tools))["token_count"]
