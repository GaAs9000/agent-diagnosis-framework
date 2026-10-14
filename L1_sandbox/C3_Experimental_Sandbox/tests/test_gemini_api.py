import json
import pytest
from c3_sandbox.integrated.gemini_api import GeminiNativeToolAgent, MODEL

def raw(parts, usage=None):
    x={"candidates":[{"content":{"parts":parts}}]}
    if usage: x["usageMetadata"]=usage
    return json.dumps(x).encode()

def test_native_tool_call_and_observation_continuation(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "do-not-record-secret")
    replies=iter([raw([{"functionCall":{"name":"lookup","args":{"q":"x"}}}]), raw([{"text":"final"}], {"totalTokenCount": 7})])
    requests=[]
    def transport(body): requests.append(body); return next(replies)
    agent=GeminiNativeToolAgent(transport=transport)
    out=agent.run_sync(user="find", tool_schema=[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}], tool_callback=lambda n,a:{"ok":a["q"]}, evidence_ids=("ev1",))
    assert out["text"] == "final" and len(requests) == 2
    assert json.loads(requests[0])["tools"][0]["functionDeclarations"][0]["name"] == "lookup"
    assert json.loads(requests[1])["contents"][-1]["parts"][0]["functionResponse"]["response"] == {"ok":"x"}
    assert "do-not-record-secret" not in repr(agent.records)
    assert agent.records[-1].usage == {"totalTokenCount": 7}

def test_malformed_response_fails_closed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    agent=GeminiNativeToolAgent(transport=lambda body: b"not-json")
    out=agent.run_sync(user="x")
    assert out["selector_invalid"] is True and agent.records[0].selector_invalid

def test_unknown_function_call_fails_closed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    agent=GeminiNativeToolAgent(transport=lambda body: raw([{"functionCall":{"name":"bad","args":{}}}]))
    out=agent.run_sync(user="x", tool_schema=[{"function":{"name":"good"}}], tool_callback=lambda *_: {})
    assert out["selector_invalid"] is True

def test_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False); monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    agent=GeminiNativeToolAgent(transport=lambda body: b"{}")
    with pytest.raises(RuntimeError, match="API key"):
        agent.run_sync(user="x")

def test_fixed_model_and_endpoint(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    with pytest.raises(ValueError): GeminiNativeToolAgent(model="other", transport=lambda body: b"{}")
    assert MODEL == "gemini-3.5-flash-lite"
