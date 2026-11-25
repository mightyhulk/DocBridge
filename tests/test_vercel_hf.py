"""Deployment & Hugging Face integration tests (Vercel entrypoint, HF router calls, serverless consult state)."""
import asyncio
import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "backend"))

from app import consult, llm, medllm  # noqa: E402
from app.config import get_settings  # noqa: E402


def test_vercel_entrypoint_imports_and_health_reports_huggingface(monkeypatch):
    spec = importlib.util.spec_from_file_location("vercel_api_index", ROOT / "api" / "index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s = get_settings()
    monkeypatch.setattr(s, "hf_token", "")
    c = TestClient(mod.app)
    r = c.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["llm"]["provider"] == "huggingface"
    assert body["med_llm"]["provider"] == "huggingface"
    assert body["vector_db"]["documents"] > 0
    assert body["patients"] > 0
    assert "pi_mic" not in body


def test_hf_chat_calls_router_and_retries_without_response_format(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "hf_token", "hf_test_token")
    monkeypatch.setattr(s, "hf_base_url", "https://router.huggingface.co/v1")
    monkeypatch.setattr(s, "llm_model", "meta-llama/Llama-3.3-70B-Instruct")

    calls = []

    class FakeResp:
        def __init__(self, code, data):
            self.status_code = code
            self._data = data

        def raise_for_status(self):
            if self.status_code >= 400:
                import httpx
                raise httpx.HTTPStatusError("err", request=None, response=self)

        def json(self):
            return self._data

    class FakeClient:
        async def post(self, url, headers=None, json=None):
            calls.append((url, headers, dict(json)))
            if "response_format" in json:
                return FakeResp(400, {"error": "response_format unsupported"})
            return FakeResp(200, {"choices": [{"message": {"content": '{"answer": "ok", "confidence": 0.9, "needs_reference": false}'}}]})

    monkeypatch.setattr(llm, "_client", lambda *a, **k: FakeClient())
    out = asyncio.run(llm.chat_json([{"role": "user", "content": "hi"}]))
    assert out == {"answer": "ok", "confidence": 0.9, "needs_reference": False}
    assert len(calls) == 2
    assert calls[0][0] == "https://router.huggingface.co/v1/chat/completions"
    assert calls[0][1]["Authorization"] == "Bearer hf_test_token"
    assert "response_format" not in calls[1][2]


def test_medllm_uses_huggingface_and_falls_back_to_llm_model(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "hf_token", "hf_test_token")
    monkeypatch.setattr(s, "med_model", "custom/med-model")
    monkeypatch.setattr(s, "llm_model", "meta-llama/Llama-3.3-70B-Instruct")
    monkeypatch.setattr(s, "med_api_url", "")

    seen = []

    async def fake_hf_chat(base_url, token, model, messages, **kw):
        seen.append(model)
        if model == "custom/med-model":
            raise llm.LLMUnavailable("404 not hosted")
        return '{"answer": "Rest and fluids.", "confidence": 0.9, "needs_doctor": false}'

    monkeypatch.setattr(medllm, "hf_chat", fake_hf_chat)
    out = asyncio.run(medllm.chat_json([{"role": "user", "content": "cough"}]))
    assert out["answer"] == "Rest and fluids."
    assert seen == ["custom/med-model", "meta-llama/Llama-3.3-70B-Instruct"]
    assert medllm.last_endpoint == "huggingface:meta-llama/Llama-3.3-70B-Instruct"


def test_consult_survives_cold_start_via_signed_cookie(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "auth_disabled", True)
    spec = importlib.util.spec_from_file_location("vercel_api_index2", ROOT / "api" / "index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    c = TestClient(mod.app)
    r1 = c.post("/api/consult/start?speak=false")
    assert r1.status_code == 200
    sid = r1.json()["session_id"]

    # Simulate a brand-new serverless container where both memory and SQLite lack this session
    consult._sessions.clear()
    from app import db
    db.conn().execute("DELETE FROM consult_sessions WHERE id=?", (sid,))
    db.conn().commit()

    r2 = c.post(f"/api/consult/{sid}/message", json={"text": "I have a mild runny nose for 1 day, 2/10", "speak": False})
    assert r2.status_code == 200
    assert r2.json()["session_id"] == sid

