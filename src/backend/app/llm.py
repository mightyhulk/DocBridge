"""Async client for Hugging Face Inference Router (OpenAI-compatible chat completions)."""
import asyncio
import json
import re

import httpx

from .config import get_settings

_clients: dict[tuple[str, float, int], httpx.AsyncClient] = {}


class LLMUnavailable(RuntimeError):
    pass


def _client(base_url: str, timeout_s: float) -> httpx.AsyncClient:
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = 0
    key = (base_url.rstrip("/"), timeout_s, loop_id)
    client = _clients.get(key)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=timeout_s)
        _clients[key] = client
    return client


def _completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


async def is_available() -> bool:
    """True when a Hugging Face API token is configured."""
    return bool(get_settings().hf_token.strip())


async def hf_chat(
    base_url: str,
    token: str,
    model: str,
    messages: list[dict],
    *,
    json_mode: bool = False,
    max_tokens: int = 350,
    temperature: float = 0.1,
    timeout_s: float = 45.0,
) -> str:
    """Call Hugging Face OpenAI-compatible chat completions, retrying without response_format if unsupported."""
    if not token.strip():
        raise LLMUnavailable("HF_TOKEN is not configured")
    url = _completions_url(base_url)
    headers = {"Authorization": f"Bearer {token.strip()}", "Content-Type": "application/json"}
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    client = _client(base_url, timeout_s)
    try:
        r = await client.post(url, headers=headers, json=payload)
        if json_mode and r.status_code in (400, 422):
            payload.pop("response_format", None)
            r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
        raise LLMUnavailable(str(e)) from e


async def chat(messages: list[dict], *, json_mode: bool = False, max_tokens: int = 350, temperature: float = 0.1) -> str:
    s = get_settings()
    return await hf_chat(
        s.hf_base_url,
        s.hf_token,
        s.llm_model,
        messages,
        json_mode=json_mode,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_s=s.llm_timeout_s,
    )


def flatten_value(v) -> str:
    """Models sometimes return a list or a dict where a string was asked for; render those as readable text."""
    if isinstance(v, dict):
        return "; ".join(f"{k}: {flatten_value(x)}" for k, x in v.items() if x not in (None, "", [], {}))
    if isinstance(v, (list, tuple)):
        return ", ".join(s for s in (flatten_value(x) for x in v) if s)
    return "" if v is None else str(v)


def parse_json(text: str) -> dict | None:
    """Parse model JSON output, tolerating code fences and trailing prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


async def chat_json(messages: list[dict], *, max_tokens: int = 450, retries: int = 1) -> dict | None:
    for _ in range(retries + 1):
        out = parse_json(await chat(messages, json_mode=True, max_tokens=max_tokens))
        if out is not None:
            return out
    return None
