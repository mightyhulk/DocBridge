"""Client for the medical language model used by Patient Consult, hosted on Hugging Face."""
import time

from .config import get_settings
from .llm import LLMUnavailable, hf_chat, parse_json

last_endpoint: str = ""
last_ms: int = 0
last_notes: list[str] = []


def _endpoint_url() -> str:
    s = get_settings()
    return (s.med_api_url or s.hf_base_url).rstrip("/")


async def is_available() -> bool:
    return bool(get_settings().hf_token.strip())


async def status() -> dict:
    s = get_settings()
    ok = await is_available()
    return {
        "model": s.med_model,
        "available": ok,
        "provider": "huggingface",
        "endpoint": _endpoint_url() if ok else "",
        "last_ms": last_ms,
    }


async def chat(
    messages: list[dict],
    *,
    json_mode: bool = False,
    max_tokens: int | None = None,
    temperature: float = 0.2,
) -> str:
    """Query the configured Hugging Face medical model, falling back to llm_model if the primary model fails."""
    global last_endpoint, last_ms, last_notes
    s = get_settings()
    ep = _endpoint_url()
    tokens = max_tokens or s.med_max_tokens
    notes: list[str] = []
    candidates = [s.med_model]
    if s.llm_model and s.llm_model != s.med_model and not s.med_api_url:
        candidates.append(s.llm_model)

    last_err: Exception | None = None
    for idx, model in enumerate(candidates):
        t0 = time.perf_counter()
        try:
            content = await hf_chat(
                ep,
                s.hf_token,
                model,
                messages,
                json_mode=json_mode,
                max_tokens=tokens,
                temperature=temperature,
                timeout_s=s.med_timeout_s,
            )
            last_endpoint = f"huggingface:{model}"
            last_ms = int((time.perf_counter() - t0) * 1000)
            last_notes = notes
            return content
        except LLMUnavailable as e:
            last_err = e
            if idx == 0 and len(candidates) > 1:
                notes.append(f"Hugging Face model {model} unavailable ({e}); used {candidates[1]} instead")
    raise LLMUnavailable(f"medical model unavailable on Hugging Face: {last_err}")


async def chat_json(messages: list[dict], *, max_tokens: int | None = None, retries: int = 0) -> dict | None:
    for _ in range(retries + 1):
        out = parse_json(await chat(messages, json_mode=True, max_tokens=max_tokens))
        if out is not None:
            return out
    return None
