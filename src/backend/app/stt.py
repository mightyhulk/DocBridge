"""Speech-to-text: Hugging Face Whisper Inference API (serverless-ready) with optional local faster-whisper."""
import os
import tempfile
from threading import Lock

import httpx

from .config import get_settings

_models: dict = {}
_lock = Lock()

MEDICAL_PROMPT = ("Doctor-patient consultation. Terms: hypertension, diabetes mellitus, HbA1c, metformin, lisinopril, "
                  "amlodipine, atorvastatin, paracetamol, ibuprofen, amoxicillin, asthma, COPD, ECG, SpO2, mg, mcg.")

_STOCK = {"thank you very much", "thanks for watching", "thank you for watching"}
_MIME = {".wav": "audio/wav", ".webm": "audio/webm", ".ogg": "audio/ogg", ".mp4": "audio/mp4", ".mp3": "audio/mpeg", ".flac": "audio/flac"}


def _local_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _get_model(name: str | None = None):
    with _lock:
        s = get_settings()
        name = name or s.whisper_model
        if name not in _models:
            from faster_whisper import WhisperModel
            _models[name] = WhisperModel(name, device="cpu", compute_type=s.whisper_compute_type, cpu_threads=os.cpu_count() or 4)
        return _models[name]


def preload() -> None:
    if not _local_available():
        return
    s = get_settings()
    _get_model()
    if s.whisper_fast_model and s.whisper_fast_model != s.whisper_model:
        _get_model(s.whisper_fast_model)


def looks_like_a_loop(text: str) -> bool:
    """The same 1-6 word phrase repeated three or more times in a row."""
    w = text.lower().split()
    for n in range(1, 7):
        for i in range(len(w) - 3 * n + 1):
            if w[i:i + n] == w[i + n:i + 2 * n] == w[i + 2 * n:i + 3 * n]:
                return True
    return False


def clean_transcript(segments: list[tuple[str, float, float]]) -> str:
    """segments = (text, avg_logprob, no_speech_prob). Keeps only what the model was reasonably sure was speech."""
    kept = [t.strip() for t, logprob, no_speech in segments if logprob > -1.3 and not (no_speech > 0.6 and logprob < -0.8)]
    text = " ".join(k for k in kept if k).strip()
    bare = "".join(ch for ch in text.lower() if ch.isalpha() or ch == " ").strip()
    return "" if looks_like_a_loop(text) or bare in _STOCK else text


def _filter_text(text: str) -> str:
    text = text.strip()
    bare = "".join(ch for ch in text.lower() if ch.isalpha() or ch == " ").strip()
    return "" if looks_like_a_loop(text) or bare in _STOCK else text


def _decode(model, path: str):
    segments, info = model.transcribe(
        path, beam_size=1, vad_filter=True, vad_parameters={"min_silence_duration_ms": 500, "min_speech_duration_ms": 100, "speech_pad_ms": 300},
        initial_prompt=MEDICAL_PROMPT, condition_on_previous_text=False, temperature=0.0)
    segs = [(seg.text, seg.avg_logprob, seg.no_speech_prob) for seg in segments]
    return segs, info


def _transcribe_hf(audio: bytes, suffix: str) -> dict:
    s = get_settings()
    token = s.hf_token.strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not configured for speech-to-text")
    base = s.hf_base_url.rstrip("/")
    if "groq.com" in base or "openai.com" in base:
        model = s.whisper_model.split("/")[-1] if "/" in s.whisper_model else s.whisper_model
        url = base if base.endswith("/audio/transcriptions") else f"{base}/audio/transcriptions"
        mime = _MIME.get(suffix.lower(), "audio/webm")
        r = httpx.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            files={"file": (f"audio{suffix}", audio, mime)},
            data={"model": model, "prompt": MEDICAL_PROMPT, "language": "en"},
            timeout=30.0,
        )
    else:
        model = s.whisper_model if "/" in s.whisper_model else "openai/whisper-large-v3-turbo"
        url = f"https://router.huggingface.co/hf-inference/models/{model}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": _MIME.get(suffix.lower(), "audio/webm"),
        }
        r = httpx.post(url, headers=headers, content=audio, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    raw = data.get("text", "") if isinstance(data, dict) else ""
    return {"text": _filter_text(raw), "language": "en", "duration_s": 0.0, "model": model}


def transcribe(audio: bytes, suffix: str = ".webm") -> dict:
    if not _local_available():
        return _transcribe_hf(audio, suffix)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio)
        path = f.name
    try:
        s = get_settings()
        fast = s.whisper_fast_model and s.whisper_fast_model != s.whisper_model
        segs, info, used = None, None, s.whisper_model
        if fast:
            segs, info = _decode(_get_model(s.whisper_fast_model), path)
            used = s.whisper_fast_model
            sure = bool(segs) and all(lp >= s.whisper_fast_min_logprob for _, lp, _ in segs) and not looks_like_a_loop(" ".join(t for t, _, _ in segs))
            if not sure:
                segs, info = None, None
        if segs is None:
            segs, info = _decode(_get_model(), path)
            used = s.whisper_model
        return {"text": clean_transcript(segs), "language": info.language, "duration_s": round(info.duration, 2), "model": used}
    finally:
        os.unlink(path)


def available() -> bool:
    return _local_available() or bool(get_settings().hf_token.strip())
