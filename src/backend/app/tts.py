"""Text-to-speech. Order: Piper (neural, fast, offline) -> pyttsx3 (OS voices) -> None (browser speechSynthesis)."""
import hashlib
import io
import os
import tempfile
import wave
from collections import OrderedDict
from threading import Lock

from .config import get_settings
from .safety import strip_for_speech

_piper = None
_piper_failed = False
_lock = Lock()
_cache: OrderedDict[str, bytes] = OrderedDict()   # repeated phrases (greetings, intake questions) cost 0 ms
_CACHE_MAX = 128


def _load_piper():
    global _piper, _piper_failed
    if _piper is not None or _piper_failed:
        return _piper
    path = get_settings().piper_voice.resolve()
    try:
        from piper import PiperVoice
        _piper = PiperVoice.load(str(path))
    except Exception:
        _piper_failed = True
    return _piper


def _piper_wav(text: str) -> bytes | None:
    voice = _load_piper()
    if voice is None:
        return None
    buf = io.BytesIO()
    with _lock, wave.open(buf, "wb") as wf:
        if hasattr(voice, "synthesize_wav"):          # piper-tts >= 1.3
            voice.synthesize_wav(text, wf)
        else:                                          # piper-tts 1.2.x
            voice.synthesize(text, wf)
    return buf.getvalue()


def _pyttsx3_wav(text: str) -> bytes | None:
    try:
        import pyttsx3
    except ImportError:
        return None
    path = tempfile.mktemp(suffix=".wav")
    try:
        with _lock:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.save_to_file(text, path)
            engine.runAndWait()
        with open(path, "rb") as f:
            return f.read() or None
    except Exception:
        return None
    finally:
        if os.path.exists(path):
            os.unlink(path)


def synthesize(text: str) -> tuple[bytes | None, str]:
    """Return (wav_bytes, engine). wav_bytes is None when the client should use browser speech."""
    text = strip_for_speech(text)[:1200]
    if not text:
        return None, "none"
    key = hashlib.sha1(text.encode()).hexdigest()
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key], "cache"
    for name, fn in (("piper", _piper_wav), ("pyttsx3", _pyttsx3_wav)):
        try:
            wav = fn(text)
        except Exception:
            wav = None
        if wav:
            _cache[key] = wav
            if len(_cache) > _CACHE_MAX:
                _cache.popitem(last=False)
            return wav, name
    return None, "none"


def engine_name() -> str:
    if _load_piper() is not None:
        return "piper"
    try:
        import pyttsx3  # noqa: F401
        return "pyttsx3"
    except ImportError:
        return "browser"
