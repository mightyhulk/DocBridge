"""Local ONNX embeddings via fastembed (CPU-friendly, no server)."""
from functools import lru_cache
from threading import Lock

from .config import get_settings

_model = None
_lock = Lock()


def _get_model():
    global _model
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding
            _model = TextEmbedding(model_name=get_settings().embed_model)
        return _model


def embed_documents(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    return [v.tolist() for v in _get_model().embed(texts, batch_size=batch_size)]


@lru_cache(maxsize=2048)
def _embed_query_cached(text: str) -> tuple[float, ...]:
    # bge models are trained with an instruction prefix on queries
    prefix = "Represent this sentence for searching relevant passages: "
    return tuple(next(iter(_get_model().embed([prefix + text]))).tolist())


def embed_query(text: str) -> list[float]:
    return list(_embed_query_cached(text.strip().lower()))
