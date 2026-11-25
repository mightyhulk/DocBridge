"""Vector store for the clinical document repository (ChromaDB when installed, pure-Python seed KB fallback for serverless)."""
import math
import re
from collections import Counter
from dataclasses import dataclass

from .config import get_settings
from .kb_seed import CONDITIONS, DRUGS, REFERENCE

_client = None
_seed_docs: list[tuple[str, str, dict]] | None = None
_seed_vocab: set[str] | None = None
_STOP = {
    "what", "when", "where", "which", "with", "that", "this", "from", "have", "does",
    "should", "about", "there", "their", "they", "would", "could", "usual", "for",
    "and", "the", "are", "how", "can", "take", "into", "some", "most", "more", "than",
}


@dataclass
class Passage:
    text: str
    source: str
    title: str
    score: float
    url: str = ""
    direct: bool = False     # retrieved by exact entity lookup (e.g. the FDA label of the drug that was named)


def _seed_chunks() -> list[tuple[str, str, dict]]:
    global _seed_docs
    if _seed_docs is not None:
        return _seed_docs
    out: list[tuple[str, str, dict]] = []
    for name, c in CONDITIONS.items():
        base = {"title": name, "source": "Clinical reference (seed)"}
        out.append((f"{name}: {c['overview']}", f"cond::{name}::overview", base))
        out.append((f"{name} symptoms: " + "; ".join(c["symptoms"]) + ".", f"cond::{name}::symptoms", base))
        out.append((f"{name} red flags - when to seek urgent care: " + "; ".join(c["red_flags"]) + ".", f"cond::{name}::red", base))
        out.append((f"{name} management and treatment: " + "; ".join(c["management"]) + ". Medicines: " + "; ".join(c["medications"]) + ".",
                    f"cond::{name}::mgmt", base))
    for name, d in DRUGS.items():
        base = {"title": name, "source": "Drug reference (seed)"}
        aka = f" (also called {d['also']})" if d["also"] else ""
        out.append((f"{name}{aka} is a {d['class']} used for {d['uses']}.", f"drug::{name}::about", base))
        out.append((f"{name} dose and dosing: {d['dose']}", f"drug::{name}::dose", base))
        out.append((f"{name} side effects: {d['side_effects']} Warnings and contraindications: {d['warnings']}", f"drug::{name}::safety", base))
    for title, text in REFERENCE.items():
        out.append((f"{title}: {text}", f"ref::{title}", {"title": title, "source": "Clinical reference (seed)"}))
    _seed_docs = out
    return _seed_docs


def collection():
    global _client
    try:
        import chromadb
    except ImportError:
        return None
    s = get_settings()
    if _client is None:
        _client = chromadb.PersistentClient(path=str(s.chroma_dir))
    return _client.get_or_create_collection(s.chroma_collection, metadata={"hnsw:space": "cosine"})


def _chroma_ready() -> bool:
    col = collection()
    if col is None:
        return False
    try:
        return col.count() > 0
    except Exception:
        return False


def count() -> int:
    col = collection()
    if col is not None:
        try:
            n = col.count()
            if n > 0:
                return n
        except Exception:
            pass
    return len(_seed_chunks())


_vocab: set[str] | None = None


def vocabulary() -> set[str]:
    """Word stems (first 5 letters) present anywhere in the knowledge base; cached until restart/re-index."""
    global _vocab, _seed_vocab
    if _chroma_ready():
        if _vocab is None:
            got = collection().get(include=["documents", "metadatas"])
            _vocab = {w[:5] for d, m in zip(got["documents"], got["metadatas"])
                      for w in re.findall(r"[a-z][a-z0-9]{3,}", (m.get("title", "") + " " + d).lower())}
        return _vocab
    if _seed_vocab is None:
        _seed_vocab = {
            w[:5]
            for doc, _, meta in _seed_chunks()
            for w in re.findall(r"[a-z][a-z0-9]{3,}", (meta.get("title", "") + " " + doc).lower())
        }
    return _seed_vocab


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in _STOP}


def _stems(text: str) -> list[str]:
    return [w[:5] for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in _STOP]


def _seed_search(query: str, k: int) -> list[Passage]:
    """Lightweight pure-Python lexical/IDF fallback when ChromaDB is absent or empty (e.g. Vercel serverless)."""
    docs = _seed_chunks()
    q_terms = _stems(query)
    if not q_terms:
        return []
    q_set = set(q_terms)
    doc_tokens = [_stems(meta.get("title", "") + " " + doc) for doc, _, meta in docs]
    df = Counter(t for toks in doc_tokens for t in set(toks))
    n_docs = len(docs)
    max_idf = sum(math.log(1 + n_docs / (1 + df[t])) for t in q_set) or 1.0

    scored: list[Passage] = []
    for (doc, _, meta), toks in zip(docs, doc_tokens):
        t_set = set(toks)
        matched = q_set & t_set
        if not matched:
            continue
        idf_hit = sum(math.log(1 + n_docs / (1 + df[t])) for t in matched)
        title_hit = len(q_set & set(_stems(meta.get("title", "")))) / max(len(q_set), 1)
        score = min(0.95, round(0.65 * (idf_hit / max_idf) + 0.30 * title_hit, 4))
        scored.append(Passage(text=doc, source=meta.get("source", "kb"), title=meta.get("title", ""), score=score, url=""))
    scored.sort(key=lambda p: p.score, reverse=True)
    return scored[:k]


def search(query: str, k: int | None = None) -> list[Passage]:
    """Vector search then a light lexical re-rank (hybrid), falling back to pure-Python seed KB search."""
    s = get_settings()
    k = k or s.rag_top_k
    if not _chroma_ready():
        return _seed_search(query, k)
    from .embeddings import embed_query
    col = collection()
    res = col.query(query_embeddings=[embed_query(query)], n_results=max(k * 3, 10),
                    include=["documents", "metadatas", "distances"])
    qk = _keywords(query)
    out: list[Passage] = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = 1.0 - dist
        overlap = len(qk & _keywords(meta.get("title", "") + " " + doc)) / max(len(qk), 1)
        out.append(Passage(text=doc, source=meta.get("source", "kb"), title=meta.get("title", ""),
                           score=round(sim + 0.15 * overlap, 4), url=meta.get("url", "")))
    out.sort(key=lambda p: p.score, reverse=True)
    return out[:k]
