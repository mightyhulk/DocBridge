"""Source 2 of the patient consult: RAG over real patient-doctor conversations (optional ChromaDB index).

Datasets (public, Hugging Face): lavita/ChatDoctor-HealthCareMagic-100k and ruslanmv/ai-medical-chatbot.
"""
import hashlib
import re
from dataclasses import dataclass
from threading import Lock

from .config import get_settings

_client = None
_model = None
_lock = Lock()

DATASETS = {
    "cd": ("ChatDoctor-HealthCareMagic-100k", "https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k"),
    "amc": ("AI-Medical-Chatbot", "https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot"),
}


@dataclass
class Case:
    patient: str        # what the earlier patient wrote
    reply: str          # what the doctor answered (cleaned)
    title: str
    source: str
    url: str
    score: float


# ---------- cleaning ----------
_OPENER = re.compile(r"\b(thanks?|thank you|welcome|greetings|hi|hello|hey|dear|good (morning|evening|afternoon)|chat ?doctor|"
                     r"health ?care ?magic|hcm|query|posting)\b", re.I)
_CLOSER = re.compile(r"\b(take care|regards|wish(ing)? you|hope (this|it) helps|feel free|revert|write back|get back|"
                     r"chat ?doctor|health ?care ?magic|hcm|thank|follow up with me|contact me|ask (me )?a? ?follow|best wishes)\b", re.I)
_ADVERT = re.compile(r"(https?://|www\.|@\w+\.\w+|\b\d{8,}\b|whatsapp|my clinic|consult me|\bmy fee\b|book (an )?appointment with me|"
                     r"\bchat ?doctor\b|health ?care ?magic)", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+")


def clean_patient(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = re.sub(r"^(hi|hello|hey|dear)( (doctor|dr\.?|sir|madam))?[,.! ]+", "", text, flags=re.I)
    return text


def clean_doctor(text: str) -> str:
    """Strip greetings/sign-offs ('Hi, Thanks for query... Take care. Chat Doctor.') and collapse whitespace."""
    sents = _SENT.split(re.sub(r"\s+", " ", str(text or "")).strip())
    for _ in range(3):
        if sents and len(sents[0]) < 140 and _OPENER.search(sents[0]):
            sents.pop(0)
        else:
            break
    for _ in range(3):
        if sents and (len(sents[-1]) < 120 and _CLOSER.search(sents[-1]) or re.fullmatch(r"(dr\.?\s+)?[A-Z][\w. ]{0,30}", sents[-1])):
            sents.pop()
        else:
            break
    return " ".join(sents).strip()


def usable_row(patient: str, reply: str) -> bool:
    return (len(patient) >= 30 and 120 <= len(reply) <= 2500 and not _ADVERT.search(reply)
            and not _ADVERT.search(patient) and reply.count(" ") > 20)


def row_url(rid: str, fallback: str = "") -> str:
    """Hugging Face dataset-viewer link to the exact row ('cd-1234' -> .../viewer/default/train?row=1234)."""
    prefix, _, n = rid.partition("-")
    if prefix in DATASETS and n.isdigit():
        return f"{DATASETS[prefix][1]}/viewer/default/train?row={n}"
    return fallback


def row_id(prefix: str, i: int) -> str:
    return f"{prefix}-{i}"


def reply_hash(reply: str) -> str:
    return hashlib.md5(re.sub(r"\W+", "", reply.lower())[:300].encode()).hexdigest()


# ---------- embeddings / store ----------
def _embedder():
    global _model
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding
            _model = TextEmbedding(model_name=get_settings().dialogue_embed_model)
        return _model


def embed(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    return [v.tolist() for v in _embedder().embed(texts, batch_size=batch_size)]


def embed_text_for(patient: str, title: str = "") -> str:
    return (title + ". " if title else "") + patient[:300]


def collection():
    global _client
    s = get_settings()
    if not s.dialogue_dir.exists():
        return None
    try:
        import chromadb
    except ImportError:
        return None
    if _client is None:
        _client = chromadb.PersistentClient(path=str(s.dialogue_dir))
    return _client.get_or_create_collection(s.dialogue_collection, metadata={"hnsw:space": "cosine"})


def count() -> int:
    try:
        col = collection()
        return col.count() if col is not None else 0
    except Exception:
        return 0


_STOP = set("i me my am is are was were be been the a an and or but of to in on at for with it its this that these those have has had "
            "do does did not no yes so very just also can could would should will may might get got been being since from about as by "
            "doctor please help hi hello thanks thank you your what which how why when where".split())


def _terms(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP}


def search(query: str, k: int | None = None) -> list[Case]:
    """Nearest earlier patients to `query`, hybrid-scored (semantic + shared symptom words), best first."""
    s = get_settings()
    k = k or s.dialogue_top_k
    col = collection()
    if col is None or col.count() == 0 or not query.strip():
        return []
    res = col.query(query_embeddings=embed([query.lower()]), n_results=max(k * 4, 12),
                    include=["documents", "metadatas", "distances"])
    qt = _terms(query)
    out: list[Case] = []
    seen: set[str] = set()
    for rid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        reply = meta.get("reply", "")
        h = reply_hash(reply)
        if h in seen:
            continue
        seen.add(h)
        overlap = len(qt & _terms(doc + " " + meta.get("title", ""))) / max(len(qt), 1)
        out.append(Case(patient=doc, reply=reply, title=meta.get("title", ""), source=meta.get("source", ""),
                        url=row_url(rid, meta.get("url", "")), score=round((1.0 - dist) + 0.15 * overlap, 4)))
    out.sort(key=lambda c: c.score, reverse=True)
    return out[:k]
