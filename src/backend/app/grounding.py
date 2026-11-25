"""Entity-coverage gate for retrieved passages.

Vector similarity alone happily returns the 'closest' passage for a question about something that is not in the
corpus (e.g. a made-up drug returns another drug's dose). A passage is only usable if it actually mentions the
specific entities the user asked about.

'Specific entity' = a rare word: one that is not a whole word in the embedding model's vocabulary (drug and disease
names such as 'zorbitrex' or 'metformin' split into several word-pieces; everyday words such as 'prescription' or
'range' are single tokens). Every rare word in the question must appear in the passage; everyday words only need to
match loosely.
"""
import re

from .safety import _STOP

# Words that describe the *type* of question rather than its subject.
_GENERIC = set(
    "dose doses dosage dosing usual adult adults normal recommended typical symptom symptoms sign signs treatment "
    "treat treated treating cause causes causing urgent care flag flags much take taking taken many often long side "
    "effect effects used uses drug drugs medicine medication medications patient patients disease diseases syndrome "
    "disorder condition conditions latest current maximum minimum daily tell know need needs list help best "
    "painkiller painkillers analgesic analgesics tablet tablets pill pills".split()
)

_tok = None
_tok_loaded = False


def _tokenizer():
    global _tok, _tok_loaded
    if not _tok_loaded:
        _tok_loaded = True
        try:
            from .embeddings import _get_model
            _tok = getattr(_get_model().model, "tokenizer", None)
        except Exception:
            _tok = None
    return _tok


_COMMON_WORDS = {
    "adult", "adults", "after", "all", "allergic", "allergies", "allergy", "also", "another", "any", "anything",
    "apart", "avoid", "avoided", "away", "bad", "before", "best", "better", "between", "blood", "body", "breath",
    "breathing", "can", "care", "cause", "causes", "check", "chest", "child", "children", "chronic", "clinic",
    "clinical", "cold", "common", "compare", "control", "cough", "could", "daily", "day", "days", "describe",
    "difference", "differences", "different", "doctor", "doctors", "dosage", "dose", "doses", "drug", "drugs",
    "during", "effect", "effects", "elderly", "emergency", "every", "everything", "explain", "far", "feel",
    "feeling", "female", "fever", "few", "find", "first", "flu", "fluid", "fluids", "food", "gain", "give",
    "given", "giving", "good", "guideline", "guidelines", "head", "headache", "health", "healthy", "heart", "help",
    "high", "home", "hospital", "hour", "hours", "infection", "infections", "internet", "large", "last", "latest",
    "less", "level", "levels", "list", "long", "loss", "low", "made", "make", "makes", "male", "man", "manage",
    "management", "many", "may", "mcg", "mean", "means", "medical", "medicine", "medicines", "men", "mg", "might",
    "mild", "minute", "minutes", "month", "months", "more", "most", "much", "must", "need", "needs", "new", "none",
    "normal", "nose", "nothing", "often", "older", "once", "online", "only", "other", "over", "pain", "possible",
    "pregnancy", "pregnant", "prescribe", "prescribed", "prescription", "prescriptions", "pressure", "prevent",
    "prevention", "problem", "problems", "range", "recent", "recently", "recommend", "recommended", "rest", "risk",
    "risks", "safe", "same", "search", "separate", "serious", "severe", "short", "should", "show", "side", "sign",
    "signs", "skin", "sleep", "small", "some", "something", "start", "starting", "stomach", "stop", "symptom",
    "symptoms", "take", "taken", "taking", "target", "targets", "tell", "throat", "time", "times", "together",
    "treat", "treated", "treatment", "twice", "under", "update", "updated", "updates", "urgent", "use", "used",
    "using", "usual", "warning", "warnings", "water", "web", "week", "weeks", "weight", "well", "with", "without",
    "woman", "women", "work", "working", "works", "worse", "would", "year", "years", "young",
}


def is_common_word(w: str) -> bool:
    """True for everyday English words (whole-word entries in the tokenizer vocabulary, or fallback word list)."""
    tok = _tokenizer()
    if tok is None:
        return w.lower() in _COMMON_WORDS
    candidates = [w]
    for suf in ("s", "es", "ed", "d", "ing"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            candidates.append(w[: -len(suf)])
    return any(tok.token_to_id(c) is not None for c in candidates)


def key_terms(text: str) -> list[str]:
    """Subject words of a question. Keeps 3-letter clinical acronyms (LDL, INR, BMI, MRI)."""
    words = re.findall(r"[a-z][a-z0-9]{2,}", text.lower())
    return [w for w in words if w not in _STOP and w not in _GENERIC]


def _stem(w: str) -> str:
    return w[:5]


def _stems(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z][a-z0-9]{2,}", text.lower())}


def covers(query: str, passage, min_coverage: float = 0.15) -> bool:
    """A passage may answer the question only if
      1. every rare entity in the question appears in it,
      2. at least one subject term matches, and
      3. it is ABOUT the topic: some rare entity is in its title or opening words, not merely mentioned in passing
         (otherwise 'What is gout?' would be answered from the furosemide passage that lists gout as a side effect).
    """
    terms = key_terms(query)
    if not terms:
        return True
    hay = _stems(passage.title + " " + passage.text)
    matched = [t for t in terms if _stem(t) in hay]
    rare = [t for t in terms if not is_common_word(t)]
    if any(t not in matched for t in rare):
        return False
    # 'about' = the title plus the section label before the first colon ("Furosemide side effects:")
    label = passage.text.split(":", 1)[0] if ":" in passage.text[:90] else ""
    topic = _stems(passage.title + " " + label)
    subject = rare or [t for t in terms if t not in ("make", "good", "bad", "long", "should", "take")]
    if getattr(passage, "source", "").startswith(("Clinical reference", "Drug reference")) or rare:
        # for the curated KB (and for any question with a rare entity) the passage must be about the subject, not merely
        # mention it; ordinary-word questions ('what vaccines do adults need') need a title/section match too
        if subject and not any(_stem(t) in topic for t in subject):
            return False
    return len(matched) >= 1 and len(matched) / len(terms) >= min_coverage


def unverified_entities(query: str, vocab: set[str]) -> list[str]:
    """Rare terms in the question that appear nowhere in the knowledge base (made-up, mistyped or out-of-corpus).

    A small LLM will happily 'explain' such a term from imagination, so Tier 1 must not answer these on its own.
    """
    return [t for t in key_terms(query) if not is_common_word(t) and _stem(t) not in vocab]
