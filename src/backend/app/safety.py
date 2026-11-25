"""Healthcare safeguards: emergency detection, PII masking, dosing guard, disclaimers."""
import re

DISCLAIMER = (
    "AI-generated information to support, not replace, a licensed clinician. "
    "Verify before acting; do not use for emergencies."
)

EMERGENCY_MESSAGE = (
    "This may be a medical emergency. Call your local emergency number now "
    "(112 / 911 / 108) or go to the nearest emergency department. Do not wait for an AI answer."
)

_EMERGENCY_PATTERNS = [
    r"chest (pain|pressure|tightness)",
    r"(can'?t|cannot|unable to|struggling to|trouble) breathe?",
    r"(severe|sudden) (shortness of breath|difficulty breathing)",
    r"not breathing",
    r"(face|facial) droop",
    r"slurred speech",
    r"sudden (weakness|numbness|confusion|loss of vision|severe headache)",
    r"(suicid|kill myself|end my life|self[- ]harm|want to die)",
    r"(severe|heavy|uncontrolled) bleeding",
    r"(unconscious|unresponsive|passed out|collapsed)",
    r"(having|had) a seizure",
    r"(throat|tongue|lips?) (is |are )?swelling",
    r"(overdose|took too many (pills|tablets))",
    r"anaphyla",
]
_EMERGENCY_RE = re.compile("|".join(_EMERGENCY_PATTERNS), re.I)
# Purely informational questions ("what are the red flags of stroke?") must not trigger the alarm.
_INFO_RE = re.compile(r"^\s*(what|which|how|why|when|list|explain|define|describe|tell me about)\b", re.I)
_NOW_RE = re.compile(r"\b(right now|currently|at the moment|i am|i'm|he is|she is|my (father|mother|child|wife|husband))\b", re.I)


def emergency_flags(text: str) -> list[str]:
    """Return matched emergency phrases. Informational questions are ignored unless they describe a live situation."""
    found = [m.group(0).lower() for m in _EMERGENCY_RE.finditer(text)]
    if found and _INFO_RE.match(text) and not _NOW_RE.search(text):
        return []
    return sorted(set(found))


_DOSE_RE = re.compile(
    r"\b(dose|dosage|dosing|how (much|many) (mg|mcg|tablets?|pills?)|max(imum)? (daily )?(dose|amount)|"
    r"\d+\s?(mg|mcg|g|ml|iu)\b|overdose|how often (can|should) (i|he|she|they) take)\b",
    re.I,
)


def is_dosing_query(text: str) -> bool:
    return bool(_DOSE_RE.search(text))


_PII_PATTERNS = [
    (re.compile(r"\bP\d{7}\b"), "[PATIENT_ID]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
    (re.compile(r"\+?\d[\d\s().-]{8,}\d"), "[PHONE]"),
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[DATE]"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[DATE]"),
    (re.compile(r"\b\d{6,}\b"), "[NUMBER]"),
]


def mask_pii(text: str, names: list[str] | None = None) -> str:
    for rx, repl in _PII_PATTERNS:
        text = rx.sub(repl, text)
    for name in names or []:
        for part in name.split():
            if len(part) > 2:
                text = re.sub(rf"\b{re.escape(part)}(?:'s)?\b", "[NAME]", text, flags=re.I)
    return text


_STOP = set(
    "a an the is are was were be been am i me my we you your he she it they them his her their of to in on at for "
    "with and or but if then than so do does did can could should would will may might have has had what which who "
    "whom how why when where please tell about explain patient doctor give me this that these those any some "
    "there here as by from into over under again also just very much more most "
    "whats hows whos wheres thats theres isnt arent dont doesnt cant".split()
)


def to_search_query(text: str, names: list[str] | None = None, max_terms: int = 8) -> str:
    """Reduce a question to anonymous medical keywords before it leaves the building (online fallback)."""
    masked = mask_pii(text, names)
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]+", masked)
    # Drop capitalised mid-sentence words (likely names); medical terms in questions are normally lowercase.
    kept = [w for i, w in enumerate(words) if w.lower() not in _STOP and not (i > 0 and w[0].isupper() and w[1:].islower())]
    kept = [w for w in kept if w not in ("PATIENT_ID", "EMAIL", "PHONE", "DATE", "NUMBER", "NAME")]
    return " ".join(kept[:max_terms])


def strip_for_speech(text: str) -> str:
    """Remove markdown, citation markers and bullets so TTS reads naturally."""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()
