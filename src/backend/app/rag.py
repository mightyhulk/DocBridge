"""Answer pipeline.

  Tier 0  exact lookups of the selected patient's record (SQL)    -> deterministic, never generated
  Tier 1  primary LLM (+ patient record if one is selected)      -> accepted only if self-rated confident
  Tier 2  RAG over the clinical vector DB                         -> accepted only if grounded in passages
  Tier 3  free online sources (MedlinePlus / openFDA / Wikipedia) -> accepted only if grounded in passages
  else    safe abstention (never a guess)

Tier-2 retrieval starts in parallel with Tier 1, so falling through costs no extra retrieval latency.
"""
import asyncio
import re
import time
from dataclasses import dataclass, field

from . import clinical, fallback, grounding, llm, patients, safety, vectorstore, websearch
from .config import get_settings

_WEB_ASK = re.compile(r"\b(?:search (?:the )?(?:web|online|internet)|google|look (?:it )?up|latest|newest|recent(?:ly)? (?:guideline|update|study|trial)s?|current guideline|updated guideline|2024|2025|2026)\b", re.I)

SYSTEM_BASE = (
    "You are a clinical information assistant inside a hospital system. Be accurate and brief: at most 2 short "
    "sentences (about 40 words) in plain language; for list-type questions one sentence listing all items. "
    "Always reply in English. Never invent facts. You support clinicians; you do not diagnose or prescribe."
)

TIER1_PROMPT = SYSTEM_BASE + """
Answer from your own medical knowledge (and the patient record if provided). Reply ONLY with a JSON object with the
keys "answer" (your actual answer sentence), "confidence" (a number between 0 and 1) and "needs_reference" (true or false).
Rules:
- confidence >= 0.85 ONLY for well-established textbook facts you are certain about.
- needs_reference = true for drug doses, interactions, numeric thresholds, local protocols, recent guidelines,
  or anything about a patient that is not in the provided record.
- If you are unsure, say so with low confidence rather than guessing."""

GROUNDED_PROMPT = SYSTEM_BASE + """
Answer ONLY from the numbered context passages (and the patient record if provided). Cite passages like [1].
Reply ONLY with a JSON object with the keys "supported" (true or false), "answer" (your actual answer sentence) and
"confidence" (a number between 0 and 1).
- Be complete on the facts asked: for doses give the starting dose, usual range AND maximum when the passages state them;
  for symptoms/red flags list every item the passages give.
- If the passages answer only part of the question, answer that part from them and set supported=true.
- If the passages do not contain the answer at all, set supported=false and answer "I could not find this in the references."
- Never add facts that are not in the passages."""

ABSTAIN = ("I could not find a reliable answer to this in the medical references available to me. "
           "Please consult a clinician or a pharmacist.")


@dataclass
class Answer:
    answer: str
    tier: str                      # record | clinical | llm | rag | web | extractive | abstain | emergency
    confidence: float = 0.0
    sources: list[dict] = field(default_factory=list)
    emergency: bool = False
    flags: list[str] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    disclaimer: str = safety.DISCLAIMER
    structured: dict | None = None   # machine-readable detail for the UI (e.g. a medication check's findings)


_NON_LATIN = re.compile(r"[Ѐ-ӿ֐-ۿऀ-෿぀-ヿ㐀-鿿가-힯]")


def _usable(text) -> bool:
    """Reject empty output, echoed schema placeholders such as '<text>', and answers that drift into another
    script (Qwen occasionally code-switches to Chinese mid-sentence)."""
    t = str(text or "").strip()
    if "could not find" in t.lower() and len(t) < 90:      # the model's own 'no answer' line, even if it flagged supported=true
        return False
    return len(t) > 3 and not (t.startswith("<") and t.endswith(">")) and not _NON_LATIN.search(t)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def ungrounded_numbers(answer: str, *sources: str) -> list[str]:
    """Numbers stated in the answer that appear in none of the sources. A small model can invent a dose, interval
    or threshold even inside an otherwise grounded answer, so any such number disqualifies the generated text."""
    allowed: set[str] = set()
    for s in sources:
        allowed |= _numbers(s)
    return sorted(_numbers(re.sub(r"\[\d+\]", "", answer)) - allowed)


def _excerpt(text: str, limit: int = 450) -> str:
    """First sentences of a source, cut at a sentence boundary."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[: end + 1] if end > 120 else cut.rstrip() + "…"


def _src(p: vectorstore.Passage) -> dict:
    return {"title": p.title, "source": p.source, "url": p.url, "score": p.score, "snippet": p.text[:240]}


def _numbered(passages: list[vectorstore.Passage]) -> str:
    # Prompt size is the main latency lever on CPU: keep only the best passages, trimmed.
    return "\n\n".join(f"[{i}] ({p.source} - {p.title}) {p.text[:900]}" for i, p in enumerate(passages, 1))


def _history_msgs(history: list[dict] | None) -> list[dict]:
    return [{"role": h["role"], "content": h["content"][:600]} for h in (history or [])[-4:]
            if h.get("role") in ("user", "assistant")]


_FOLLOW_UP = re.compile(r"^\s*(and|also|what about|how about|but)\b|\b(it|its|they|them|their|that|this|those|these|he|she)\b", re.I)


def _retrieval_query(message: str, history: list[dict] | None) -> str:
    """Follow-ups ('and how is it treated?') borrow the previous question; self-contained questions never do."""
    if history and len(message.split()) < 8 and (_FOLLOW_UP.search(message) or not grounding.key_terms(message)):
        prev = next((h["content"] for h in reversed(history) if h.get("role") == "user"), "")
        return f"{prev} {message}"
    return message


async def _grounded(message: str, passages: list[vectorstore.Passage], patient_ctx: str, history) -> tuple[dict | None, bool]:
    ctx = f"CONTEXT PASSAGES:\n{_numbered(passages)}"
    if patient_ctx:
        ctx += f"\n\nPATIENT RECORD (confidential):\n{patient_ctx}"
    msgs = [{"role": "system", "content": GROUNDED_PROMPT}, *_history_msgs(history),
            {"role": "user", "content": f"{ctx}\n\nQUESTION: {message}"}]
    out = await llm.chat_json(msgs, max_tokens=200, retries=0)
    return out, out is not None


async def answer(message: str, patient_id: str | None = None, history: list[dict] | None = None) -> Answer:
    s = get_settings()
    t0 = time.perf_counter()
    timings: dict[str, int] = {}

    def lap(name: str, since: float) -> None:
        timings[name] = int((time.perf_counter() - since) * 1000)

    # --- safeguards -------------------------------------------------------------------------------
    flags = safety.emergency_flags(message)
    if flags:
        return Answer(safety.EMERGENCY_MESSAGE, "emergency", 1.0, emergency=True, flags=flags)

    patient = patients.get(patient_id) if patient_id else None
    # Tier 0a: clinical safety questions (is drug X safe for this patient, do X and Y interact, what is wrong in this chart) are answered
    # by curated rules, never by a model: deterministic, instant, and they cite exactly what was checked.
    if (c := clinical.answer(message, patient)):
        src = [{"title": "Clinical decision-support rules (curated from BNF / FDA labels / ACC-AHA / ADA)", "source": "Clinical rules", "url": "", "score": 1.0, "snippet": ""}]
        if patient:
            src.append({"title": f"Patient record {patient['patient_id']}", "source": "EHR", "url": "", "score": 1.0, "snippet": ""})
        return Answer(c["text"], "clinical", 1.0, src, flags=["clinical_rules", c["kind"]], structured={"kind": c["kind"], **c["data"]},
                      timings_ms={"total": int((time.perf_counter() - t0) * 1000)})

    # --- Tier 0: exact EHR lookup (deterministic, ~0 ms, cannot hallucinate) ---------------------------
    if patient and (fact := patients.answer_from_record(message, patient)):
        src = [{"title": f"Patient record {patient['patient_id']}", "source": "EHR", "url": "", "score": 1.0, "snippet": ""}]
        return Answer(fact, "record", 1.0, src, timings_ms={"total": int((time.perf_counter() - t0) * 1000)})

    patient_ctx = patients.to_context(patient) if patient else ""
    names = [patient["full_name"]] if patient else []
    dosing = safety.is_dosing_query(message)
    rq = _retrieval_query(message, history)

    # Retrieval costs ~10 ms, so run it first: it tells us whether the curated KB already has a strong, on-topic
    # source (in which case a small model's memory should not override it, and Tier 1 would be a wasted LLM call).
    t = time.perf_counter()
    kb = await asyncio.to_thread(vectorstore.search, rq)
    lap("retrieval", t)
    kb_good = [p for p in kb if p.score >= s.rag_min_score and grounding.covers(rq, p)]
    strong_kb = any(p.score >= s.kb_strong_score for p in kb_good)
    # A specialised term found nowhere in the KB (made-up, mistyped, or simply out of corpus): the model's memory is
    # not to be trusted for it, so skip Tier 1 and let the online sources verify it (or the system abstains).
    # Questions about "this patient" are answered from the record only (never sent to the web).
    patient_specific = bool(patient and patients._PATIENT_REF.search(message))
    unverified = [] if patient_specific else await asyncio.to_thread(
        lambda: grounding.unverified_entities(rq, vectorstore.vocabulary()))
    if unverified:
        flags = [*flags, "unverified_term:" + ",".join(unverified)]
    llm_up = await llm.is_available()
    web_asked = bool(not patient_specific and _WEB_ASK.search(message))      # "search the web" / "latest guideline": go straight online

    # Doses are quoted verbatim from the reference, never paraphrased by a model (a small LLM once reported an
    # elderly starting dose as "the usual adult dose"). Zero LLM latency as a side effect.
    if dosing and kb_good and "dose and dosing:" in kb_good[0].text and not web_asked:
        top = kb_good[0]
        timings["total"] = int((time.perf_counter() - t0) * 1000)
        return Answer(top.text.replace(" dose and dosing:", " - reference dosing:"), "rag", top.score, [_src(top)],
                      flags=["verbatim_reference_dose"], timings_ms=timings)

    # --- Tier 1: primary LLM ----------------------------------------------------------------------
    if llm_up and not unverified and not web_asked and not (strong_kb and not s.strict_llm_first):
        t = time.perf_counter()
        user = message if not patient_ctx else f"PATIENT RECORD (confidential):\n{patient_ctx}\n\nQUESTION: {message}"
        try:
            out = await llm.chat_json([{"role": "system", "content": TIER1_PROMPT}, *_history_msgs(history),
                                       {"role": "user", "content": user}], max_tokens=160, retries=0)
        except llm.LLMUnavailable:
            out, llm_up = None, False
        lap("tier1_llm", t)
        if out and llm_up:
            conf = float(out.get("confidence") or 0)
            if (conf >= s.llm_confidence_threshold and not out.get("needs_reference") and not dosing and _usable(out.get("answer"))
                    and not ungrounded_numbers(str(out["answer"]), message, patient_ctx)):
                src = [{"title": f"Patient record {patient['patient_id']}", "source": "EHR", "url": "", "score": 1.0, "snippet": ""}] if patient else []
                timings["total"] = int((time.perf_counter() - t0) * 1000)
                return Answer(str(out["answer"]).strip(), "llm", conf, src, timings_ms=timings)

    # --- Tier 2: RAG over the vector DB -----------------------------------------------------------
    if kb_good and not web_asked:
        if not llm_up:   # degraded mode: no generator available, return the best extract verbatim
            timings["total"] = int((time.perf_counter() - t0) * 1000)
            return Answer(kb_good[0].text, "extractive", kb_good[0].score, [_src(p) for p in kb_good[:2]],
                          flags=["llm_unavailable"], timings_ms=timings)
        t = time.perf_counter()
        out, _ = await _grounded(message, kb_good, patient_ctx, history)
        lap("tier2_llm", t)
        if out and out.get("supported") and _usable(out.get("answer")):
            timings["total"] = int((time.perf_counter() - t0) * 1000)
            bad = ungrounded_numbers(str(out["answer"]), message, patient_ctx, *(p.text for p in kb_good))
            if bad:   # the model stated a number the references do not contain: quote the reference instead
                return Answer(kb_good[0].text, "rag", kb_good[0].score, [_src(p) for p in kb_good],
                              flags=["numeric_mismatch_verbatim:" + ",".join(bad)], timings_ms=timings)
            return Answer(str(out["answer"]).strip(), "rag", float(out.get("confidence") or kb_good[0].score),
                          [_src(p) for p in kb_good], timings_ms=timings)

    if patient_specific:
        timings["total"] = int((time.perf_counter() - t0) * 1000)
        return Answer("I could not find this in the patient's record or the medical references. Please check the chart directly.",
                      "abstain", 0.0, flags=["not_in_record"], timings_ms=timings)

    # --- Tier 3: free online fallback (anonymised keywords only) ----------------------------------
    t = time.perf_counter()
    web = await fallback.search(rq, names)
    web = [p for p in web if p.direct or grounding.covers(rq, p)]     # direct = the named drug's own FDA label
    # Full web search (trusted medical sites first, pages with links): when the reference APIs found nothing, or when the
    # clinician explicitly asks to search / for the latest guidance. Only anonymised topic keywords leave the machine.
    if not web or web_asked:
        try:
            found = await websearch.search(rq, max_pages=3, suffix="clinical guideline dosing", names=names)
        except Exception:
            found = []
        seen = {p.url for p in web if p.url}
        web += [p for p in found if p.url not in seen and (p.direct or grounding.covers(rq, p))]
        if found:
            flags.append("web_search")
    lap("tier3_web", t)
    if web:
        if not llm_up:
            timings["total"] = int((time.perf_counter() - t0) * 1000)
            return Answer(_excerpt(web[0].text), "extractive", web[0].score, [_src(p) for p in web[:2]],
                          flags=["llm_unavailable", "online_source"], timings_ms=timings)
        t = time.perf_counter()
        web = web[:3] if sum(p.direct for p in web) >= 2 else web[:2]    # two labels needed for interaction questions
        out, _ = await _grounded(message, web, patient_ctx, history)
        lap("tier3_llm", t)
        if out and out.get("supported") and _usable(out.get("answer")):
            timings["total"] = int((time.perf_counter() - t0) * 1000)
            bad = ungrounded_numbers(str(out["answer"]), message, patient_ctx, *(p.text for p in web))
            if bad:
                return Answer(_excerpt(web[0].text), "web", web[0].score, [_src(p) for p in web],
                              flags=["online_source", "verbatim_excerpt", "numeric_mismatch_verbatim:" + ",".join(bad)],
                              timings_ms=timings)
            return Answer(str(out["answer"]).strip(), "web", float(out.get("confidence") or web[0].score),
                          [_src(p) for p in web], flags=["online_source"], timings_ms=timings)
        # The model would not (or could not) answer, but an on-topic authoritative page exists: quote it rather than
        # leave the user empty-handed. Verbatim text cannot be hallucinated, and the source is shown.
        timings["total"] = int((time.perf_counter() - t0) * 1000)
        direct = [p for p in web if p.direct]
        text = (" ".join(f"{p.title}: {_excerpt(p.text, 300)}" for p in direct[:2]) if len(direct) >= 2
                else _excerpt(web[0].text))
        return Answer(text, "web", min(web[0].score, 0.6), [_src(p) for p in web],
                      flags=["online_source", "verbatim_excerpt"], timings_ms=timings)

    timings["total"] = int((time.perf_counter() - t0) * 1000)
    return Answer(ABSTAIN, "abstain", 0.0, flags=[*flags, "no_reliable_source"], timings_ms=timings)
