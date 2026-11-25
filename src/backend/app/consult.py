"""Patient consult: a patient describes a problem, the bot interviews them, rates the severity, answers their medical
questions from one of three knowledge sources, and says which specialist to see.

  Interview   deterministic (no model call): complaint -> duration -> severity -> a symptom-specific question -> age/conditions
  Severity    triage.assess(): rules, never a model's opinion  ->  self_care | see_doctor | urgent | emergency
  Answer      chosen by complexity / severity, escalating when a source has nothing reliable to say:
                1. minor problem          -> the Hugging Face medical model's own knowledge
                2. complex / not confident -> RAG over real patient-doctor conversations (Hugging Face datasets)
                3. nothing relevant found  -> live web search on reputable medical sites, with the links it used
              No source at all -> an honest "please see a doctor" (never a guess).
"""
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field

from . import clinic, db, dialogues, grounding, interview, medicines, medllm, safety, triage, websearch
from .config import get_settings
from .llm import LLMUnavailable
from .rag import _excerpt, _usable, ungrounded_numbers
from .triage import Assessment, Facts

SOURCE_LABELS = {
    "model": "AI medical model (Hugging Face)",
    "dialogues": "Real doctor-patient conversations (ChatDoctor / AI-Medical-Chatbot datasets)",
    "web": "Web search (reputable medical sites)",
    "formulary": "Medicine reference (standard OTC label doses) checked against your details",
    "none": "No reliable source found",
}

SYSTEM = ("You are a careful medical information assistant talking to a patient in plain English. No diagnosis, never state drug doses. "
          "Only mention causes that fit the symptoms the patient actually reported.")

MODEL_PROMPT = SYSTEM + """
Patient: {facts}
Question: {question}
Give safe practical advice in at most 3 short sentences: the likely harmless explanation, simple home care, and the warning signs that mean seeing a doctor.
Reply ONLY with JSON: {{"answer": "<advice>", "confidence": <0 to 1>, "needs_doctor": <true|false>}}.
confidence 0.7+ only for common, benign, well-established situations; otherwise lower."""

GROUNDED_PROMPT = SYSTEM + """
Patient: {facts}
Question: {question}
{label}:
{context}
Using ONLY the numbered passages, answer in at most 4 short sentences and cite like [1]. If they do not address the question, set supported to false.
Reply ONLY with JSON: {{"supported": <true|false>, "answer": "<advice>", "confidence": <0 to 1>}}."""

DEFAULT_QUESTION = "What could be causing this, what can I do at home, and when should I see a doctor?"
_DOSE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|g|ml|iu|units?)\b", re.I)
_RISKY_Q = re.compile(r"interact|together with|safe to (take|use|give)|can i (take|give|use)|pregnan|breast ?feed|alternative|side effect|"
                      r"antibiotic|prescri|steroid|\bdose|dosage|how (much|many)|\bmg\b|children|infant|baby", re.I)
_SKIP = re.compile(r"^\s*(skip|pass|i? ?don'?t know|i? ?do not know|i'?m not sure|i am not sure|not sure|can'?t say|cannot say|no idea|n/?a|unknown)\W*$", re.I)
_NONE = re.compile(r"^\s*(no|none|nothing|nope|nil|not really)\W*$", re.I)


# ---------- session ----------
@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)
    facts: Facts = field(default_factory=Facts)
    texts: list[str] = field(default_factory=list)       # everything the patient has said
    symptom_texts: list[str] = field(default_factory=list)   # ... except the answer about age / existing conditions
    history: list[dict] = field(default_factory=list)    # chat log {role, content}
    asked: str | None = None                             # interview slot awaiting an answer
    done_slots: set[str] = field(default_factory=set)
    retried_severity: bool = False
    stage: str = "interview"                             # interview | advice
    assessment: Assessment | None = None
    location: dict | None = None                         # {lat, lon, label}


_sessions: dict[str, Session] = {}
MAX_SESSIONS = 300


def serialize_session(s: Session) -> dict:
    f = s.facts
    last_bot = next((h["content"][:250] for h in reversed(s.history) if h.get("role") == "assistant"), "")
    return {
        "id": s.id,
        "created": s.created,
        "facts": {
            "text": f.text[:1200],
            "symptom_text": f.symptom_text[:800],
            "duration_days": f.duration_days,
            "trend": f.trend,
            "severity": f.severity,
            "age": f.age,
            "risks": sorted(f.risks),
            "temp_c": f.temp_c,
            "allergies": sorted(f.allergies),
            "medicines": sorted(f.medicines),
            "allergies_asked": f.allergies_asked,
            "slots": f.slots,
        },
        "texts": [t[:200] for t in s.texts[-10:]],
        "symptom_texts": [t[:200] for t in s.symptom_texts[-10:]],
        "history": [{"role": "assistant", "content": last_bot}] if last_bot else [],
        "asked": s.asked,
        "done_slots": sorted(s.done_slots),
        "retried_severity": s.retried_severity,
        "stage": s.stage,
        "assessment": asdict(s.assessment) if s.assessment else None,
        "location": s.location,
    }


def restore_session(d: dict) -> Session:
    fd = d.get("facts") or {}
    facts = Facts(
        text=fd.get("text", ""),
        symptom_text=fd.get("symptom_text", ""),
        duration_days=fd.get("duration_days"),
        trend=fd.get("trend"),
        severity=fd.get("severity"),
        age=fd.get("age"),
        risks=set(fd.get("risks") or ()),
        temp_c=fd.get("temp_c"),
        allergies=set(fd.get("allergies") or ()),
        medicines=set(fd.get("medicines") or ()),
        allergies_asked=bool(fd.get("allergies_asked", False)),
        slots=dict(fd.get("slots") or {}),
    )
    ad = d.get("assessment")
    assessment = Assessment(**ad) if ad else None
    s = Session(
        id=d["id"],
        created=float(d.get("created") or time.time()),
        facts=facts,
        texts=list(d.get("texts") or ()),
        symptom_texts=list(d.get("symptom_texts") or ()),
        history=list(d.get("history") or ()),
        asked=d.get("asked"),
        done_slots=set(d.get("done_slots") or ()),
        retried_severity=bool(d.get("retried_severity", False)),
        stage=d.get("stage", "interview"),
        assessment=assessment,
        location=d.get("location"),
    )
    _sessions[s.id] = s
    return s


def save_session(s: Session) -> None:
    _sessions[s.id] = s
    try:
        db.init_schema()
        c = db.conn()
        c.execute(
            "INSERT INTO consult_sessions(id, updated_at, data_json) VALUES (?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at, data_json=excluded.data_json",
            (s.id, time.time(), json.dumps(serialize_session(s), ensure_ascii=False)),
        )
        c.commit()
    except Exception:
        pass


def start() -> tuple[Session, str]:
    if len(_sessions) >= MAX_SESSIONS:
        for sid in sorted(_sessions, key=lambda i: _sessions[i].created)[: MAX_SESSIONS // 3]:
            _sessions.pop(sid, None)
    s = Session()
    q = ("Hello, I'm your medical assistant. Please describe the problem you are facing - what you feel, and where. "
         "I'll ask a few questions, then tell you how urgent it is and which doctor to see.")
    s.asked = "complaint"
    s.history.append({"role": "assistant", "content": q})
    save_session(s)
    return s, q


def get(sid: str) -> Session | None:
    if sid in _sessions:
        return _sessions[sid]
    try:
        db.init_schema()
        row = db.conn().execute("SELECT data_json FROM consult_sessions WHERE id=?", (sid,)).fetchone()
        if row:
            return restore_session(json.loads(row["data_json"]))
    except Exception:
        pass
    return None


def _next_step(s: Session) -> tuple[str, str, str] | None:
    """The next interview question (id, text, why): chosen fresh each turn from what is known (see interview.py), never one the
    patient has already answered."""
    steps = interview.plan(s.facts, s.done_slots)
    return steps[0] if steps else None


def progress(s: Session) -> float:
    if s.stage == "advice":
        return 1.0
    answered = len(s.done_slots)
    remaining = len(interview.plan(s.facts, s.done_slots))
    return round(answered / ((answered + remaining) or 1), 2)


def summary(s: Session) -> str:
    """Compact patient summary for the prompts."""
    f = s.facts
    parts = []
    if f.age is not None:
        parts.append(f"age {f.age}")
    parts.append("problem: " + (s.texts[0][:260] if s.texts else "not described"))
    if f.duration_days is not None:
        parts.append(f"duration: {f.duration_days:g} days" + (f", {f.trend}" if f.trend else ""))
    if f.severity is not None:
        parts.append(f"severity {f.severity}/10")
    if f.temp_c is not None:
        parts.append(f"temperature {triage.f_from_c(f.temp_c)}°F")
    if f.risks:
        parts.append("conditions: " + ", ".join(sorted(f.risks)))
    if s.assessment and s.assessment.possible:
        parts.append("possible: " + ", ".join(d["name"].split(" (")[0] for d in s.assessment.possible[:2]))
    extra = [t[:160] for t in s.texts[1:3]]
    if extra:
        parts.append("also said: " + " | ".join(extra))
    return "; ".join(parts)


def _learn(s: Session, text: str, asked: str | None) -> None:
    """Pull every fact we can out of this message, whichever question it answers."""
    f = s.facts
    f.text = "\n".join(s.texts)
    if asked not in ("profile", "meds") and (d := triage.parse_duration_days(text)) is not None:      # "39 years" there is an age
        f.duration_days = d
    if (t := triage.parse_trend(text)):
        f.trend = t
    if (v := triage.parse_severity(text, bare_ok=asked == "severity")) is not None:
        f.severity = v
    if (a := triage.parse_age(text, bare_ok=asked == "profile")) is not None:
        f.age = a
    # "I do not have any existing disease like lung disease or diabetes": the denial covers its own examples, so they are not conditions
    f.risks |= triage.parse_risks(_NO_CONDITIONS.sub(" ", text))
    if (tc := triage.parse_temperature_c(text, bare_ok=asked == "temp")) is not None:
        f.temp_c = tc if f.temp_c is None else max(f.temp_c, tc)
    al = triage.parse_allergies(text)
    f.allergies |= al
    # Regular medicines only count when they say they take them: "can I take ibuprofen?" is a question, not a medicine they are on.
    says_taking = asked == "meds" or (not text.rstrip().endswith("?") and re.search(
        r"\bi (?:take|am taking|use|am on)\b|\bi'?m (?:taking|on|using)\b|currently (?:on|taking)|prescribed|my (?:regular )?(?:medicines?|medications?|tablets?)", text, re.I))
    md = triage.parse_medicines(text) if says_taking else set()
    f.medicines |= md
    if asked == "meds" or al or md or re.search(r"allerg", text, re.I):
        f.allergies_asked = True                         # they have told us about allergies / medicines (even 'none')
    f.slots = clinic.merge(f.slots, clinic.extract(text))     # which symptoms they have / explicitly do not have


_NO_CONDITIONS = re.compile(r"\b(?:no|not|none|without|nil|don'?t)\b[^.;]{0,30}(?:disease|conditions?|illness(?:es)?|problems?|medical)[^.;]*", re.I)
NON_SYMPTOM_SLOTS = {"can_keep_fluids"}                  # 'yes' to these is good news, not a symptom to remember in the text


def _answer_slot(s: Session, asked: str | None, text: str) -> None:
    """A plain 'yes' / 'no' answers the question that was asked: file it under that question's symptom slot."""
    q = interview.BY_ID.get(asked) if asked else None
    if not q or not q.slot:
        return
    f = s.facts
    if not q.yn:                                         # either/or question: read the words of the answer
        for slot, value in interview.either_or(q.id, text).items():
            f.slots.setdefault(slot, value)
        return
    said = clinic.extract(text)
    if q.slot in said:                                   # the words themselves already told us
        return
    if clinic.is_yes(text):
        f.slots[q.slot] = True
        if q.slot not in NON_SYMPTOM_SLOTS:              # keep the specialist / medicines / model prompt in step with confirmed symptoms
            phrase = clinic.BY_SLOT[q.slot].phrase
            s.texts.append(phrase)
            s.symptom_texts.append(phrase)
            f.text = "\n".join(s.texts)
    elif clinic.is_no(text):
        f.slots[q.slot] = False


def strip_echo(text: str, bot_text: str, min_words: int = 5) -> str:
    """Remove the bot's own words from the patient's transcript.

    With a microphone near the speaker, the doctor's spoken question is transcribed together with the answer ("...heart or lung
    disease, or are you pregnant? 39 years, no existing disease"), and the echoed question would otherwise be read as symptoms or
    conditions the patient never mentioned. Any run of `min_words` or more consecutive words that also appears, in the same
    order, in the bot's last message is dropped. If nothing would be left, the text is returned unchanged."""
    def norm(w: str) -> str:
        return re.sub(r"[^a-z0-9']", "", w.lower())

    bot = [w for w in (norm(x) for x in bot_text.split()) if w]
    grams = {tuple(bot[i:i + min_words]) for i in range(len(bot) - min_words + 1)}
    words = text.split()
    nw = [norm(w) for w in words]
    keep = [True] * len(words)
    i = 0
    while i <= len(words) - min_words:
        if "" not in nw[i:i + min_words] and tuple(nw[i:i + min_words]) in grams:
            j = i + min_words
            while j < len(words) and "" != nw[j] and tuple(nw[j - min_words + 1:j + 1]) in grams:
                j += 1
            for k in range(i, j):
                keep[k] = False
            i = j
        else:
            i += 1
    cleaned = " ".join(w for w, k in zip(words, keep) if k).strip(" ,.;:-")
    return cleaned if re.search(r"[A-Za-z0-9]", cleaned) else text


def _fmt_days(d: float) -> str:
    if d < 1:
        return "a few hours"
    if d < 14:
        return f"{d:g} day" + ("" if d == 1 else "s")
    return f"about {round(d / 7)} weeks" if d < 60 else f"about {round(d / 30)} months"


# ---------- the three knowledge sources ----------
@dataclass
class Reply:
    text: str
    source: str = "none"                                 # model | dialogues | web | none
    sources: list[dict] = field(default_factory=list)
    trail: list[str] = field(default_factory=list)       # why earlier sources were skipped / not good enough
    confidence: float = 0.0
    flags: list[str] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    served_by: str = ""

    def as_dict(self) -> dict:
        return {"text": self.text, "source": self.source, "source_label": SOURCE_LABELS[self.source], "sources": self.sources,
                "trail": self.trail, "confidence": round(self.confidence, 2), "flags": self.flags,
                "timings_ms": self.timings_ms, "served_by": self.served_by}


def is_complex(a: Assessment, question: str) -> bool:
    """Complexity decides where to start: minor + simple question -> the model alone; anything else -> evidence first."""
    return a.complex or bool(_RISKY_Q.search(question)) or safety.is_dosing_query(question)


N_PASSAGES = 2


def _numbered_cases(cases: list[dialogues.Case]) -> str:
    return "\n".join(f"[{i}] {c.reply[:360]}" for i, c in enumerate(cases, 1))     # the doctor's words only: the patient's story is already in the facts line


def _numbered_pages(pages) -> str:
    return "\n".join(f"[{i}] ({p.source}) {p.text[:480]}" for i, p in enumerate(pages, 1))


def _note_failover(trail: list[str]) -> None:
    """Record any model fallback notes from medllm in the answer trail."""
    for n in medllm.last_notes:
        if n not in trail:
            trail.append(n)


_CHATTY = set("should would could really actually anything something everything normal serious dangerous harmful worried worry "
              "problem problems please thanks better worse again always never sometimes usually probably possible possibly".split())


def _covers_question(question: str, story: str, texts: list[str]) -> bool:
    """Does the evidence mention the *specific* thing asked about? A follow-up like 'is it safe to take antibiotics?' must not be
    'answered' from passages about the sore throat that never mention antibiotics (nor cost minutes of model time trying)."""
    if question == DEFAULT_QUESTION:
        return True
    known = {w[:5] for w in grounding.key_terms(story)}
    # New, content-heavy words only: short or conversational ones ('safe', 'okay', 'better') say nothing about the subject.
    rare = [w for w in grounding.key_terms(question) if w[:5] not in known and len(w) >= 6 and w not in _CHATTY]
    if not rare:
        return True
    blob = " ".join(texts).lower()
    return all(w[:5] in blob for w in rare)


def _best_excerpt(text: str, question: str, story: str, limit: int = 420) -> tuple[str, bool]:
    """The sentences of `text` that best match the question (not simply the opening paragraph); bool = anything matched."""
    terms = {w[:5] for w in grounding.key_terms(question + " " + story)}
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    scores = [len(terms & {w[:5] for w in re.findall(r"[a-z]{3,}", x.lower())}) for x in sents]
    best = max(range(len(sents)), key=lambda i: (scores[i], -i)) if sents else 0
    if not sents or scores[best] == 0:
        return _excerpt(text, limit), False
    out, i = sents[best], best + 1
    while i < len(sents) and len(out) + len(sents[i]) < limit:
        out += " " + sents[i]
        i += 1
    return _excerpt(out, limit), True


async def _try_model(question: str, facts: str, trail: list[str], timings: dict) -> Reply | None:
    """Source 1: the medical model alone. Accepted only when it is confident, free of doses, and does not itself send the
    patient to a doctor (our rules said self-care; a disagreement is escalated to evidence rather than averaged away)."""
    if not await medllm.is_available():
        trail.append("Medical model not reachable")
        return None
    t = time.perf_counter()
    try:
        out = await medllm.chat_json([{"role": "user", "content": MODEL_PROMPT.format(facts=facts, question=question)}])
    except LLMUnavailable as e:
        trail.append(f"Medical model failed ({e})")
        return None
    finally:
        timings["model_ms"] = int((time.perf_counter() - t) * 1000)
    _note_failover(trail)
    ans = str((out or {}).get("answer") or "").strip()
    conf = float((out or {}).get("confidence") or 0)
    if not _usable(ans):
        trail.append("Medical model gave no usable answer")
    elif _DOSE.search(ans):
        trail.append("Medical model's answer contained a drug dose, so it was not used")
    elif (out or {}).get("needs_doctor") in (True, "true", "True"):
        trail.append("Medical model suggested a doctor's check, so evidence from real cases was consulted")
    elif conf < 0.7:
        trail.append(f"Medical model was not confident enough ({conf:.2f})")
    else:
        return Reply(ans, "model", [{"title": "Hugging Face Medical AI", "source": "Model's own knowledge", "url": "", "snippet": ""}],
                     trail, conf, ["model_only"], timings, medllm.last_endpoint)
    return None


async def _try_dialogues(question: str, facts: str, query: str, patient_text: str, trail: list[str], timings: dict) -> Reply | None:
    """Source 2: doctor replies to patients who described the same thing."""
    s = get_settings()
    t = time.perf_counter()
    cases = await _in_thread(dialogues.search, query)
    timings["dialogue_search_ms"] = int((time.perf_counter() - t) * 1000)
    good = [c for c in cases if c.score >= s.dialogue_min_score][:N_PASSAGES]
    if not good:
        trail.append("No sufficiently similar doctor-patient conversation found" if cases or dialogues.count()
                     else "Doctor-patient conversation index is empty")
        return None
    if not _covers_question(question, patient_text, [c.reply for c in good]):
        trail.append("Similar doctor conversations exist but none mentions what you asked about, so the model was not asked to read them")
        return None
    sources = [{"title": c.title, "source": c.source, "url": c.url, "score": c.score, "snippet": _excerpt(c.reply, 260)} for c in good]
    if not await medllm.is_available():
        top = good[0]
        return Reply(_excerpt(top.reply, 500), "dialogues", sources, trail, top.score,
                     ["model_unavailable", "verbatim_doctor_reply"], timings)
    t = time.perf_counter()
    try:
        out = await medllm.chat_json([{"role": "user", "content": GROUNDED_PROMPT.format(
            facts=facts, question=question, label="DOCTORS' REPLIES TO SIMILAR PATIENTS", context=_numbered_cases(good))}])
    except LLMUnavailable as e:
        trail.append(f"Medical model failed while reading the cases ({e})")
        out = None
    timings["dialogue_llm_ms"] = int((time.perf_counter() - t) * 1000)
    if out is not None:
        _note_failover(trail)
    ans = str((out or {}).get("answer") or "").strip()
    supported = (out or {}).get("supported") in (True, "true", "True")
    if out is None:
        top = good[0]                                    # model down mid-request: show the doctor's words, not nothing
        return Reply(_excerpt(top.reply, 500), "dialogues", sources, trail, top.score, ["model_unavailable", "verbatim_doctor_reply"], timings)
    if supported and _usable(ans):
        bad = ungrounded_numbers(ans, patient_text, *(c.reply for c in good))
        if bad:                                          # a number the doctors never said (e.g. an invented dose): quote them instead
            return Reply(_excerpt(good[0].reply, 500), "dialogues", sources, trail, good[0].score,
                         ["verbatim_doctor_reply", "numeric_mismatch:" + ",".join(bad)], timings, medllm.last_endpoint)
        return Reply(ans, "dialogues", sources, trail, float((out or {}).get("confidence") or good[0].score), [], timings, medllm.last_endpoint)
    trail.append("Similar doctor conversations did not actually address this question")
    return None


async def _try_web(question: str, facts: str, query: str, patient_text: str, trail: list[str], timings: dict) -> Reply | None:
    """Source 3: reputable web pages; always returns the links it used."""
    t = time.perf_counter()
    pages = await websearch.search(query)
    timings["web_search_ms"] = int((time.perf_counter() - t) * 1000)
    if not pages:
        trail.append("Web search found nothing relevant")
        return None
    pages = [p for p in pages if _covers_question(question, patient_text, [p.text])][:N_PASSAGES]
    if not pages:
        trail.append("Web pages were found but none mentions what you asked about")
        return None
    sources = [{"title": p.title, "source": p.source, "url": p.url, "score": p.score, "snippet": _excerpt(p.text, 260)} for p in pages]
    flags = ["online_source"]
    ans = ""
    if await medllm.is_available():
        t = time.perf_counter()
        try:
            out = await medllm.chat_json([{"role": "user", "content": GROUNDED_PROMPT.format(
                facts=facts, question=question, label="WEB PAGES", context=_numbered_pages(pages))}])
        except LLMUnavailable:
            out = None
        timings["web_llm_ms"] = int((time.perf_counter() - t) * 1000)
        if out is not None:
            _note_failover(trail)
        cand = str((out or {}).get("answer") or "").strip()
        if (out or {}).get("supported") in (True, "true", "True") and _usable(cand):
            bad = ungrounded_numbers(cand, patient_text, *(p.text for p in pages))
            if bad:
                flags.append("numeric_mismatch:" + ",".join(bad))
            else:
                ans = cand
    if not ans:                                          # quote the most relevant sentences verbatim; the link is shown either way
        excerpt, matched = _best_excerpt(pages[0].text, question if question != DEFAULT_QUESTION else "", patient_text)
        if not matched and question != DEFAULT_QUESTION:
            trail.append("The best page did not contain a passage that answers the question")
            return None
        ans = f"{pages[0].source}: {excerpt}"
        flags.append("verbatim_excerpt")
    return Reply(ans, "web", sources, trail, min(pages[0].score, 0.9), flags, timings, medllm.last_endpoint)


async def _in_thread(fn, *a):
    import asyncio
    return await asyncio.to_thread(fn, *a)


async def answer(s: Session, question: str, a: Assessment) -> Reply:
    """Route one medical question through the three sources, escalating when a source is not good enough."""
    trail: list[str] = []
    timings: dict = {}
    facts = summary(s)
    patient_text = " ".join(s.texts)
    story = " ".join(s.texts)[:450]                      # everything the patient described, not just the first message
    query = story if question == DEFAULT_QUESTION else f"{story[:300]} {question}"

    if a.level == "emergency":
        return Reply(safety.EMERGENCY_MESSAGE, "none", [], ["Emergency warning sign detected: no model was consulted"], 1.0, ["emergency"])

    if not is_complex(a, question):
        if (r := await _try_model(question, facts, trail, timings)):
            return r
    else:
        trail.append("Complex or higher-risk case: started from evidence instead of the model's memory")
    if (r := await _try_dialogues(question, facts, query, patient_text, trail, timings)):
        return r
    # Only anonymised keywords may reach a search engine: names, phone numbers and e-mails are removed first.
    web_query = safety.to_search_query(f"{'' if question == DEFAULT_QUESTION else question} {s.texts[0][:300]}", max_terms=10)
    if (r := await _try_web(question, facts, web_query, patient_text, trail, timings)):
        return r
    return Reply("I could not find a reliable answer to this. Please speak to a doctor or pharmacist about it.", "none", [], trail, 0.0,
                 ["no_reliable_source"], timings)


# ---------- conversation ----------
def _level_sentence(a: Assessment) -> str:
    return f"{a.info['label']}. {a.info['window']}."


def _specialist_sentence(a: Assessment) -> str:
    if a.level == "emergency":
        return "Please go to the nearest emergency department."
    article = "an" if a.specialty[:1].upper() in "AEIOU" else "a"
    if a.level == "self_care":
        return f"If it does not settle, {article} {a.specialty} or general physician can help."
    return f"I suggest seeing {article} {a.specialty}."


def _rx_prefix(s: Session, question: str, a: Assessment) -> str:
    """Questions about prescription-only medicines (antibiotics, steroids...) get a clear 'I cannot prescribe that' before the facts."""
    if question and question != DEFAULT_QUESTION and (med := medicines.answer_about(question, s.facts, a)) and med["verdict"] == "rx":
        return med["text"] + " For information: "
    return ""


_THANKS = re.compile(r"^\s*(thanks?( you)?|thank you( so much)?|ok(ay)?|okay thanks|got it|great|bye|goodbye|that'?s all)\W*$", re.I)


async def handle(s: Session, text: str, with_answer: bool = True) -> dict:
    """One patient message in -> everything the UI needs out (the API layer adds doctors, audio and audit).

    with_answer=False returns the instant, rule-based part (severity, specialist) and the question to answer; the model
    answer is then fetched with respond() so urgency never waits for the LLM.
    """
    text = text.strip()
    last_bot = next((h["content"] for h in reversed(s.history) if h["role"] == "assistant"), "")
    text = strip_echo(text, last_bot)                    # a microphone that hears the speaker transcribes the question too
    s.history.append({"role": "user", "content": text})
    asked, s.asked = s.asked, None                       # the slot this message answers
    skipped = bool(_SKIP.match(text))
    nothing = bool(_NONE.match(text)) and asked not in (None, "complaint", "severity")
    if asked == "meds" and not skipped:
        s.facts.allergies_asked = True                   # 'none' is an answer; 'skip' is not: doses stay hidden until they tell us
    if s.stage == "advice" and _THANKS.match(text):
        msg = "You're welcome. Please do see a doctor if things get worse, and take care."
        s.history.append({"role": "assistant", "content": msg})
        return {"stage": "advice", "question": None, "message": msg, "progress": 1.0, "assessment": None, "answer": None}

    f = s.facts
    before_slots = dict(f.slots)
    before = (f.temp_c, f.duration_days, f.severity, f.age)
    if not skipped and not nothing:
        s.texts.append(text)
        if asked not in ("profile", "meds"):             # history / medicine answers must not decide the specialist
            s.symptom_texts.append(text)
        _learn(s, text, asked)
    if not skipped:
        _answer_slot(s, asked, text)
    flags = [] if skipped else safety.emergency_flags(triage.strip_negated(text, strict=True) + " I am")
    flags += [label for lvl, label in clinic.red_flags(f) if lvl == "emergency"]      # e.g. a plain 'yes' to "any chest pain?"
    if asked:
        s.done_slots.add(asked)

    # What did we just learn? Say so, like a doctor taking notes.
    extras = []
    if f.temp_c is not None and f.temp_c != before[0]:
        extras.append(f"fever of {triage.f_from_c(f.temp_c)}°F ({f.temp_c:.1f}°C)" if f.temp_c >= 38 else f"temperature {triage.f_from_c(f.temp_c)}°F")
    if f.duration_days is not None and f.duration_days != before[1]:
        extras.append(f"for {_fmt_days(f.duration_days)}")
    if f.severity is not None and f.severity != before[2] and (asked == "severity" or re.search(r"\d\s*(?:/|out of)\s*10", text)):   # only a number they actually gave
        extras.append(f"severity {f.severity}/10")
    if f.age is not None and f.age != before[3]:
        extras.append(f"age {f.age}")
    ack = clinic.acknowledge(before_slots, f.slots, extras) if not skipped else ""

    # severity must be a number: re-ask once rather than file prose as a score
    if asked == "severity" and s.facts.severity is None and not skipped and not flags and not s.retried_severity:
        s.retried_severity = True
        s.done_slots.discard("severity")
        s.asked = "severity"
        q = "Sorry, I need a number. From 0 (no problem) to 10 (worst imaginable), how bad is it right now?"
        s.history.append({"role": "assistant", "content": q})
        return {"stage": "interview", "question": q, "message": q, "why": interview.BASIC_WHY["severity"], "progress": progress(s),
                "assessment": None, "answer": None}

    if s.stage == "interview" and not flags:
        step = _next_step(s)
        if step:
            s.asked, q, why = step
            message = f"{ack} {q}".strip()
            s.history.append({"role": "assistant", "content": message})
            return {"stage": "interview", "question": q, "message": message, "why": why, "ack": ack, "progress": progress(s),
                    "assessment": None, "answer": None}

    # interview complete (or an emergency interrupted it): assess and answer
    first_time = s.stage == "interview"
    s.stage = "advice"
    s.facts.text = "\n".join(s.texts)
    s.facts.symptom_text = "\n".join(s.symptom_texts)       # the specialist follows the symptoms, not the medical history
    a = triage.assess(s.facts)
    previous = s.assessment
    s.assessment = a
    question = DEFAULT_QUESTION if first_time else text
    changed = bool(previous and previous.level != a.level)
    new_info = first_time or changed or not previous or previous.specialty != a.specialty

    # "Can I take ibuprofen?" / "what dose of paracetamol?" is answered at once from the medicine reference + this patient's
    # safety screen: deterministic, no model, no minutes of waiting.
    reply = None
    gave_safety_info = (not first_time and "?" not in text and bool(
        triage.parse_allergies(text) or (triage.parse_medicines(text) and re.search(
            r"\bi (?:take|am taking|use|am on)\b|\bi'?m (?:taking|on|using)\b|prescribed", text, re.I))))
    if gave_safety_info:
        reply = Reply("Thank you. I have updated your medicine safety check, so the suggestions on the right now take that into account.",
                      "formulary", [], ["Safety information recorded: no model needed"], 1.0, ["formulary", "safety_update"], {})
    elif not first_time and (med := medicines.answer_about(text, s.facts, a)) and med["verdict"] != "rx":
        reply = Reply(med["text"], "formulary", [], ["Answered from the medicine reference and your safety screen: no model needed"],
                      1.0, ["formulary", med["verdict"]], {})
    elif with_answer:
        reply = await answer(s, question, a)
        reply.text = _rx_prefix(s, question, a) + reply.text
    meds = medicines.suggest(s.facts, a)

    if a.level == "emergency":
        message = safety.EMERGENCY_MESSAGE
    else:
        parts = []
        if first_time or changed:
            parts.append(_level_sentence(a))
        if reply:
            parts.append(reply.text)
        if (first_time or changed) and meds["suggestions"]:
            parts.append("I have listed over-the-counter medicines that look suitable for you, with doses; this is not a prescription.")
        if new_info:
            parts.append(_specialist_sentence(a))                 # the doctor referral always comes last
        if (first_time or changed) and meds["suggestions"]:
            parts.append(meds["referral"])
        message = " ".join(p for p in parts if p)
    if message:
        s.history.append({"role": "assistant", "content": message})
    return {
        "stage": "advice", "question": None, "message": message, "progress": 1.0, "first_assessment": first_time,
        "pending_question": question if (not with_answer and reply is None and a.level != "emergency") else None,
        "medicines": meds,
        "assessment": {"level": a.level, "label": a.info["label"], "window": a.info["window"], "score": a.score, "reasons": a.reasons,
                       "specialty": a.specialty, "complex": a.complex,
                       "possible_causes": [{"name": d["name"], "score": d["score"], "because": d["because"], "note": d["note"],
                                            "likelihood": "most likely" if i == 0 and d["score"] >= 0.5 else "possible"}
                                           for i, d in enumerate(a.possible)],
                       "changed": bool(previous and previous.level != a.level)},
        "answer": reply.as_dict() if reply else None,
    }


async def respond(s: Session, question: str) -> dict:
    """The slow half: answer one question for an already-assessed session from the three sources."""
    if not s.assessment:
        raise ValueError("assessment not ready")
    reply = await answer(s, question or DEFAULT_QUESTION, s.assessment)
    reply.text = _rx_prefix(s, question or "", s.assessment) + reply.text
    s.history.append({"role": "assistant", "content": reply.text})
    save_session(s)
    return {"message": reply.text, "answer": reply.as_dict()}
