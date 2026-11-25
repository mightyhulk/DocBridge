"""Patient consult: parsing, severity rules, specialist choice, interview flow, 3-source routing, nearby doctors."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "backend"))
import pytest  # noqa: E402

from app import consult, dialogues, doctors, medllm, triage, websearch  # noqa: E402
from app.triage import Facts  # noqa: E402
from app.vectorstore import Passage  # noqa: E402


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- parsing
@pytest.mark.parametrize("text,days", [("3 days", 3), ("for two weeks", 14), ("since yesterday", 1), ("a few days", 3),
                                       ("about 6 hours", 0.25), ("for a month", 30), ("I feel odd", None)])
def test_duration(text, days):
    assert triage.parse_duration_days(text) == days


def test_severity_forms():
    assert triage.parse_severity("it is 6/10") == 6
    assert triage.parse_severity("about 7 out of 10") == 7
    assert triage.parse_severity("my pain is 8") == 8
    assert triage.parse_severity("it is severe") == 8
    assert triage.parse_severity("I have 3 dogs") is None                 # a stray number is not a severity
    assert triage.parse_severity("about 4", bare_ok=True) == 4
    assert triage.parse_severity("five", bare_ok=True) == 5


def test_age():
    assert triage.parse_age("I'm 34 years old") == 34
    assert triage.parse_age("my 5 year old son") == 5
    assert triage.parse_age("45") is None and triage.parse_age("45", bare_ok=True) == 45
    assert triage.parse_age("I have had this for 200 days") is None


def test_risks_and_negation():
    assert triage.parse_risks("I have diabetes and asthma") == {"diabetes", "lung disease"}
    assert triage.parse_risks("no diabetes, no heart problems") == set()
    assert "pregnancy" in triage.parse_risks("I am 12 weeks pregnant")


# ---------------------------------------------------------------- severity + specialist
def lv(text, **kw):
    return triage.assess(Facts(text=text, **kw))


def test_mild_cold_is_self_care():
    a = lv("runny nose and sneezing", duration_days=1, severity=2)
    assert a.level == "self_care" and not a.complex


def test_emergency_overrides_everything():
    a = lv("I have crushing chest pain and sweating", severity=2)
    assert a.level == "emergency" and a.specialty == "Cardiologist"


def test_warning_signs_are_urgent():
    assert lv("there is blood in my stool", duration_days=1, severity=3).level == "urgent"
    assert lv("fever 103 and a stiff neck", duration_days=1).level == "urgent"
    assert lv("mild headache", severity=9).level == "urgent"


def test_confusion_rule_is_specific():
    assert lv("I am confused about when to take my pills", severity=1).level == "self_care"
    assert lv("my father is suddenly confused and drowsy", severity=1).level == "urgent"


def test_duration_severity_and_age_escalate():
    assert lv("a cough", duration_days=1, severity=2).level == "self_care"
    assert lv("a cough", duration_days=20, severity=4, trend="worse").level == "see_doctor"
    base = lv("fever", duration_days=2, severity=4)
    older = lv("fever", duration_days=2, severity=4, age=78)
    assert triage.LEVELS.index(older.level) >= triage.LEVELS.index(base.level)
    assert older.score > base.score


def test_conditions_raise_risk_and_complexity():
    a = lv("sore throat", duration_days=4, severity=4, risks={"diabetes"})
    assert a.complex and a.score > lv("sore throat", duration_days=4, severity=4).score


@pytest.mark.parametrize("text,age,spec", [
    ("my knee hurts when I climb stairs", 40, "Orthopedist"), ("itchy red rash on my arms", 30, "Dermatologist"),
    ("burning when I pass urine", 29, "Urologist"), ("migraine with nausea", 35, "Neurologist"),
    ("my toddler has a fever", None, "Pediatrician"), ("fever", 6, "Pediatrician"), ("tired and weak", 30, "General Physician"),
    ("toothache and swollen gum", 25, "Dentist"), ("sore throat and blocked nose", 22, "ENT specialist")])
def test_specialist(text, age, spec):
    assert triage.pick_specialty(text, age)[0] == spec


def test_specialist_comes_from_the_symptoms_not_the_medical_history():
    f = Facts(text="fever and body ache for 6 days\nI'm 45 and I have diabetes", symptom_text="fever and body ache for 6 days",
              duration_days=6, severity=6, risks={"diabetes"})
    assert triage.assess(f).specialty == "General Physician"
    assert triage.assess(Facts(text="my blood sugar is always high", symptom_text="my blood sugar is always high")).specialty == "Endocrinologist"


# ---------------------------------------------------------------- negation (found by the live probe)
@pytest.mark.parametrize("text,strict,expected", [
    ("no rash and no stiff neck", False, ""),
    ("I have a fever, no rash", False, "I have a fever,"),
    ("no fever and blood in stool", False, "and blood in stool"),          # 'and' starts a new clause: a real symptom survives
    ("no fever or chills but a bad cough", False, "but a bad cough"),
    ("I do not have chest pain", True, "I"),
    ("without chest pain or fever", True, ""),
    ("he is not breathing", True, "he is not breathing"),                  # 'not' is not a denial cue in the emergency check
    ("I can't breathe", True, "I can't breathe"),
    ("no fever and cannot breathe", True, "and cannot breathe"),
    ("no, I have crushing chest pain", True, ", I have crushing chest pain"),
])
def test_strip_negated(text, strict, expected):
    assert triage.strip_negated(text, strict=strict) == expected


def test_denied_symptoms_do_not_trigger_warning_signs_or_the_wrong_specialist():
    a = lv("I have had fever and body ache for 6 days. temperature was 101, no rash and no stiff neck", duration_days=6, severity=6)
    assert not any("stiff neck" in r for r in a.reasons) and a.specialty == "General Physician"
    assert lv("a rash on my arm", severity=2).specialty == "Dermatologist"          # the same word, not denied, still counts
    assert lv("fever, no blood in stool, no black stools", severity=2).level == "self_care"
    assert lv("fever and blood in stool", severity=2).level == "urgent"


def test_denied_emergency_phrases_are_not_emergencies_but_real_ones_still_are():
    assert lv("cough for two days, I do not have chest pain and no shortness of breath", severity=2).level != "emergency"
    assert lv("no fever and cannot breathe").level == "emergency"
    assert lv("my grandfather is not breathing").level == "emergency"


def test_interview_answer_denying_red_flags_does_not_interrupt(monkeypatch):
    Stub(monkeypatch, model={"answer": "Rest and fluids.", "confidence": 0.9, "needs_doctor": False})
    s, _ = consult.start()
    talk(s, "I have a cough for 3 days, mild 2/10")
    r = talk(s, "no chest pain, no shortness of breath and no fever")[0]
    assert r["stage"] == "interview"                                   # denying red flags does not raise an alarm...
    assert s.facts.slots["chest_pain"] is False and s.facts.slots["breath_short"] is False and s.facts.slots["fever"] is False
    r, asked = finish(s, r)
    assert not {"chest_pain", "breath_short", "fever_check", "temp"} & set(asked)      # ...and none of those is asked again
    assert r["assessment"]["level"] == "self_care"


# ---------------------------------------------------------------- stubs for the three sources
class Stub:
    """Replaces the model, the dialogue index and the web search; records which sources were consulted."""

    def __init__(self, monkeypatch, *, model_up=True, model=None, cases=(), grounded=None, pages=()):
        self.calls = []
        self.model_up, self.model, self.cases, self.grounded, self.pages = model_up, model, list(cases), grounded, list(pages)

        async def avail():
            return self.model_up

        async def chat_json(messages, **kw):
            prompt = messages[0]["content"]
            if "WEB PAGES" in prompt:
                self.calls.append("web_llm")
                return self.grounded
            if "DOCTORS' REPLIES" in prompt:
                self.calls.append("dialogue_llm")
                return self.grounded
            self.calls.append("model")
            return self.model

        async def web(q):
            self.calls.append("web_search")
            return self.pages

        monkeypatch.setattr(medllm, "is_available", avail)
        monkeypatch.setattr(medllm, "chat_json", chat_json)
        monkeypatch.setattr(medllm, "last_endpoint", "huggingface:google/gemma-4-31B-it", raising=False)
        monkeypatch.setattr(dialogues, "search", lambda q, k=None: (self.calls.append("dialogue_search"), self.cases)[1])
        monkeypatch.setattr(dialogues, "count", lambda: 100)
        monkeypatch.setattr(websearch, "search", web)


def case(score=0.8, reply="Rest and drink fluids. See a doctor if the fever lasts beyond three days."):
    return dialogues.Case(patient="I have fever and sore throat", reply=reply, title="fever", source="ChatDoctor-HealthCareMagic-100k",
                          url="https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k", score=score)


PAGE = Passage(text="A sore throat is usually caused by a virus and gets better within a week.", source="NHS", title="Sore throat",
               score=0.75, url="https://www.nhs.uk/conditions/sore-throat/")
MINOR = triage.Assessment("self_care", 0, ["mild"], "General Physician", False)
COMPLEX = triage.Assessment("see_doctor", 4, ["lasting 10 days"], "General Physician", True)


def session_with(text="I have a runny nose and a sore throat"):
    s, _ = consult.start()
    s.texts = [text]
    s.facts.text = text
    return s


# ---------------------------------------------------------------- routing across the 3 sources
def test_minor_case_is_answered_by_the_model_alone(monkeypatch):
    st = Stub(monkeypatch, model={"answer": "It is probably a cold. Rest and drink fluids.", "confidence": 0.85, "needs_doctor": False})
    r = run(consult.answer(session_with(), consult.DEFAULT_QUESTION, MINOR))
    assert r.source == "model" and st.calls == ["model"] and r.served_by == "huggingface:google/gemma-4-31B-it"


def test_unconfident_model_escalates_to_doctor_conversations(monkeypatch):
    st = Stub(monkeypatch, model={"answer": "Maybe a cold.", "confidence": 0.4, "needs_doctor": False}, cases=[case()],
              grounded={"supported": True, "answer": "Doctors advised rest and fluids and a visit if fever lasts beyond three days [1].", "confidence": 0.8})
    r = run(consult.answer(session_with(), consult.DEFAULT_QUESTION, MINOR))
    assert r.source == "dialogues" and st.calls == ["model", "dialogue_search", "dialogue_llm"]
    assert any("not confident" in t for t in r.trail)
    assert r.sources[0]["url"].startswith("https://huggingface.co/datasets/")


def test_complex_case_skips_the_model_and_starts_from_evidence(monkeypatch):
    st = Stub(monkeypatch, cases=[case()],
              grounded={"supported": True, "answer": "Doctors advised rest and fluids [1].", "confidence": 0.8})
    r = run(consult.answer(session_with(), consult.DEFAULT_QUESTION, COMPLEX))
    assert r.source == "dialogues" and "model" not in st.calls


def test_risky_question_is_complex_even_when_the_case_is_minor(monkeypatch):
    st = Stub(monkeypatch, cases=[case(reply="Some antibiotics are safe in pregnancy but others are not, so ask your doctor before taking any. "
                                              "Rest and fluids help a viral infection and need no medicine at all.")],
              grounded={"supported": True, "answer": "Ask your doctor before taking antibiotics in pregnancy [1].", "confidence": 0.8})
    r = run(consult.answer(session_with(), "Is it safe to take antibiotics while pregnant?", MINOR))
    assert "model" not in st.calls and r.source == "dialogues"


def test_no_similar_case_falls_through_to_the_web_with_links(monkeypatch):
    st = Stub(monkeypatch, cases=[case(score=0.2)], pages=[PAGE],
              grounded={"supported": True, "answer": "A sore throat is usually viral and improves within a week [1].", "confidence": 0.8})
    r = run(consult.answer(session_with(), consult.DEFAULT_QUESTION, COMPLEX))
    assert r.source == "web" and r.sources[0]["url"] == PAGE.url
    assert st.calls == ["dialogue_search", "web_search", "web_llm"]
    assert any("No sufficiently similar" in t for t in r.trail)


def test_web_query_never_contains_names_phone_numbers_or_emails(monkeypatch):
    seen = []
    Stub(monkeypatch, cases=[], pages=[])

    async def spy(q):
        seen.append(q)
        return []
    monkeypatch.setattr(websearch, "search", spy)
    s = session_with("My name is Rahul Sharma, call 9876543210 or mail rahul@example.com. I have had a fever and sore throat for two days")
    run(consult.answer(s, consult.DEFAULT_QUESTION, COMPLEX))
    q = seen[0].lower()
    assert "fever" in q and "sore" in q
    assert not any(x in q for x in ("rahul", "sharma", "9876543210", "example.com", "@"))


def test_followup_mentioning_something_the_evidence_never_does_skips_the_slow_model_call(monkeypatch):
    st = Stub(monkeypatch, cases=[case(reply="Sore throats are usually viral. Gargle with warm salt water and drink fluids and rest your voice at home.")],
              pages=[Passage(text="The common cold is a mild infection of the upper respiratory tract with sneezing and a runny nose.", source="MedlinePlus (NIH)",
                             title="Common cold", score=0.7, url="https://medlineplus.gov/commoncold.html")])
    r = run(consult.answer(session_with("I have a sore throat for 2 days"), "Is it safe to take antibiotics for a sore throat?", COMPLEX))
    assert r.source == "none" and "no_reliable_source" in r.flags
    assert "dialogue_llm" not in st.calls and "web_llm" not in st.calls                 # no minutes of Pi time spent on irrelevant text
    assert any("none mentions what you asked" in t for t in r.trail) and any("Web pages were found but none" in t for t in r.trail)


def test_followup_is_answered_when_the_evidence_does_mention_it(monkeypatch):
    page = Passage(text="Most sore throats are caused by viruses, so antibiotics will not help and are not needed. Antibiotics are only used for strep throat.",
                   source="NHS", title="Sore throat", score=0.75, url="https://www.nhs.uk/conditions/sore-throat/")
    st = Stub(monkeypatch, cases=[], pages=[page],
              grounded={"supported": True, "answer": "Antibiotics usually do not help a viral sore throat [1].", "confidence": 0.8})
    r = run(consult.answer(session_with("I have a sore throat for 2 days"), "Is it safe to take antibiotics for a sore throat?", COMPLEX))
    assert r.source == "web" and r.sources[0]["url"] == page.url and "antibiotics" in r.text.lower()


def test_quoted_excerpt_is_the_relevant_sentences_not_the_first_paragraph():
    text = "Cookie notice and navigation text here for readers. The history of this site began long ago in a town. "            "Antibiotics do not work on viral sore throats and should only be used for strep throat. A doctor can test for strep."
    out, matched = consult._best_excerpt(text, "Do antibiotics work for sore throat?", "sore throat")
    assert matched and out.startswith("Antibiotics do not work")
    assert consult._best_excerpt("Nothing relevant at all in this text about gardening and tools.", "antibiotics?", "")[1] is False


def test_dialogues_that_do_not_address_the_question_fall_through(monkeypatch):
    st = Stub(monkeypatch, cases=[case()], pages=[PAGE], grounded={"supported": False, "answer": "I could not find this", "confidence": 0.1})
    r = run(consult.answer(session_with(), consult.DEFAULT_QUESTION, COMPLEX))
    assert r.source == "web"
    assert any("did not actually address" in t for t in r.trail)


def test_nothing_anywhere_is_an_honest_no(monkeypatch):
    Stub(monkeypatch, cases=[], pages=[], model={"answer": "x", "confidence": 0.1})
    r = run(consult.answer(session_with(), consult.DEFAULT_QUESTION, COMPLEX))
    assert r.source == "none" and "doctor" in r.text.lower() and "no_reliable_source" in r.flags


def test_model_dose_is_never_shown(monkeypatch):
    st = Stub(monkeypatch, model={"answer": "Take paracetamol 500 mg every 4 hours.", "confidence": 0.95, "needs_doctor": False}, cases=[])
    r = run(consult.answer(session_with(), consult.DEFAULT_QUESTION, MINOR))
    assert r.source != "model" and any("dose" in t for t in r.trail)


def test_model_saying_see_a_doctor_overrides_our_minor_rating(monkeypatch):
    Stub(monkeypatch, model={"answer": "Please get this checked.", "confidence": 0.95, "needs_doctor": True}, cases=[])
    r = run(consult.answer(session_with(), consult.DEFAULT_QUESTION, MINOR))
    assert r.source != "model"


def test_invented_number_in_grounded_answer_is_replaced_by_the_doctors_words(monkeypatch):
    Stub(monkeypatch, cases=[case()], grounded={"supported": True, "answer": "Take 650 mg three times a day [1].", "confidence": 0.9})
    r = run(consult.answer(session_with(), consult.DEFAULT_QUESTION, COMPLEX))
    assert r.source == "dialogues" and "verbatim_doctor_reply" in r.flags and "650" not in r.text


def test_model_offline_still_answers_from_sources_verbatim(monkeypatch):
    Stub(monkeypatch, model_up=False, cases=[case()])
    r = run(consult.answer(session_with(), consult.DEFAULT_QUESTION, COMPLEX))
    assert r.source == "dialogues" and "model_unavailable" in r.flags and "Rest and drink fluids" in r.text
    Stub(monkeypatch, model_up=False, cases=[], pages=[PAGE])
    r = run(consult.answer(session_with(), consult.DEFAULT_QUESTION, COMPLEX))
    assert r.source == "web" and "verbatim_excerpt" in r.flags and r.sources[0]["url"] == PAGE.url


def test_model_fallback_note_is_reported_in_the_trail(monkeypatch):
    Stub(monkeypatch, cases=[case()], grounded={"supported": True, "answer": "Doctors advised rest [1].", "confidence": 0.8})
    monkeypatch.setattr(medllm, "last_notes", ["Hugging Face model custom-med unavailable; used meta-llama/Llama-3.3-70B-Instruct instead"], raising=False)
    monkeypatch.setattr(medllm, "last_endpoint", "huggingface:meta-llama/Llama-3.3-70B-Instruct", raising=False)
    r = run(consult.answer(session_with(), consult.DEFAULT_QUESTION, COMPLEX))
    assert any("custom-med unavailable" in t and "meta-llama/Llama-3.3-70B-Instruct" in t for t in r.trail)


def test_dialogue_links_point_to_the_exact_dataset_row():
    assert dialogues.row_url("cd-1234") == "https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k/viewer/default/train?row=1234"
    assert dialogues.row_url("amc-7").endswith("ai-medical-chatbot/viewer/default/train?row=7")
    assert dialogues.row_url("weird", "fallback") == "fallback"


def test_emergency_consults_no_source(monkeypatch):
    st = Stub(monkeypatch)
    r = run(consult.answer(session_with(), consult.DEFAULT_QUESTION, triage.Assessment("emergency", 99, ["chest pain"], "Cardiologist", True)))
    assert "emergency" in r.flags and st.calls == []


# ---------------------------------------------------------------- interview flow
def talk(s, *msgs):
    return [run(consult.handle(s, m)) for m in msgs]


REPLIES = {"profile": "I'm 30, healthy", "meds": "none", "severity": "4", "duration": "2 days"}


def finish(s, r, answers=None, with_answer=True, limit=30):
    """Keep answering whatever the doctor asks (by question id, default 'no') until the assessment arrives. -> (result, [asked ids])."""
    answers = {**REPLIES, **(answers or {})}
    asked = []
    for _ in range(limit):
        if r["stage"] == "advice":
            return r, asked
        asked.append(s.asked)
        r = run(consult.handle(s, answers.get(s.asked, answers.get("*", "no")), with_answer=with_answer))
    raise AssertionError(f"never reached the assessment; asked {asked}")


def converse(s, first, answers=None, with_answer=True):
    return finish(s, run(consult.handle(s, first, with_answer=with_answer)), answers, with_answer)


def test_interview_skips_what_the_patient_already_said(monkeypatch):
    Stub(monkeypatch, model={"answer": "Rest and fluids.", "confidence": 0.9, "needs_doctor": False})
    s, first = consult.start()
    assert "describe the problem" in first
    r, asked = converse(s, "I have had a headache for 3 days and the pain is about 4/10")
    assert "duration" not in asked and "severity" not in asked                         # already said: never asked
    assert any(q in asked for q in ("worst_headache", "neuro_deficit", "head_side"))   # headache-specific questions instead
    assert asked.index("meds") == len(asked) - 1                                       # medicine safety check last, before any suggestion
    assert r["assessment"]["level"] == "self_care" and r["assessment"]["specialty"] == "Neurologist" and r["answer"]["source"] == "model"


def test_interview_asks_each_missing_basic_in_order(monkeypatch):
    Stub(monkeypatch)
    s, _ = consult.start()
    q = [talk(s, "my stomach hurts")[0]["question"]]
    q.append(talk(s, "since yesterday, getting worse")[0]["question"])
    assert "0 (no problem)" in q[0] or "how long" in q[0].lower()
    assert "0 (no problem)" in q[1] or "vomiting" in q[1].lower()


def test_specialist_sentence_uses_the_right_article():
    assert consult._specialist_sentence(triage.Assessment("see_doctor", 4, [], "ENT specialist", True)) == "I suggest seeing an ENT specialist."
    assert "a Dermatologist" in consult._specialist_sentence(triage.Assessment("urgent", 6, [], "Dermatologist", True))
    assert "an Orthopedist" in consult._specialist_sentence(triage.Assessment("self_care", 0, [], "Orthopedist", False))


def test_deferred_answer_returns_the_assessment_first_then_the_model_answer(monkeypatch):
    st = Stub(monkeypatch, model={"answer": "Rest and fluids.", "confidence": 0.9, "needs_doctor": False})
    s, _ = consult.start()
    r, _ = converse(s, "runny nose for 1 day, 2/10", with_answer=False)
    assert r["stage"] == "advice" and r["answer"] is None and r["pending_question"] == consult.DEFAULT_QUESTION
    assert r["assessment"]["level"] == "self_care" and "Minor" in r["message"] and st.calls == []        # nothing slow has run yet
    out = run(consult.respond(s, r["pending_question"]))
    assert out["answer"]["source"] == "model" and "Rest and fluids" in out["message"] and st.calls == ["model"]


def test_deferred_emergency_has_nothing_pending(monkeypatch):
    st = Stub(monkeypatch)
    s, _ = consult.start()
    r = run(consult.handle(s, "I have crushing chest pain", with_answer=False))
    assert r["assessment"]["level"] == "emergency" and r["pending_question"] is None and st.calls == []


def test_history_answer_does_not_change_the_specialist_in_a_full_interview(monkeypatch):
    Stub(monkeypatch, cases=[case()], grounded={"supported": True, "answer": "Doctors advised rest [1].", "confidence": 0.8})
    s, _ = consult.start()
    r, _ = converse(s, "I have had fever and body ache for 6 days, getting worse, 6/10",
                    {"temp": "temperature was 101, no rash and no stiff neck", "profile": "I'm 45 and I have diabetes"})
    assert r["assessment"]["specialty"] == "General Physician" and r["assessment"]["level"] in ("see_doctor", "urgent")
    assert not any("stiff neck" in x for x in r["assessment"]["reasons"])


def test_emergency_in_the_first_message_skips_the_interview(monkeypatch):
    st = Stub(monkeypatch)
    s, _ = consult.start()
    r = talk(s, "I have crushing chest pain and my left arm is numb")[0]
    assert r["stage"] == "advice" and r["assessment"]["level"] == "emergency" and st.calls == []
    assert "emergency" in r["message"].lower()


def test_emergency_mid_interview_interrupts(monkeypatch):
    Stub(monkeypatch)
    s, _ = consult.start()
    talk(s, "I feel unwell")
    r = talk(s, "actually now I cannot breathe")[0]
    assert r["assessment"]["level"] == "emergency"


def test_severity_reasked_once_then_moves_on(monkeypatch):
    Stub(monkeypatch)
    s, _ = consult.start()
    talk(s, "my back hurts for 2 weeks")
    r = talk(s, "quite a lot")[0]                          # 'a lot' is not a number
    assert "I need a number" in r["question"]
    r = talk(s, "no idea")[0]
    assert "need a number" not in r["question"]            # asked only once


def test_skip_keeps_the_interview_moving(monkeypatch):
    Stub(monkeypatch)
    s, _ = consult.start()
    talk(s, "cough")
    r = talk(s, "skip")[0]
    assert r["stage"] == "interview" and r["question"]


def test_follow_up_questions_after_assessment_use_the_router_and_can_escalate_severity(monkeypatch):
    st = Stub(monkeypatch, model={"answer": "Rest and fluids.", "confidence": 0.9, "needs_doctor": False}, cases=[case()],
              grounded={"supported": True, "answer": "Doctors advised rest [1].", "confidence": 0.8})
    s, _ = consult.start()
    converse(s, "runny nose for 1 day, 2/10")
    assert s.stage == "advice" and s.assessment.level == "self_care"
    r = talk(s, "Is it safe to take antibiotics?")[0]
    assert r["answer"]["source"] == "dialogues" and r["assessment"]["level"] == "self_care"
    assert "prescription-only" in r["message"] and "cannot suggest or prescribe" in r["message"]          # honest about not prescribing
    assert "Doctors advised rest" in r["message"]                                                        # ...and still informs
    r = talk(s, "now I am coughing up blood")[0]
    assert r["assessment"]["level"] == "urgent" and r["assessment"]["changed"]


def test_thanks_is_not_sent_to_a_model(monkeypatch):
    st = Stub(monkeypatch, model={"answer": "Rest and fluids.", "confidence": 0.9, "needs_doctor": False})
    s, _ = consult.start()
    converse(s, "runny nose for 1 day, 2/10")
    n = len(st.calls)
    r = talk(s, "thanks")[0]
    assert len(st.calls) == n and r["answer"] is None


# ---------------------------------------------------------------- nearby doctors
OVERPASS = {"elements": [
    {"type": "node", "lat": 20.30, "lon": 85.82, "tags": {"name": "City Heart Clinic", "amenity": "clinic", "healthcare:speciality": "cardiology", "phone": "+91 1"}},
    {"type": "way", "center": {"lat": 20.31, "lon": 85.83}, "tags": {"name": "Capital Hospital", "amenity": "hospital", "emergency": "yes"}},
    {"type": "node", "lat": 20.29, "lon": 85.81, "tags": {"name": "Smile Dental", "amenity": "dentist"}},
    {"type": "node", "lat": 20.30, "lon": 85.80, "tags": {"name": "Dr Rao", "amenity": "doctors"}},
    {"type": "node", "lat": 20.30, "lon": 85.80, "tags": {"amenity": "clinic"}},                       # unnamed: dropped
    {"type": "node", "lat": 20.30, "lon": 85.82, "tags": {"name": "City Heart Clinic", "amenity": "clinic"}},   # duplicate: dropped
]}


def test_parse_and_rank_specialist():
    fac = doctors.parse_elements(OVERPASS, 20.3, 85.8)
    assert {f["name"] for f in fac} == {"City Heart Clinic", "Capital Hospital", "Smile Dental", "Dr Rao"}
    ranked = doctors.rank(fac, ["cardio"])
    assert ranked[0]["name"] == "City Heart Clinic" and ranked[0]["match"] == "verified"
    assert ranked[1]["name"] == "Capital Hospital" and ranked[1]["match"] == "hospital"
    assert "Smile Dental" not in [r["name"] for r in ranked]                  # no dentist for a heart complaint
    assert all(r["directions_url"].startswith("https://www.google.com/maps/dir/") for r in ranked)


def test_rank_dentist_and_general_and_emergency():
    fac = doctors.parse_elements(OVERPASS, 20.3, 85.8)
    assert doctors.rank(fac, ["dent"])[0]["name"] == "Smile Dental"
    gp = doctors.rank(fac, ["general", "internal"])
    assert gp[0]["match"] == "verified" and gp[0]["kind"] in ("doctors", "clinic")
    em = doctors.rank(fac, ["cardio"], emergency=True)
    assert [r["name"] for r in em] == ["Capital Hospital"]


def _fac(name, kind="hospital", spec="", dist=1.0):
    return {"name": name, "kind": kind, "speciality": spec, "emergency": False, "distance_km": dist, "address": "", "phone": "", "website": "",
            "opening_hours": "", "lat": 0, "lon": 0, "map_url": "", "directions_url": ""}


def test_alternative_medicine_and_veterinary_are_never_offered():
    fac = [_fac("Utkal Ayurveda Pratisthan", dist=0.1), _fac("Government Ayurvedic Hospital", dist=0.2), _fac("Homoeopathic Clinic", "clinic", dist=0.1),
           _fac("Pet Care Animal Hospital", dist=0.1), _fac("City General Hospital", dist=3.0)]
    assert [r["name"] for r in doctors.rank(fac, ["neuro"])] == ["City General Hospital"]
    assert [r["name"] for r in doctors.rank(fac, [], emergency=True)] == ["City General Hospital"]


def test_facility_name_hints_rank_specialist_hospitals_sensibly():
    fac = [_fac("Lotus Eye Hospital", dist=0.1), _fac("City General Hospital", dist=2.0), _fac("Heart & Vascular Institute", dist=5.0),
           _fac("Bright Maternity Home", dist=0.2)]
    got = {r["name"]: r["match"] for r in doctors.rank(fac, ["cardio"])}
    assert got["Heart & Vascular Institute"] == "verified"            # the name says cardiology
    assert got["City General Hospital"] == "hospital"                 # a general hospital probably has the department
    assert got["Lotus Eye Hospital"] == "unverified" and got["Bright Maternity Home"] == "unverified"
    assert doctors.rank([_fac("Shishu Bhavan Hospital")], ["neuro"])[0]["match"] == "unverified"       # 'shishu' = child (Hindi)
    order = [r["name"] for r in doctors.rank(fac, ["cardio"])]
    assert order[0] == "Heart & Vascular Institute" and order[1] == "City General Hospital"
    # an eye hospital is not an emergency department
    assert "Lotus Eye Hospital" not in [r["name"] for r in doctors.rank(fac, [], emergency=True)]


def test_distance_and_privacy_rounding():
    assert doctors.haversine_km(20.3, 85.8, 20.3, 85.8) == 0
    assert 110 < doctors.haversine_km(0, 0, 1, 0) < 112
    assert doctors.round_coord(20.296058) == 20.3 and doctors.round_coord(85.824) == 85.82
    assert "around:5000,20.3,85.82" in doctors.build_query(20.3, 85.82, 5000)


def test_nearby_expands_radius_and_reports_overpass_failure(monkeypatch):
    calls = []

    async def fake(client, query):
        calls.append(query)
        return {"elements": []} if len(calls) < 3 else OVERPASS
    monkeypatch.setattr(doctors, "_overpass", fake)
    doctors._cache.clear()
    out = run(doctors.nearby(20.3, 85.8, ["cardio"]))
    assert len(calls) == 3 and out["results"] and out["radius_km"] == 40
    n = len(calls)
    run(doctors.nearby(20.3, 85.8, [], emergency=True))      # a second lookup for the same place reuses the cached facilities
    assert len(calls) == n

    async def boom(client, query):
        raise RuntimeError("Overpass unavailable")

    async def no_fallback(client, lat, lon, radius):
        raise ConnectionError("offline")
    monkeypatch.setattr(doctors, "_overpass", boom)
    monkeypatch.setattr(doctors, "_nominatim_facilities", no_fallback)
    doctors._cache.clear()
    out = run(doctors.nearby(11.1, 77.3, ["cardio"]))
    assert out["results"] == [] and "Overpass" in out["error"]


def test_nominatim_fallback_when_overpass_is_down(monkeypatch):
    rows = [{"lat": "20.30", "lon": "85.82", "name": "Apollo Hospital", "type": "hospital", "extratags": {"phone": "+91 123", "healthcare:speciality": "cardiology"},
             "address": {"road": "Plot 251", "city": "Bhubaneswar", "postcode": "751005"}},
            {"lat": "20.29", "lon": "85.83", "name": "Smile Dental", "type": "dentist", "address": {}},
            {"lat": "20.28", "lon": "85.80", "name": "A bus stop", "type": "bus_stop", "address": {}},        # not a facility: dropped
            {"lat": "20.28", "lon": "85.80", "type": "clinic", "address": {}}]                                  # unnamed: dropped
    fac = doctors.parse_nominatim(rows, 20.3, 85.8)
    assert {f["name"] for f in fac} == {"Apollo Hospital", "Smile Dental"}
    apollo = next(f for f in fac if f["name"] == "Apollo Hospital")
    assert apollo["phone"] == "+91 123" and apollo["speciality"] == "cardiology" and "Bhubaneswar" in apollo["address"]

    async def down(client, query):
        raise RuntimeError("Overpass unavailable (all mirrors 504)")

    async def nominatim(client, lat, lon, radius):
        return fac
    monkeypatch.setattr(doctors, "_overpass", down)
    monkeypatch.setattr(doctors, "_nominatim_facilities", nominatim)
    doctors._cache.clear()
    out = run(doctors.nearby(20.3, 85.8, ["cardio"]))
    assert out["results"] and out["results"][0]["name"] == "Apollo Hospital" and out["results"][0]["match"] == "verified" and out["error"] is None

    async def nominatim_down(client, lat, lon, radius):
        raise ConnectionError("no route")
    monkeypatch.setattr(doctors, "_nominatim_facilities", nominatim_down)
    doctors._cache.clear()
    out = run(doctors.nearby(20.3, 85.8, ["cardio"]))
    assert out["results"] == [] and "Overpass unavailable" in out["error"] and "Nominatim fallback failed" in out["error"]


# ---------------------------------------------------------------- dataset cleaning + web helpers
def test_dialogue_cleaning():
    raw = "Hello, Welcome to Chat Doctor. Thanks for your query. You likely have a tension headache from stress and poor sleep. " \
          "Try regular meals, hydration and rest, and avoid screens before bed. Take care. Regards. Dr. Sharma"
    out = dialogues.clean_doctor(raw)
    assert out.startswith("You likely have") and "Regards" not in out and "Sharma" not in out and "Welcome" not in out
    good = "You likely have a tension headache from stress and poor sleep and this is common. " * 3
    assert dialogues.usable_row("I have had a headache for three days now doctor", good)
    assert not dialogues.usable_row("I have had a headache for three days now doctor", good + " Call me on 9876543210 or WhatsApp")
    assert not dialogues.usable_row("short", good)


def test_web_helpers():
    assert websearch.trusted_name("https://www.mayoclinic.org/diseases-conditions/cold") == "Mayo Clinic"
    assert websearch.trusted_name("https://my.clevelandclinic.org/health/x") == "Cleveland Clinic"
    assert websearch.trusted_name("https://random-blog.example/post") is None
    text = "Cookie banner accept all.\n\nA sore throat is pain, scratchiness or irritation of the throat that often worsens when you swallow.\n\nShop our vitamins and supplements today for great deals online."
    got = websearch.relevant_paragraphs(text, "sore throat")
    assert "sore throat" in got.lower() and "Shop our vitamins" not in got
