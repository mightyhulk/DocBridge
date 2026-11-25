"""Adaptive interview, temperature / allergy / medicine parsing, and the OTC medicine screen."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "backend"))
import pytest  # noqa: E402

from app import consult, interview, medicines, triage  # noqa: E402
from app.triage import Facts  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def facts(text, **kw):
    return Facts(text=text, symptom_text=text, **kw)


def assess(f):
    return triage.assess(f)


# ---------------------------------------------------------------- parsing
@pytest.mark.parametrize("text,c", [
    ("I have a fever of 102 degree fahrenheit", 38.9), ("temperature was 101", 38.3), ("my temp is 38.5 C", 38.5),
    ("fever 39 celsius", 39.0), ("it was 103.2 farheight", 39.6), ("fever 99", 37.2), ("highest reading 104 F", 40.0),
])
def test_temperature_forms(text, c):
    assert triage.parse_temperature_c(text) == pytest.approx(c, abs=0.1)


@pytest.mark.parametrize("text", ["I am 39 years old and have a cold", "pain 8/10 for 3 days", "I take 500 mg twice a day", "my pulse is 100",
                                  "I have 102 emails", "fever for 38 days"])
def test_numbers_that_are_not_temperatures(text):
    assert triage.parse_temperature_c(text) is None


def test_age_is_never_a_duration():
    """Screenshot bug: 'I am 39 years old' was filed as an illness lasting 39 years (14235 days)."""
    assert triage.parse_duration_days("I am 39 years old, I do not have any pre-existing disease.") is None
    assert triage.parse_duration_days("I'm 45 years old and it has been 2 days") == 2
    assert triage.parse_duration_days("for 3 years I have had this") == 1095


def test_allergies_and_regular_medicines():
    assert triage.parse_allergies("I am allergic to penicillin and aspirin") == {"penicillin", "nsaid"}
    assert triage.parse_allergies("no known allergies") == set()
    assert triage.parse_allergies("allergic to many medicines") == {"many"}
    assert triage.parse_medicines("I take warfarin and amlodipine") == {"anticoagulant"}
    assert triage.parse_medicines("I am on losartan and sertraline") == {"bp_kidney", "antidepressant"}
    assert "aspirin" not in triage.parse_medicines("I am allergic to aspirin")          # an allergy is not a medicine they take


def test_high_fever_is_urgent_moderate_fever_is_not():
    assert assess(facts("fever of 104 F for a day", severity=3)).level == "urgent"
    a = assess(facts("fever of 102 degree fahrenheit and a blocked nose", duration_days=2, severity=3))
    assert a.level == "self_care" and any("Fever: 102" in r for r in a.reasons)
    assert assess(facts("fever 101", age=1)).level == "urgent"                            # a baby with a fever


# ---------------------------------------------------------------- medicine screen
def asm(text, level="self_care", **kw):
    f = facts(text, allergies_asked=kw.pop("asked", True), **kw)
    return f, triage.Assessment(level, 1, [], "General Physician", False)


def names(out):
    return [s["key"] for s in out["suggestions"]]


def test_simple_fever_and_body_ache_suggests_paracetamol_with_label_dose():
    f, a = asm("fever and body ache", duration_days=1, age=30)
    out = medicines.suggest(f, a)
    assert "paracetamol" in names(out)
    p = next(s for s in out["suggestions"] if s["key"] == "paracetamol")
    assert "500 mg to 1 g" in p["dose"] and "4 g" in p["maximum"]
    assert "not a prescription" in out["disclaimer"].lower() and out["not_prescribed"]


def test_ibuprofen_is_screened_out_with_reasons():
    cases = [
        ({"age": 70}, "65"), ({"risks": {"stomach ulcer or gastritis"}}, "stomach"), ({"risks": {"kidney disease"}}, "kidneys"),
        ({"risks": {"pregnancy"}}, "pregnancy"), ({"risks": {"lung disease"}}, "asthma"), ({"medicines": {"anticoagulant"}}, "blood thinner"),
        ({"allergies": {"nsaid"}}, "allergic"), ({"age": 30, "risks": {"heart disease"}}, "blood pressure"),
    ]
    for kw, word in cases:
        f, a = asm("headache and body pain", duration_days=1, **{"age": 40, **kw})
        out = medicines.suggest(f, a)
        assert "ibuprofen" not in names(out), kw
        assert any(x["key"] == "ibuprofen" and word in x["reason"] for x in out["avoid"]), (kw, out["avoid"])
        assert "paracetamol" in names(out)                              # the safer option is still offered


def test_ibuprofen_is_offered_to_a_healthy_adult_with_a_headache():
    f, a = asm("headache", duration_days=1, age=30)
    out = medicines.suggest(f, a)
    ib = next(s for s in out["suggestions"] if s["key"] == "ibuprofen")
    assert "200 mg to 400 mg" in ib["dose"] and "1,200 mg" in ib["maximum"] and "food" in ib["dose"]


def test_prolonged_fever_avoids_nsaids_until_dengue_is_excluded():
    f, a = asm("fever and body ache for 4 days", duration_days=4, age=30)
    out = medicines.suggest(f, a)
    assert "ibuprofen" not in names(out) and "paracetamol" in names(out)
    assert any("dengue" in x["reason"] for x in out["avoid"])


def test_no_dosed_medicine_before_the_allergy_and_medicine_check():
    f, a = asm("fever and headache", duration_days=1, age=30, asked=False)
    out = medicines.suggest(f, a)
    assert names(out) == [] and out["needs_screening"] and "allerg" in out["note"].lower()


def test_children_get_no_doses():
    f, a = asm("fever and cough", duration_days=1, age=6)
    out = medicines.suggest(f, a)
    assert out["child"] and not any(s["key"] in ("paracetamol", "ibuprofen", "cetirizine") for s in out["suggestions"])
    assert "weight" in out["note"]


def test_emergency_gets_no_medicines_and_urgent_only_paracetamol():
    f, a = asm("chest pain and fever", level="emergency", age=40)
    assert medicines.suggest(f, a)["suggestions"] == []
    f, a = asm("fever and headache", level="urgent", duration_days=2, age=40)
    out = medicines.suggest(f, a)
    assert names(out) == ["paracetamol"] and "Do not delay" in out["note"]


def test_symptom_specific_choices():
    f, a = asm("runny nose, sneezing and itchy eyes", duration_days=2, age=28)
    assert "cetirizine" in names(medicines.suggest(f, a)) and "saline" in names(medicines.suggest(f, a))
    f, a = asm("loose motions and vomiting", duration_days=1, age=28)
    assert "ors" in names(medicines.suggest(f, a))
    f, a = asm("itchy rash on my arm", duration_days=1, age=28)
    assert "hydrocortisone" in names(medicines.suggest(f, a))
    f, a = asm("heartburn and acidity", duration_days=1, age=28)
    assert "antacid" in names(medicines.suggest(f, a))


def test_denied_symptoms_do_not_attract_medicines():
    f, a = asm("fever, no cough, no rash, no sore throat", duration_days=1, age=30)
    out = medicines.suggest(f, a)
    assert "hydrocortisone" not in names(out) and "gargle" not in names(out)


def test_cetirizine_screen():
    f, a = asm("sneezing and itching", duration_days=1, age=30, risks={"kidney disease"})
    out = medicines.suggest(f, a)
    assert "cetirizine" not in names(out) and any(x["key"] == "cetirizine" for x in out["avoid"])


# ---------------------------------------------------------------- "can I take X?"
def test_can_i_take_questions_are_answered_from_the_screen():
    f, a = asm("headache", duration_days=1, age=30)
    ok = medicines.answer_about("Can I take ibuprofen for this?", f, a)
    assert ok["verdict"] == "ok" and "200 mg to 400 mg" in ok["text"]
    f, a = asm("headache", duration_days=1, age=72)
    no = medicines.answer_about("can i take brufen", f, a)
    assert no["verdict"] == "avoid" and "65" in no["text"] and "Paracetamol is the safer choice" in no["text"]
    f, a = asm("headache", duration_days=1, age=30)
    assert medicines.answer_about("what dose of crocin?", f, a)["verdict"] == "ok"


def test_prescription_medicines_are_never_prescribed():
    f, a = asm("sore throat for 2 days", duration_days=2, age=30)
    for q in ("Can you prescribe antibiotics?", "which steroid tablets should I take", "give me amoxicillin", "is tramadol ok"):
        r = medicines.answer_about(q, f, a)
        assert r["verdict"] == "rx" and "cannot suggest or prescribe" in r["text"], q
    assert medicines.answer_about("what is a good vitamin", f, a)["verdict"] == "ask"
    assert medicines.answer_about("how are you", f, a) is None


def test_children_and_unscreened_patients_get_no_dose_in_answers():
    f, a = asm("fever", duration_days=1, age=5)
    assert medicines.answer_about("can I give paracetamol", f, a)["verdict"] == "ask"
    f, a = asm("fever", duration_days=1, age=30, asked=False)
    assert medicines.answer_about("can I take paracetamol", f, a)["verdict"] == "ask"


# ---------------------------------------------------------------- inside the conversation
class NoModel:
    """Any model call here would be a bug: medicine safety answers must be instant and deterministic."""

    def __init__(self, monkeypatch):
        async def boom(*a, **k):
            raise AssertionError("model must not be called")
        monkeypatch.setattr(consult, "answer", boom)


def talk(s, *msgs):
    return [run(consult.handle(s, m, with_answer=False)) for m in msgs]


REPLIES = {"profile": "I'm 30, healthy", "meds": "none", "severity": "4", "duration": "2 days"}


def converse(s, first, answers=None):
    """Say `first`, then answer whatever the doctor asks (by question id, default 'no') until the assessment arrives."""
    answers = {**REPLIES, **(answers or {})}
    r, asked = run(consult.handle(s, first, with_answer=False)), []
    for _ in range(30):
        if r["stage"] == "advice":
            return r, asked
        asked.append(s.asked)
        r = run(consult.handle(s, answers.get(s.asked, answers.get("*", "no")), with_answer=False))
    raise AssertionError(f"never reached the assessment; asked {asked}")


def test_full_conversation_screenshot_case_end_to_end():
    s, _ = consult.start()
    r, asked = converse(s, "I have a fever of 102 degree fahrenheit and severe nasal blockage",
                        {"duration": "It has been 2 days and it is staying the same", "runny_nose": "yes", "facial_pain": "yes around my eyes",
                         "profile": "I am 39 years old, I do not have any pre-existing disease.", "meds": "no allergies, I take no regular medicines"})
    assert "temp" not in asked and "fever_check" not in asked                   # 102 F was in the first sentence
    assert any(q in asked for q in ("facial_pain", "thick_discharge", "runny_nose", "sneezing"))      # nose questions: the actual complaint
    assert asked[-1] == "meds" and len(asked) <= 12
    a = r["assessment"]
    assert not any("14235" in x for x in a["reasons"]) and any("Fever: 102" in x for x in a["reasons"])
    assert a["possible_causes"] and a["possible_causes"][0]["name"].startswith("Acute sinusitis")
    keys = [m["key"] for m in r["medicines"]["suggestions"]]
    assert "paracetamol" in keys and "saline" in keys
    assert r["pending_question"] == consult.DEFAULT_QUESTION


def test_medicine_questions_are_instant_and_never_reach_the_model(monkeypatch):
    s, _ = consult.start()
    converse(s, "headache for 1 day, 3/10")
    NoModel(monkeypatch)
    r = talk(s, "Can I take ibuprofen?")[0]
    assert r["answer"]["source"] == "formulary" and r["pending_question"] is None and "200 mg to 400 mg" in r["message"]
    r = talk(s, "I am allergic to aspirin")[0]                          # new safety information: instant update, no model
    assert r["answer"]["source"] == "formulary" and "safety_update" in r["answer"]["flags"]
    assert "ibuprofen" not in [m["key"] for m in r["medicines"]["suggestions"]]
    assert any(x["key"] == "ibuprofen" for x in r["medicines"]["avoid"])
    r = talk(s, "can I take ibuprofen then?")[0]
    assert "I would not suggest Ibuprofen" in r["message"] and "allergic" in r["message"]


def test_can_i_take_is_not_mistaken_for_taking_it(monkeypatch):
    s, _ = consult.start()
    converse(s, "headache for 1 day, 3/10")
    NoModel(monkeypatch)
    talk(s, "Can I take ibuprofen?")
    assert "nsaid" not in s.facts.medicines
    talk(s, "I take ibuprofen every day")
    assert "nsaid" in s.facts.medicines


def test_skipping_the_medicine_question_keeps_doses_hidden():
    s, _ = consult.start()
    r, _ = converse(s, "headache for 1 day, 3/10", {"meds": "skip"})
    assert r["stage"] == "advice" and r["medicines"]["needs_screening"]
    assert not any(m["key"] in ("paracetamol", "ibuprofen", "cetirizine") for m in r["medicines"]["suggestions"])      # no dosed medicine unscreened
    assert "allerg" in r["medicines"]["note"].lower()
    r = talk(s, "I am allergic to penicillin, no other medicines")[0]          # telling us later unlocks the screened suggestions
    assert not r["medicines"]["needs_screening"] and "paracetamol" in [m["key"] for m in r["medicines"]["suggestions"]]


# ---------------------------------------------------------------- severity scaling + the closing doctor referral
def test_every_level_ends_with_a_doctor_referral():
    for level, word in (("self_care", "3 days"), ("see_doctor", "2-3 days"), ("urgent", "today"), ("emergency", "emergency care")):
        f, a = asm("fever and headache", level=level, duration_days=1, age=30)
        a.specialty = "ENT specialist"
        out = medicines.suggest(f, a)
        assert word in out["referral"], (level, out["referral"])
    f, a = asm("blocked nose", level="see_doctor", duration_days=1, age=30)
    a.specialty = "ENT specialist"
    assert "ENT specialist" in medicines.suggest(f, a)["referral"]
    a.specialty = "General Physician"
    assert "doctor" in medicines.suggest(f, a)["referral"] and "General Physician" not in medicines.suggest(f, a)["referral"]


def test_dose_guidance_scales_with_severity():
    guide = {}
    for sev in (2, 5, 9):
        f, a = asm("headache", duration_days=1, age=30, severity=sev)
        guide[sev] = medicines.suggest(f, a)["dose_guidance"]
    assert "lower dose" in guide[2] and "usual dose range" in guide[5] and "never more" in guide[9] and "do not wait" in guide[9]


def test_medicine_answers_also_end_with_the_referral():
    f, a = asm("headache", level="see_doctor", duration_days=1, age=30, severity=6)
    ok = medicines.answer_about("Can I take paracetamol?", f, a)["text"]
    assert ok.endswith("only ease symptoms, they do not treat the cause.") and "usual dose range" in ok
    f, a = asm("headache", duration_days=1, age=75)
    no = medicines.answer_about("can i take ibuprofen", f, a)["text"]
    assert no.endswith("please see a doctor.") or "see a doctor" in no.split("Paracetamol")[-1]


def test_the_chat_message_ends_with_the_doctor_referral():
    s, _ = consult.start()
    r, _ = converse(s, "headache for 4 days, 6/10")
    assert r["medicines"]["suggestions"] and r["message"].rstrip().endswith(r["medicines"]["referral"])
