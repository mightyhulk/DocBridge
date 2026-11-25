"""How the bot reads what patients actually say: negations, family history, third parties, bare numbers, odd phrasings.

Every case here is something a real patient might type or say. A misreading is not cosmetic: a phantom allergy blocks the right
medicine, a phantom condition changes the advice, a missed denial raises a false alarm. These tests exist because a real session
turned "I am not allergic to any medicine" into "allergic to many medicines".
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "backend"))
import pytest  # noqa: E402

from app import clinic, consult, triage  # noqa: E402


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- allergies: a denial is NOT an allergy
@pytest.mark.parametrize("text", [
    "I am not allergic to any medicine.", "I am not allergic to any medicines", "not allergic to anything", "no allergies", "No known allergies",
    "no known drug allergies", "NKDA", "none, I have no allergies at all", "I don't have any allergies", "I do not have any drug allergies",
    "I have never had an allergic reaction to a medicine", "nope, not allergic", "I'm not allergic to penicillin or aspirin", "No allergy to any drug",
    "I am not allergic to any medicines and I take no regular medicines", "not allergic to anything that I know of", "allergies: none", "I have no medicine allergy",
    "Not that I know of, no allergies", "I don't think I'm allergic to any medicine", "never been allergic to any tablet",
])
def test_denying_allergies_records_no_allergy(text):
    assert triage.parse_allergies(text) == set(), text


@pytest.mark.parametrize("text,expected", [
    ("I am allergic to penicillin", {"penicillin"}), ("allergic to aspirin and ibuprofen", {"nsaid"}), ("penicillin allergy", {"penicillin"}),
    ("I have an allergy to sulfa drugs", {"sulfa"}), ("allergic to amoxicillin", {"penicillin"}), ("I react badly to ibuprofen, I'm allergic", {"nsaid"}),
    ("allergic to penicillin but not aspirin", {"penicillin"}), ("not allergic to penicillin, but allergic to aspirin", {"nsaid"}),
    ("I am allergic to many medicines", {"many"}), ("allergic to all antibiotics", set()), ("allergies: penicillin and sulfa", {"penicillin", "sulfa"}),
    ("I'm allergic to paracetamol", {"paracetamol"}), ("I have a codeine allergy", set()),
])
def test_real_allergies_are_still_recorded(text, expected):
    got = triage.parse_allergies(text)
    assert expected <= got and not (got - expected - {"nsaid"} if "nsaid" in expected else got - expected), (text, got)


def test_an_allergy_conversation_never_blocks_medicines_for_a_denial():
    s, _ = consult.start()
    run(consult.handle(s, "I have a fever of 102 F and a blocked nose for 2 days, 5/10", with_answer=False))
    for _ in range(25):
        if s.asked == "meds" or s.stage == "advice":
            break
        run(consult.handle(s, {"profile": "39 years, I do not have any existing conditions."}.get(s.asked, "no"), with_answer=False))
    r = run(consult.handle(s, "I am not allergic to any medicine.", with_answer=False))
    assert r["stage"] == "advice" and not s.facts.allergies
    keys = {m["key"] for m in r["medicines"]["suggestions"]}
    assert "paracetamol" in keys and not r["medicines"]["avoid"] and not r["medicines"]["needs_screening"]


# ---------------------------------------------------------------- regular medicines: "I take no X" is not "I take X"
@pytest.mark.parametrize("text", [
    "I take no regular medicines", "I'm not taking any blood thinners", "no medicines", "I don't take aspirin", "I take nothing", "none",
    "I do not take any tablets", "I am not on any medication", "I take no aspirin or warfarin", "not on blood thinners", "I stopped taking warfarin last year",
])
def test_denied_or_past_medicines_are_not_current(text):
    assert triage.parse_medicines(text) == set(), text


@pytest.mark.parametrize("text,expected", [
    ("I take warfarin", {"anticoagulant"}), ("I'm on aspirin every day", {"aspirin"}), ("I take losartan and sertraline", {"bp_kidney", "antidepressant"}),
    ("I take blood thinners", {"anticoagulant"}), ("I take crocin when I have fever", {"paracetamol"}), ("not allergic, I take lisinopril", {"bp_kidney"}),
])
def test_real_medicines_are_recorded(text, expected):
    assert triage.parse_medicines(text) == expected, text


# ---------------------------------------------------------------- conditions: denials, family history, third parties
@pytest.mark.parametrize("text", [
    "I do not have any existing conditions", "no existing disease", "none", "I'm healthy", "I don't have diabetes", "I am not pregnant",
    "no diabetes, no asthma", "I do not have heart or lung disease", "my mother has diabetes", "my father had a heart attack", "diabetes runs in my family",
    "family history of heart disease", "my wife is pregnant", "my grandmother has asthma and kidney disease", "my sister is diabetic",
    "I am expecting to feel better soon", "I'm up to date with my immunisation", "I use a steroid cream on my elbow", "never had asthma",
    "no chance I'm pregnant", "I have no history of kidney problems",
])
def test_conditions_that_are_denied_or_belong_to_someone_else_are_not_the_patients(text):
    assert triage.parse_risks(text) == set(), text


@pytest.mark.parametrize("text,expected", [
    ("I have diabetes", {"diabetes"}), ("I am diabetic and have high blood pressure", {"diabetes", "heart disease"}), ("I'm 12 weeks pregnant", {"pregnancy"}),
    ("I have asthma", {"lung disease"}), ("I have kidney disease", {"kidney disease"}), ("I have diabetes but no asthma", {"diabetes"}),
    ("my mother has diabetes and I have asthma", {"lung disease"}), ("I have a stomach ulcer", {"stomach ulcer or gastritis"}),
    ("I am on chemotherapy for cancer", {"weakened immunity"}), ("I have a pacemaker", {"heart disease"}),
])
def test_real_conditions_are_still_recorded(text, expected):
    assert triage.parse_risks(text) == expected, text


def test_a_profile_answer_with_a_family_history_clause_is_read_correctly():
    s, _ = consult.start()
    run(consult.handle(s, "fever of 102 F for 2 days, 5/10", with_answer=False))
    while s.asked != "profile":
        run(consult.handle(s, "no", with_answer=False))
    run(consult.handle(s, "I am 45, my father has heart disease and my mother has diabetes, but I am healthy", with_answer=False))
    assert s.facts.age == 45 and not s.facts.risks


# ---------------------------------------------------------------- severity words
@pytest.mark.parametrize("text,expected", [
    ("it is not severe", None), ("not very painful", None), ("no pain", None), ("it is severe", 8), ("mild", 2), ("it's not mild, it's quite bad", None),
    ("6/10", 6), ("about 7 out of 10", 7), ("pain is 4", 4), ("unbearable", 10), ("moderate", 5), ("not too bad", None),
])
def test_severity_words_respect_negation(text, expected):
    assert triage.parse_severity(text) == expected, text


# ---------------------------------------------------------------- age
@pytest.mark.parametrize("text,expected", [
    ("I'm 39", 39), ("39 years old", 39), ("I am 39 years old, no conditions", 39), ("age 39", 39), ("my son is 5", 5), ("my daughter is 8 years old", 8),
    ("I have a 5 year old son and I'm 39", 39), ("I am 39 and my son is 5", 39), ("I'm 39, my wife is 36", 39), ("2 months old", 0),
])
def test_age_prefers_the_patient_over_relatives(text, expected):
    assert triage.parse_age(text) == expected, text


# ---------------------------------------------------------------- temperature
@pytest.mark.parametrize("text,celsius", [
    ("I have a fever of 102 degree fahrenheit", 38.9), ("temperature is 38.5 C", 38.5), ("99.1 F", None), ("my pulse is 100 and I have fever", None),
    ("fever of 101", 38.3), ("I took 2 tablets for the fever", None), ("no fever, temperature 98.6", 37.0),
])
def test_temperature_parsing(text, celsius):
    got = triage.parse_temperature_c(text)
    assert (got is None) if celsius is None else got == pytest.approx(celsius, abs=0.15), (text, got)


# ---------------------------------------------------------------- symptoms: denials never become symptoms
@pytest.mark.parametrize("text,absent", [
    ("no fever, no cough", {"fever", "cough"}), ("I do not have a headache", {"headache"}), ("not short of breath", {"breath_short"}),
    ("I haven't vomited", {"vomiting"}), ("without any rash", {"rash"}), ("I don't have chest pain or fever", {"chest_pain", "fever"}),
    ("never had a cough like this", {"cough"}), ("there is no blood in my stool", {"blood_stool"}),
])
def test_denied_symptoms_are_never_recorded_as_present(text, absent):
    got = clinic.extract(text)
    assert not any(got.get(k) is True for k in absent), (text, got)


@pytest.mark.parametrize("text,present", [
    ("I have a fever and a cough", {"fever", "cough"}), ("fever but no cough", {"fever"}), ("no fever but I have a cough", {"cough"}),
    ("I have not eaten but I have a headache", {"headache"}), ("I'm not sleeping well and I have a headache", {"headache"}),
])
def test_present_symptoms_survive_nearby_denials(text, present):
    got = clinic.extract(text)
    assert all(got.get(k) is True for k in present), (text, got)


# ---------------------------------------------------------------- yes / no answers
@pytest.mark.parametrize("text,yes,no", [
    ("yes", True, False), ("yeah a bit", True, False), ("no", False, True), ("nope", False, True), ("not really", False, True), ("none", False, True),
    ("I don't know", False, False), ("not sure", False, False), ("maybe", False, False), ("no, but I have a fever", False, True),
])
def test_yes_no_detection(text, yes, no):
    assert clinic.is_yes(text) is yes and clinic.is_no(text) is no, text


def test_dont_know_does_not_deny_a_symptom():
    """'I don't know' to a symptom question must leave the symptom unknown, not record it as absent."""
    s, _ = consult.start()
    run(consult.handle(s, "I have a fever of 102 F and a headache for 2 days, 5/10", with_answer=False))
    while s.asked != "stiff_neck":
        run(consult.handle(s, "no", with_answer=False))
    run(consult.handle(s, "I don't know", with_answer=False))
    assert s.facts.slots.get("stiff_neck") is None


@pytest.mark.parametrize("text,days", [
    ("39 years, no conditions", None), ("39 years", None), ("I have had this cough for 3 years", 1095), ("for about 20 years", 7300),
    ("about 2 days", 2), ("for 2 years", 730), ("I am 45 years old", None),
])
def test_an_age_is_never_a_symptom_duration(text, days):
    got = triage.parse_duration_days(text)
    assert got == days or (days and got and abs(got - days) < 40), (text, got)
