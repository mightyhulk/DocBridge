"""The digital-doctor interview: symptom slots, differential, and dynamic question selection."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "backend"))
import pytest  # noqa: E402

from app import clinic, consult, interview, triage  # noqa: E402
from app.triage import Facts  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def facts(text="", **kw):
    f = Facts(text=text, symptom_text=text, **kw)
    f.slots = clinic.extract(text)
    temp = triage.parse_temperature_c(text)
    if temp is not None:
        f.temp_c = temp
    return f


def asked_ids(f, done=()):
    return [k for k, _, _ in interview.plan(f, set(done))]


# ---------------------------------------------------------------- extraction: what the patient has / does not have
@pytest.mark.parametrize("text,has,lacks", [
    ("I have a fever and a blocked nose", {"fever", "nasal_block"}, set()),
    ("fever of 102, no chills and no rash", {"fever"}, {"chills", "rash"}),
    ("my back hurts for two weeks", {"back_pain"}, set()),
    ("I have a dry cough but no fever", {"cough", "dry_cough"}, {"fever", "wet_cough"}),
    ("cough with yellow phlegm", {"cough", "wet_cough", "sputum_coloured"}, {"dry_cough"}),
    ("sore throat, painful to swallow, no cough", {"sore_throat", "painful_swallow"}, {"cough"}),
    ("throbbing headache on one side of my head with nausea", {"headache", "throbbing", "head_one_sided", "nausea"}, set()),
    ("tight band like headache from stress", {"headache", "head_band", "stress_trigger"}, {"head_one_sided"}),
    ("burning when I pass urine and going to the toilet very often", {"burning_urine", "frequency"}, set()),
    ("loose motions and vomiting after street food", {"diarrhoea", "vomiting", "recent_food"}, set()),
    ("my chest hurts", {"chest_pain"}, set()),
    ("I do not have chest pain", set(), {"chest_pain"}),
    ("itchy red rash on my arms after a new soap", {"rash", "itchy_skin", "new_product"}, set()),
    ("heartburn after meals", {"heartburn", "after_meals"}, set()),
])
def test_slot_extraction(text, has, lacks):
    got = clinic.extract(text)
    assert {k for k in has if got.get(k) is True} == has, got
    assert {k for k in lacks if got.get(k) is False} == lacks, got


def test_acknowledgement_reads_like_notes():
    before = {"fever": True}
    after = {"fever": True, "chills": False, "rash": False, "cough": True}
    ack = clinic.acknowledge(before, after, ["fever of 102°F"])
    assert ack.startswith("Noted: fever of 102°F") and "a cough" in ack and "no chills" in ack
    assert clinic.acknowledge(after, after) == ""                            # nothing new: nothing to acknowledge


def test_yes_and_no_detection():
    assert clinic.is_yes("yes a bit") and clinic.is_yes("yeah") and not clinic.is_yes("no")
    assert clinic.is_no("no, not really") and clinic.is_no("none") and not clinic.is_no("yes")


# ---------------------------------------------------------------- what is asked follows what was said
def test_nothing_is_asked_that_the_patient_already_told_us():
    f = facts("I have a fever of 102 degree fahrenheit and severe nasal blockage")
    ks = asked_ids(f)
    assert "temp" not in ks and "fever_check" not in ks and "fever" not in ks      # temperature and fever: given
    assert any(k in ks for k in ("runny_nose", "facial_pain", "sneezing"))          # nose questions: still open


def test_a_denial_closes_the_question_too():
    ks = asked_ids(facts("fever, no chills, no rash, no stiff neck"))
    assert not {"chills", "fever_rash", "stiff_neck"} & set(ks)


def test_questions_come_from_the_symptom_area():
    assert any(k in asked_ids(facts("a bad headache")) for k in ("worst_headache", "head_side", "neuro_deficit"))
    assert any(k in asked_ids(facts("stomach pain and vomiting")) for k in ("diarrhoea", "blood_stool", "can_keep_fluids"))
    assert "injury" in asked_ids(facts("my knee hurts"))
    assert "painful_swallow" in asked_ids(facts("sore throat"))
    assert "burning_urine" in asked_ids(facts("I pass urine very often"))
    assert "itchy_skin" in [q.id for _, q in interview.candidates(facts("a rash on my arm"), set())]


def test_prerequisites_are_respected():
    assert "sputum_colour" not in asked_ids(facts("a dry cough"))               # no phlegm: no question about its colour
    assert "cough_kind" in asked_ids(facts("a cough"))
    assert "stiff_neck" not in asked_ids(facts("a headache"))                   # no fever: no meningitis screen
    assert "stiff_neck" in asked_ids(facts("fever and headache"))
    assert "double_worsening" not in asked_ids(facts("blocked nose since yesterday", duration_days=1))


def test_red_flag_screens_come_before_detail():
    ks = [k for k in asked_ids(facts("fever and headache", duration_days=2, severity=5))]
    assert ks[0] in ("stiff_neck", "worst_headache", "neuro_deficit")
    ks = [k for k in asked_ids(facts("stomach pain and vomiting", duration_days=1, severity=5))]
    assert ks[0] in ("blood_stool", "can_keep_fluids")
    assert ks[0] != "recent_food"


def test_new_information_adds_new_questions():
    f = facts("runny nose for 2 days")
    assert "cough_kind" not in asked_ids(f)
    f2 = facts("runny nose for 2 days and now I have also started coughing")
    assert "cough_kind" in asked_ids(f2)


def test_branching_differs_by_what_the_patient_says():
    migraine = facts("headache on one side, throbbing", duration_days=1, severity=6)
    tension = facts("headache like a tight band, stressful week", duration_days=1, severity=6)
    km, kt = asked_ids(migraine), asked_ids(tension)
    assert "aura" in km or "light_sensitive" in km or "head_nausea" in km
    assert "aura" not in kt and "head_side" not in kt                           # band/all-over is already answered
    assert "stress_trigger" not in kt                                           # stress already mentioned


def test_questions_that_separate_the_likely_conditions_rank_higher():
    """With sinusitis and a cold both plausible, facial pain outranks an unrelated detail; with an allergy picture, itching does."""
    f = facts("blocked nose and thick yellow mucus for a week", duration_days=8, severity=4)
    order = [q.id for _, q in interview.candidates(f, set())]
    assert "facial_pain" in order[:3]
    assert order.index("facial_pain") < order.index("one_sided_nose") if "one_sided_nose" in order else True
    g = facts("sneezing and a runny nose every morning, dust makes it worse", duration_days=30, severity=3)
    og = [q.id for _, q in interview.candidates(g, set())]
    assert "itchy_nose_eyes" in og[:4]


def test_at_most_seven_symptom_questions_then_the_basics():
    f = facts("fever, cough, stomach pain, rash, headache and ear pain", duration_days=2, severity=5)
    sym = [k for k in asked_ids(f) if k in interview.BY_ID]
    assert len(sym) <= 4                                                         # shown per turn; the overall cap is 7
    done = set(list(interview.BY_ID)[:7])
    assert [k for k in asked_ids(f, done) if k in interview.BY_ID] == []


def test_basics_and_safety_questions_come_last_and_only_when_unknown():
    f = facts("fever 102 F for 3 days, 6/10", duration_days=3, severity=6, age=40)
    ks = asked_ids(f)
    assert ks[-1] == "meds" and not {"duration", "severity", "profile"} & set(ks)
    f.allergies_asked = True
    assert "meds" not in asked_ids(f)


def test_vague_complaint_gets_an_open_question_first():
    assert asked_ids(facts("I feel unwell"))[0] == "open"
    assert asked_ids(facts("I feel unwell"), done={"open"})[0] != "open"


def test_a_question_is_asked_only_once_even_if_still_unknown():
    f = facts("headache", duration_days=1, severity=4)
    first = asked_ids(f)[0]
    assert first not in asked_ids(f, done={first})


# ---------------------------------------------------------------- differential
def names(f):
    return [d["name"].split(" (")[0] for d in clinic.differential(f)]


def test_differential_separates_cold_allergy_and_sinusitis():
    cold = facts("runny nose, sneezing, mild sore throat and a cough", severity=2, duration_days=2)
    assert names(cold)[0] == "Common cold"
    allergy = facts("sneezing, itchy eyes and a runny nose every morning, no fever", duration_days=30, severity=2)
    assert names(allergy)[0] == "Allergic rhinitis"
    sinus = facts("blocked nose, facial pain and thick yellow mucus, got better then worse again", duration_days=9, severity=5)
    assert names(sinus)[0] == "Acute sinusitis"


def test_differential_needs_the_defining_symptom():
    f = facts("fever of 102 and body ache, no rash", duration_days=2)
    assert "Viral rash illness" not in names(f) and "Contact or allergic skin reaction" not in names(f)
    assert names(f)[0] in ("Influenza-like illness", "Dengue or other tropical viral fever")


def test_differential_flags_dangerous_look_alikes():
    f = facts("fever, cough with yellow phlegm and shortness of breath", duration_days=4, severity=6)
    assert "Chest infection / pneumonia" in names(f)
    f = facts("stomach pain in the lower right abdomen, vomiting and fever", duration_days=1, severity=7)
    assert "Appendicitis" in names(f)
    f = facts("fever with burning urine and pain in my side", duration_days=2)
    assert "Kidney infection" in names(f)


def test_differential_gives_reasons_not_a_diagnosis():
    d = clinic.differential(facts("sore throat, painful to swallow, fever, swollen glands, white patches, no cough", duration_days=2))[0]
    assert d["name"].startswith("Bacterial sore throat") and "painful swallowing" in d["because"] and d["note"]


# ---------------------------------------------------------------- red flags confirmed with a plain "yes"
def converse_asking(first, answers):
    s, _ = consult.start()
    r = run(consult.handle(s, first, with_answer=False))
    asked = []
    for _ in range(30):
        if r["stage"] == "advice":
            return s, r, asked
        asked.append(s.asked)
        r = run(consult.handle(s, answers.get(s.asked, answers.get("*", "no")), with_answer=False))
    raise AssertionError(asked)


def test_a_plain_yes_to_chest_pain_is_an_emergency():
    s, r, asked = converse_asking("I have a cough for three days, mild", {"chest_pain": "yes", "severity": "3", "duration": "3 days"})
    assert r["assessment"]["level"] == "emergency" and "chest pain" in " ".join(r["assessment"]["reasons"]).lower()
    assert "profile" not in asked and "meds" not in asked                      # the interview stops: no more questions in an emergency


def test_a_plain_yes_to_stiff_neck_with_fever_is_urgent():
    s, r, asked = converse_asking("I have a headache and fever for 2 days, 5/10", {"stiff_neck": "yes", "temp": "102"})
    assert r["assessment"]["level"] == "urgent" and any("stiff neck" in x for x in r["assessment"]["reasons"])


def test_blood_in_stool_and_blood_in_phlegm_are_urgent():
    _, r, _ = converse_asking("stomach pain and vomiting since yesterday, 5/10", {"blood_stool": "yes black stools"})
    assert r["assessment"]["level"] == "urgent"
    _, r, _ = converse_asking("a cough for a week, 4/10", {"blood_sputum": "yes a little blood"})
    assert r["assessment"]["level"] == "urgent"


def test_unable_to_keep_fluids_down_is_urgent_but_able_is_not():
    _, bad, _ = converse_asking("vomiting and loose motions since yesterday, 5/10", {"can_keep_fluids": "no"})
    assert bad["assessment"]["level"] == "urgent" and any("fluids" in x for x in bad["assessment"]["reasons"])
    _, ok, _ = converse_asking("vomiting and loose motions since yesterday, 5/10", {"can_keep_fluids": "yes"})
    assert ok["assessment"]["level"] != "urgent"


# ---------------------------------------------------------------- whole conversations: never a repeat, always relevant
def test_no_question_is_ever_asked_twice_and_none_is_already_known():
    openers = ["I have a fever of 102 degree fahrenheit and severe nasal blockage", "bad headache and fever since yesterday",
               "my stomach hurts and I vomited", "my knee hurts when I climb stairs", "burning when I pass urine", "an itchy rash on my arms",
               "sore throat and cough", "I feel unwell"]
    for first in openers:
        s, _ = consult.start()
        r = run(consult.handle(s, first, with_answer=False))
        seen = []
        for _ in range(30):
            if r["stage"] == "advice":
                break
            q = s.asked
            assert q not in seen, (first, q, seen)
            seen.append(q)
            known = clinic.state(s.facts)
            item = interview.BY_ID.get(q)
            if item and item.slot:
                assert known.get(item.slot) is None, (first, q, "was already known")            # never ask what we know
            r = run(consult.handle(s, {"profile": "I'm 30, healthy", "meds": "none", "severity": "4", "duration": "2 days"}.get(q, "no"), with_answer=False))
        assert r["stage"] == "advice", (first, seen)
        assert len([x for x in seen if x in interview.BY_ID]) <= interview.MAX_SYMPTOM_QUESTIONS


def test_acknowledgements_and_reasons_accompany_every_question():
    s, _ = consult.start()
    r = run(consult.handle(s, "I have a fever of 102 degree fahrenheit and a blocked nose", with_answer=False))
    assert r["why"] and r["ack"].startswith("Noted:") and "102°F" in r["ack"] and r["message"].startswith(r["ack"])
    r = run(consult.handle(s, "2 days", with_answer=False))
    assert "for 2 days" in r["ack"] and r["why"]


def test_either_or_questions_ignore_a_bare_yes():
    s, _ = consult.start()
    run(consult.handle(s, "I have a cough for 2 days, 3/10", with_answer=False))
    while s.asked != "cough_kind":
        run(consult.handle(s, "no", with_answer=False))
    run(consult.handle(s, "yes", with_answer=False))                          # "dry, or with phlegm?" - 'yes' says nothing
    assert "wet_cough" not in s.facts.slots or s.facts.slots["wet_cough"] is None
    s2, _ = consult.start()
    run(consult.handle(s2, "I have a cough for 2 days, 3/10", with_answer=False))
    while s2.asked != "cough_kind":
        run(consult.handle(s2, "no", with_answer=False))
    run(consult.handle(s2, "it is dry", with_answer=False))
    assert s2.facts.slots["wet_cough"] is False and s2.facts.slots["dry_cough"] is True


def test_confirmed_symptoms_reach_the_specialist_and_medicine_logic():
    s, _ = consult.start()
    r = run(consult.handle(s, "I have a bad cold", with_answer=False))
    while s.stage == "interview":
        r = run(consult.handle(s, {"sore_throat": "yes", "profile": "30, healthy", "meds": "none", "severity": "3", "duration": "2 days"}.get(s.asked, "no"), with_answer=False))
    assert "sore throat" in s.facts.symptom_text.lower() or s.facts.slots.get("sore_throat") is not True
    assert r["assessment"]["specialty"] in ("ENT specialist", "General Physician")


def test_the_possible_causes_are_returned_with_likelihood_and_reasons():
    _, r, _ = converse_asking("blocked nose, facial pain and thick yellow mucus for 9 days", {"duration": "9 days", "severity": "5", "facial_pain": "yes",
                                                                                           "thick_discharge": "yes", "double_worsening": "yes"})
    pc = r["assessment"]["possible_causes"]
    assert pc and pc[0]["likelihood"] in ("most likely", "possible") and pc[0]["because"] and pc[0]["note"]


def test_nausea_with_a_headache_does_not_open_stomach_questions():
    ks = asked_ids(facts("a throbbing headache on the left side and I feel sick", duration_days=1, severity=6))
    assert not {"vomiting", "diarrhoea", "abdominal_pain", "blood_stool", "can_keep_fluids"} & set(ks)
    assert any(k in ks for k in ("worst_headache", "neuro_deficit", "light_sensitive"))
    stomach = asked_ids(facts("I feel sick and have stomach pain", duration_days=1, severity=6))
    assert {"vomiting", "diarrhoea"} & set(stomach) or "blood_stool" in stomach          # with stomach pain the gut questions do open


def test_acknowledgement_lists_findings_in_a_natural_order():
    after = clinic.extract("a throbbing headache on the left side and I feel sick")
    ack = clinic.acknowledge({}, after)
    assert ack.index("headache") < ack.index("throbbing") < ack.index("nausea")


# ---------------------------------------------------------------- a microphone that hears the speaker (found in a real session)
BOT_Q = ("Noted: for 2 days. How old are you, and do you have any existing conditions (such as diabetes, heart or lung disease), "
         "or are you pregnant?")


def test_echo_of_the_doctors_question_is_removed_from_the_answer():
    heard = "Dr. Dibedis, heart or lung disease, or are you pregnant? 39 years, no existing disease."
    got = consult.strip_echo(heard, BOT_Q)
    assert "lung" not in got and "pregnant" not in got and "39 years" in got and "no existing disease" in got


def test_a_genuine_answer_is_never_trimmed():
    assert consult.strip_echo("I am 39, no conditions", BOT_Q) == "I am 39, no conditions"
    assert consult.strip_echo("yes", BOT_Q) == "yes"
    assert consult.strip_echo("I have diabetes and heart disease", BOT_Q) == "I have diabetes and heart disease"      # <5 words in a row
    only_echo = "how old are you and do you have any existing conditions"
    assert consult.strip_echo(only_echo, BOT_Q) == only_echo                    # nothing would be left: keep it


def test_the_screenshot_session_no_phantom_conditions_and_no_14235_days():
    """The patient said '39 years, no existing disease'; the mic also transcribed the spoken question."""
    s, _ = consult.start()
    heard = "Dr. Dibedis, heart or lung disease, or are you pregnant? 39 years, no existing disease."
    run(consult.handle(s, "I have a fever of 102 degree fahrenheit and severe nasal blockage", with_answer=False))
    r = run(consult.handle(s, "2 days", with_answer=False))
    while s.asked != "profile":
        r = run(consult.handle(s, "no", with_answer=False))
    r = run(consult.handle(s, heard, with_answer=False))
    assert s.facts.age == 39 and not s.facts.risks
    assert s.facts.duration_days == 2                                      # "39 years" was not filed as an illness duration
    assert "months" not in r["message"] and "age 39" in r["message"]
    r = run(consult.handle(s, "None.", with_answer=False))
    assert r["stage"] == "advice" and not any("lung" in x or "pregnan" in x or "14235" in x for x in r["assessment"]["reasons"])


def test_even_without_echo_no_existing_disease_is_an_answer_not_a_list():
    s, _ = consult.start()
    run(consult.handle(s, "fever of 102 F for 2 days", with_answer=False))
    while s.asked != "profile":
        run(consult.handle(s, "no", with_answer=False))
    run(consult.handle(s, "I am 39, I do not have any existing disease like lung disease or diabetes", with_answer=False))
    assert not s.facts.risks


def test_a_denial_covers_its_own_examples_but_not_a_real_condition_mentioned_before_it():
    s, _ = consult.start()
    run(consult.handle(s, "fever of 102 F for 2 days", with_answer=False))
    while s.asked != "profile":
        run(consult.handle(s, "no", with_answer=False))
    run(consult.handle(s, "I am 50, I have diabetes, but no other medical problems like asthma", with_answer=False))
    assert s.facts.risks == {"diabetes"}
