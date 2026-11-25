"""History titles name the symptoms, never a reply like "No"."""
from app.history import derive_title


def u(t): return {"role": "user", "content": t}
def a(t): return {"role": "assistant", "content": t}


def test_own_words_are_kept_when_they_describe_the_problem():
    assert derive_title([u("I have a fever and a sore throat")], None, "I have a fever and a sore throat") == "I have a fever and a sore throat"


def test_filler_title_is_rebuilt_from_the_symptoms_the_assistant_noted():
    log = [a("Welcome back..."), u("No"), a("Noted: severity 10/10, a headache. How long has this been going on?"), u("2 days")]
    assert derive_title(log, None, "No") == "Headache, severity 10/10"


def test_negatives_and_durations_are_not_symptoms():
    log = [u("no"), a("Noted: no chills, for 2 days. Do you have a fever?"), a("Noted: fever of 102°F, no chills. Anything else?")]
    assert derive_title(log, None, "no") == "Fever of 102°F"


def test_falls_back_to_the_assessment_reason_then_a_plain_label():
    res = {"assessment": {"reasons": ["Emergency warning sign: a sudden, worst-ever headache"]}}
    assert derive_title([u("No")], res, "No") == "Sudden, worst-ever headache"
    assert derive_title([u("No")], None, "No") == "Consultation"


def test_long_titles_are_cut_at_a_word():
    log = [u("No"), a("Noted: severe knee pain all over the knee, worse when I walk upstairs and at night. Anything else?")]
    t = derive_title(log, None, "No")
    assert len(t) <= 49 and t.endswith("…") and " " not in t[-2:]
