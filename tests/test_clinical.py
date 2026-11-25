"""Clinical Assistant decision support: medication safety, interactions, chart review, and question routing."""
import asyncio
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "backend"))
import pytest  # noqa: E402

from app import clinical as c  # noqa: E402

TODAY = date(2026, 10, 7)


def pt(**kw):
    base = {"patient_id": "P0000099", "full_name": "Test Patient", "age": 50, "sex": "M", "conditions": "", "medications": "", "allergies": "",
            "smoker": 0, "bp_sys": 118, "bp_dia": 76, "heart_rate": 72, "bmi": 24.0, "hba1c": 0, "ldl": 90, "last_visit": "2026-08-01", "recent_encounters": []}
    base.update(kw)
    return base


def sev(res):
    return {f["title"]: f["severity"] for f in res["findings"]}


def has(res, severity, word):
    return any(f["severity"] == severity and word.lower() in (f["title"] + f["detail"]).lower() for f in res["findings"])


# ---------------------------------------------------------------- drug recognition
def test_drug_names_brands_and_classes_are_recognised():
    assert c.find_drugs("can I give Brufen with Coumadin?") == ["ibuprofen", "warfarin"]
    assert c.find_drugs("Is a statin ok with clarithromycin") == ["clarithromycin", "atorvastatin"] or set(c.find_drugs("Is a statin ok with clarithromycin")) == {"atorvastatin", "clarithromycin"}
    assert c.find_drugs("she takes lisinopril and ibuprofens") == ["lisinopril", "ibuprofen"]
    assert c.find_drugs("how are you today") == []


def test_medication_text_is_parsed_with_classes():
    meds = c.parse_medications("Atorvastatin 40 mg nightly; Salbutamol inhaler as needed; Sertraline 50 mg daily")
    assert [m["name"] for m in meds] == ["atorvastatin", "salbutamol", "sertraline"]
    assert "statin" in meds[0]["classes"] and "ssri" in meds[2]["classes"] and all(m["known"] for m in meds)


# ---------------------------------------------------------------- interactions
@pytest.mark.parametrize("a,b,severity", [
    ("warfarin", "ibuprofen", "major"), ("warfarin", "aspirin", "major"), ("warfarin", "metronidazole", "major"), ("warfarin", "clarithromycin", "major"),
    ("lisinopril", "losartan", "major"), ("lisinopril", "spironolactone", "major"), ("sertraline", "tramadol", "major"),
    ("atorvastatin", "clarithromycin", "major"), ("metoprolol", "verapamil", "major"), ("digoxin", "amiodarone", "major"),
    ("sildenafil", "isosorbide mononitrate", "contraindicated"), ("levothyroxine", "ferrous sulfate", "moderate"),
    ("lisinopril", "ibuprofen", "moderate"), ("warfarin", "paracetamol", "moderate"), ("glimepiride", "co-trimoxazole", "major"),
    ("propranolol", "salbutamol", "major"), ("ibuprofen", "naproxen", "major"), ("sertraline", "fluoxetine", "major"),
])
def test_known_interactions(a, b, severity):
    res = c.pair_check(a, b)
    assert res["findings"], (a, b)
    assert res["findings"][0]["severity"] == severity, (a, b, res["findings"][0])
    assert res["verdict"] == ("avoid" if severity in ("major", "contraindicated") else "caution")


def test_the_pair_check_is_symmetric_and_quiet_for_unrelated_drugs():
    assert [f["severity"] for f in c.pair_check("ibuprofen", "warfarin")["findings"]] == [f["severity"] for f in c.pair_check("warfarin", "ibuprofen")["findings"]]
    assert c.pair_check("atorvastatin", "amlodipine")["findings"] == [] and c.pair_check("paracetamol", "omeprazole")["verdict"] == "ok"


def test_ace_plus_arb_is_not_double_reported_as_duplicate_class():
    f = c.pair_check("lisinopril", "losartan")["findings"]
    assert len(f) == 1 and "renin-angiotensin" in f[0]["detail"]


def test_paracetamol_warfarin_mentions_the_inr():
    assert "INR" in c.pair_check("paracetamol", "warfarin")["findings"][0]["detail"]


# ---------------------------------------------------------------- allergies
def test_penicillin_allergy_blocks_penicillins_and_flags_cephalosporins():
    p = pt(allergies="Penicillin")
    assert c.medication_check(p, "amoxicillin")["verdict"] == "avoid" and has(c.medication_check(p, "amoxicillin"), "contraindicated", "penicillin")
    ceph = c.medication_check(p, "ceftriaxone")
    assert ceph["verdict"] == "caution" and has(ceph, "moderate", "cross-reactivity")
    assert c.medication_check(p, "azithromycin")["findings"] == []


def test_nsaid_and_aspirin_allergies():
    ib = c.medication_check(pt(allergies="Ibuprofen"), "ibuprofen")
    assert ib["verdict"] == "avoid" and has(ib, "contraindicated", "allergy")
    assert c.medication_check(pt(allergies="Ibuprofen"), "naproxen")["verdict"] == "avoid"          # cross-reactive
    asp = c.medication_check(pt(allergies="Aspirin"), "ibuprofen")
    assert asp["verdict"] == "avoid" and has(asp, "major", "cross-reactivity")
    assert c.medication_check(pt(allergies="Aspirin"), "paracetamol")["findings"] == []


def test_sulfa_allergy_blocks_cotrimoxazole_but_only_notes_non_antibiotic_sulfonamides():
    assert c.medication_check(pt(allergies="Sulfa drugs"), "co-trimoxazole")["verdict"] == "avoid"
    f = c.medication_check(pt(allergies="Sulfa drugs"), "furosemide")
    assert f["verdict"] == "ok" and any(x["severity"] == "info" for x in f["findings"])               # information, not a block


def test_non_drug_allergies_are_ignored():
    assert c.medication_check(pt(allergies="Shellfish, Eggs, Pollen, Latex, Peanuts"), "ibuprofen")["findings"] == []


# ---------------------------------------------------------------- conditions and age
def test_ibuprofen_in_kidney_disease_and_heart_failure_is_avoided():
    assert c.medication_check(pt(conditions="Chronic Kidney Disease"), "ibuprofen")["verdict"] == "avoid"
    assert c.medication_check(pt(conditions="Heart Failure"), "naproxen")["verdict"] == "avoid"
    assert c.medication_check(pt(conditions="Hypertension"), "ibuprofen")["verdict"] == "caution"
    assert c.medication_check(pt(conditions="GERD"), "diclofenac")["verdict"] == "caution"


def test_beta_blockers_and_airway_disease():
    assert c.medication_check(pt(conditions="Asthma"), "propranolol")["verdict"] == "avoid"
    assert c.medication_check(pt(conditions="COPD"), "propranolol")["verdict"] == "avoid"
    assert c.medication_check(pt(conditions="Asthma"), "bisoprolol")["verdict"] == "caution"


def test_metformin_and_sulfonylurea_in_ckd_and_steroid_in_diabetes():
    assert has(c.medication_check(pt(conditions="Chronic Kidney Disease"), "metformin"), "major", "egfr")
    assert c.medication_check(pt(conditions="Chronic Kidney Disease"), "glimepiride")["verdict"] == "avoid"
    assert c.medication_check(pt(conditions="Type 2 Diabetes"), "prednisolone")["verdict"] == "avoid"
    assert c.medication_check(pt(conditions="Heart Failure"), "verapamil")["verdict"] == "avoid"
    assert c.medication_check(pt(conditions="Heart Failure"), "pioglitazone")["verdict"] == "avoid"


def test_age_and_sex_specific_cautions():
    assert has(c.medication_check(pt(age=72), "ibuprofen"), "moderate", "age 72")
    assert has(c.medication_check(pt(age=68), "ciprofloxacin"), "moderate", "tendon")
    assert has(c.medication_check(pt(age=30, sex="F"), "lisinopril"), "moderate", "child-bearing")
    assert not has(c.medication_check(pt(age=30, sex="M"), "lisinopril"), "moderate", "child-bearing")


def test_checks_against_the_current_medicines():
    p = pt(medications="Warfarin 5 mg daily; Lisinopril 10 mg daily", conditions="Atrial Fibrillation, Hypertension")
    res = c.medication_check(p, "ibuprofen")
    assert res["verdict"] == "avoid" and has(res, "major", "warfarin") and has(res, "moderate", "lisinopril")
    again = c.medication_check(p, "warfarin")
    assert any(f["type"] == "duplication" for f in again["findings"])                      # already on it


def test_a_clean_check_says_what_it_covered_and_not_that_it_is_safe():
    p = pt(medications="Atorvastatin 40 mg nightly", conditions="Hyperlipidemia")
    res = c.medication_check(p, "paracetamol")
    assert res["verdict"] == "ok" and res["findings"] == []
    text = c.render_medcheck(p, res)
    assert "No problems found in these checks" in text and "not a guarantee" in text and "Checked against" in text


# ---------------------------------------------------------------- chart review: what is already wrong
def test_chart_review_finds_the_prescribing_problems_already_present():
    p = pt(conditions="Chronic Kidney Disease, Type 2 Diabetes, Hypertension", medications="Metformin 1000 mg twice daily; Ibuprofen 400 mg three times daily; Lisinopril 10 mg daily",
           allergies="Ibuprofen", age=70)
    titles = [a["title"] for a in c.review(p, TODAY)]
    assert any("Ibuprofen despite a recorded allergy" in t for t in titles)
    assert any(a["severity"] == "critical" and a["category"] == "allergy" for a in c.review(p, TODAY))
    assert any("Metformin in this patient: CKD" in t for t in titles)
    assert any("Ibuprofen + Lisinopril" in t for t in titles)
    assert c.review(p, TODAY)[0]["severity"] == "critical"                                  # most severe first


def test_dual_raas_blockade_in_the_chart_is_flagged():
    a = c.review(pt(medications="Lisinopril 10 mg daily; Losartan 50 mg daily", conditions="Hypertension"), TODAY)
    assert any(x["category"] == "interaction" and "major" in x["title"] for x in a)


def test_chart_with_nothing_wrong_has_no_alerts():
    p = pt(conditions="Migraine", medications="Paracetamol 500 mg as needed", last_visit="2026-09-01")
    assert c.review(p, TODAY) == []


# ---------------------------------------------------------------- care gaps and thresholds
def gaps(**kw):
    return {a["title"]: a for a in c.care_gaps(pt(**kw), TODAY)}


def test_blood_pressure_thresholds():
    assert any(a["severity"] == "critical" for a in gaps(bp_sys=190, bp_dia=110).values())
    t = gaps(bp_sys=150, bp_dia=95, conditions="Hypertension")
    assert any("not at target" in a["detail"] for a in t.values())
    u = gaps(bp_sys=150, bp_dia=95)
    assert any("no hypertension diagnosis" in a["detail"] for a in u.values())
    assert not gaps(bp_sys=118, bp_dia=76)
    assert c.bp_status(124, 82)[0] == "elevated" and c.bp_status(128, 78)[0] == "borderline" and c.bp_status(110, 70)[0] == "normal"


def test_diabetes_control_thresholds():
    assert any("poorly controlled" in t for t in gaps(conditions="Type 2 Diabetes", hba1c=9.4, medications="Metformin 500 mg daily"))
    assert any("above the general <7% target" in t for t in gaps(conditions="Type 2 Diabetes", hba1c=7.8, medications="Metformin 500 mg daily"))
    assert not any("HbA1c" in t for t in gaps(conditions="Type 2 Diabetes", hba1c=6.5, medications="Metformin 500 mg daily"))
    assert any("No HbA1c recorded" in t for t in gaps(conditions="Type 2 Diabetes", hba1c=0, medications="Metformin 500 mg daily"))
    assert any("diabetes range, no diagnosis recorded" in t for t in gaps(hba1c=6.8))
    assert any("prediabetes" in t for t in gaps(hba1c=6.0))
    assert any("No glucose-lowering medicine" in t for t in gaps(conditions="Type 2 Diabetes", hba1c=7.5, medications="Atorvastatin 20 mg"))


def test_lipid_and_statin_gaps():
    assert any("very high" in t for t in gaps(ldl=195))
    assert any("no statin recorded" in t for t in gaps(ldl=150, age=55, conditions="Type 2 Diabetes", medications="Metformin 500 mg daily"))
    assert not any("no statin" in t for t in gaps(ldl=150, age=55, conditions="Type 2 Diabetes", medications="Atorvastatin 40 mg nightly"))
    assert any("above goal" in t for t in gaps(ldl=120, conditions="Type 2 Diabetes", medications="Atorvastatin 40 mg nightly; Metformin 500 mg daily"))


def test_other_gaps():
    assert any("Atrial fibrillation with no anticoagulant" in t for t in gaps(conditions="Atrial Fibrillation"))
    assert not any("anticoagulant" in t for t in gaps(conditions="Atrial Fibrillation", medications="Warfarin 5 mg daily"))
    assert any("BMI 33" in t for t in gaps(bmi=33.0)) and any("Current smoker" in t for t in gaps(smoker=1))
    assert gaps(smoker=1, conditions="COPD")["Current smoker"]["severity"] == "warning"
    assert any("regular medicines" in t for t in gaps(medications="Paracetamol 1 g; Omeprazole 20 mg; Atorvastatin 20 mg; Lisinopril 5 mg; Amlodipine 5 mg"))
    assert any("Last visit" in t for t in gaps(conditions="Hypertension", last_visit="2024-10-01"))
    assert not any("Last visit" in t for t in gaps(conditions="Hypertension", last_visit="2026-06-01"))


def test_vital_status_helpers():
    assert c.a1c_status(8.4)[0] == "elevated" and c.a1c_status(9.5)[0] == "high" and c.a1c_status(5.2)[0] == "normal"
    assert c.ldl_status(185)[0] == "elevated" and c.ldl_status(195)[0] == "high" and c.ldl_status(90)[0] == "normal"
    assert c.bmi_status(32)[0] == "elevated" and c.bmi_status(17)[0] == "low" and c.hr_status(110)[0] == "high" and c.hr_status(55)[0] == "low"


def test_summary_is_structured_for_the_ui():
    s = c.summarize(pt(conditions="Hypertension", medications="Lisinopril 10 mg daily", allergies="Penicillin", bp_sys=150, bp_dia=95, hba1c=0), TODAY)
    assert s["patient"]["full_name"] == "Test Patient" and s["conditions"] == ["Hypertension"] and s["allergies"] == ["Penicillin"]
    assert s["medications"][0]["name"] == "Lisinopril" and "ace" in s["medications"][0]["classes"]
    keys = {v["key"]: v for v in s["vitals"]}
    assert keys["bp"]["status"] == "high" and "hba1c" not in keys                              # no HbA1c recorded: not shown as 0
    assert s["alerts"] and all({"severity", "category", "title", "detail", "action"} <= set(a) for a in s["alerts"])


# ---------------------------------------------------------------- chat routing
def ans(q, p=None):
    return c.answer(q, p)


def test_clinical_questions_are_routed_and_ordinary_ones_are_not():
    p = pt(conditions="Chronic Kidney Disease", medications="Lisinopril 10 mg daily")
    r = ans("Can I prescribe ibuprofen for this patient?", p)
    assert r and r["kind"] == "medcheck" and "Not recommended" in r["text"]
    assert ans("is it ok to start naproxen", p)["kind"] == "medcheck"
    assert ans("Does warfarin interact with ibuprofen?")["kind"] == "pair"
    assert ans("ibuprofen and warfarin together?")["kind"] == "pair"
    assert ans("Summarize this patient", p)["kind"] == "review" and ans("any care gaps for him?", p)["kind"] == "review"
    assert ans("Is his blood pressure at target?", pt(bp_sys=150, bp_dia=95))["kind"] == "target"


def test_dosing_and_general_questions_are_left_to_the_normal_pipeline():
    p = pt()
    for q in ("What is the usual adult dose of metformin?", "How many mg of ibuprofen can I take in a day?", "What are the red flags of dengue?",
              "What causes tension headaches?", "What is this patient's HbA1c?", "What medications is she on?", "Is it safe to take ibuprofen with kidney disease?"):
        assert ans(q, p) is None, q
    assert ans("Can I give ibuprofen?") is None                                              # a single-drug safety check needs a patient


def test_a_question_about_a_condition_the_patient_lacks_is_general_not_a_patient_check():
    healthy = pt(conditions="Migraine")
    assert ans("Is it safe to take ibuprofen with kidney disease?", healthy) is None
    kidney = pt(conditions="Chronic Kidney Disease")
    r = ans("Is it safe to take ibuprofen with kidney disease?", kidney)                  # this patient HAS it: now it is about them
    assert r and r["kind"] == "medcheck" and "Not recommended" in r["text"]


def test_two_named_drugs_are_always_an_interaction_check_even_with_a_patient():
    r = ans("Can I give ibuprofen with warfarin?", pt(medications="Lisinopril 10 mg daily"))
    assert r["kind"] == "pair" and "Warfarin" in r["text"] and "Ibuprofen" in r["text"]


def test_a_medcheck_without_a_patient_does_not_invent_one():
    assert ans("Should I prescribe warfarin?") is None


def test_target_answers_are_precise():
    t = ans("Is her BP at target?", pt(bp_sys=124, bp_dia=82, conditions="Hypertension"))["text"]
    assert "above target" in t and "130/80" in t
    t = ans("Is her BP at target?", pt(bp_sys=118, bp_dia=76))["text"]
    assert "at target" in t and "above" not in t.split("Goal")[0]
    t = ans("Is his HbA1c at target?", pt(hba1c=8.4, conditions="Type 2 Diabetes"))["text"]
    assert "8.4%" in t and "below 7%" in t
    assert "no HbA1c" in ans("Is his HbA1c controlled?", pt(hba1c=0, conditions="Type 2 Diabetes"))["text"]


# ---------------------------------------------------------------- through the real answer pipeline and the real database
def run(coro):
    return asyncio.run(coro)


def test_the_pipeline_answers_clinical_safety_questions_before_the_record_lookup():
    from app import patients, rag
    p = patients.get("P0000002")
    if not p:
        pytest.skip("patient database not built")
    r = run(rag.answer("Can I prescribe ibuprofen for this patient?", "P0000002"))
    assert r.tier == "clinical" and r.structured["kind"] == "medcheck" and r.structured["drug"] == "ibuprofen"
    assert any(s["source"] == "Clinical rules" for s in r.sources) and r.confidence == 1.0
    r = run(rag.answer("What medications is she on?", "P0000002"))
    assert r.tier == "record"                                                                # plain lookups keep their exact-record path
    r = run(rag.answer("Does warfarin interact with ibuprofen?", None))
    assert r.tier == "clinical" and "bleeding" in r.answer


def test_every_real_patient_chart_can_be_summarised_without_error():
    from app import db, patients
    if not patients.count():
        pytest.skip("patient database not built")
    rows = db.conn().execute("SELECT patient_id FROM patients ORDER BY RANDOM() LIMIT 300").fetchall()
    seen = set()
    for r in rows:
        s = c.summarize(patients.get(r["patient_id"]))
        seen.update(a["category"] for a in s["alerts"])
        assert s["vitals"] and s["patient"]["patient_id"] == r["patient_id"]
    assert {"blood pressure", "diabetes", "interaction"} <= seen or len(seen) >= 3             # the rules actually fire on realistic charts


def test_every_phrasing_of_a_chart_review_is_routed_including_the_ui_chip():
    p = pt(conditions="Hypertension", medications="Lisinopril 10 mg daily")
    for q in ("Review this patient's medications and chart for problems", "any issues with his meds?", "medication review please", "Summarize this patient",
              "any problems with her medication?", "check her medicines", "what care gaps does he have?", "any interactions among his medicines?"):
        r = ans(q, p)
        assert r and r["kind"] == "review", q
    for q in ("What medications is she on?", "What is her last visit?", "What are her allergies?"):
        assert ans(q, p) is None, q                                                     # plain lookups still use the exact record
