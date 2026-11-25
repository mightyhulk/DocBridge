"""Unit tests for safeguards and routing logic (no LLM, models or network needed)."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "backend"))

from app import rag, safety, vectorstore  # noqa: E402


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fixed_kb_vocabulary(monkeypatch):
    """Keep the routing tests independent of whatever is currently indexed."""
    monkeypatch.setattr(rag.vectorstore, "vocabulary", lambda: {"asthm", "parac", "malar", "levot", "metfo", "warfa", "dengu"})


def test_confident_llm_cannot_describe_an_unverifiable_condition(monkeypatch):
    """Regression: a 3B model once invented symptoms/treatment for the made-up 'Valtherian syndrome'."""
    async def llm_up():
        return True

    async def hallucinate(*a, **k):
        return {"answer": "Valtherian syndrome causes seizures.", "confidence": 0.97, "needs_reference": False}

    async def no_web(*a, **k):
        return []
    monkeypatch.setattr(rag.llm, "is_available", llm_up)
    monkeypatch.setattr(rag.llm, "chat_json", hallucinate)
    monkeypatch.setattr(rag.vectorstore, "search", lambda q, k=None: [])
    monkeypatch.setattr(rag.fallback, "search", no_web)
    a = asyncio.run(rag.answer("What are the symptoms and treatment of Valtherian syndrome?"))
    assert a.tier == "abstain" and any(f.startswith("unverified_term") for f in a.flags)


def test_emergency_detected_for_live_situation():
    assert safety.emergency_flags("I have crushing chest pain right now")
    assert safety.emergency_flags("my father collapsed and is unresponsive")


def test_informational_question_is_not_emergency():
    assert safety.emergency_flags("What are the red flag symptoms of stroke like slurred speech?") == []


def test_pii_masking_and_search_query():
    t = safety.mask_pii("Patient P0001234 John Smith phone +91 9876543210 dob 12/03/1980", names=["John Smith"])
    assert "P0001234" not in t and "9876543210" not in t and "John" not in t and "Smith" not in t
    q = safety.to_search_query("What is the dose of metformin for John Smith P0001234?", names=["John Smith"])
    assert "metformin" in q and "P0001234" not in q and "John" not in q


def test_grounding_rejects_passage_about_a_different_entity():
    from app import grounding
    warfarin = vectorstore.Passage(text="Warfarin dose and dosing: individualised to the INR", source="kb", title="Warfarin", score=0.78)
    assert not grounding.covers("What is the usual adult dose of zorbitrex?", warfarin)
    assert grounding.covers("What is the usual adult dose of warfarin?", warfarin)
    dengue = vectorstore.Passage(text="Dengue Fever red flags - when to seek urgent care: bleeding gums", source="kb", title="Dengue Fever", score=0.8)
    assert grounding.covers("What are the red flag symptoms of dengue fever that need urgent care?", dengue)


def test_grounding_requires_rare_entities_but_tolerates_everyday_words():
    from app import grounding
    dengue = vectorstore.Passage(text="Dengue Fever symptoms: high fever; headache. Avoid aspirin and NSAIDs.", source="kb", title="Dengue Fever", score=0.8)
    assert not grounding.covers("What are the symptoms and treatment of Lumora fever?", dengue)       # unknown entity
    assert grounding.covers("Which painkillers should be avoided in dengue?", dengue)                 # paraphrase
    ibu = vectorstore.Passage(text="Ibuprofen dose: over-the-counter maximum 1200 mg per day.", source="kb", title="Ibuprofen", score=0.8)
    assert grounding.covers("How many mg of ibuprofen can I take in a day without a prescription?", ibu)
    assert not grounding.covers("How many mg of metphormin can I take?", ibu)                         # typo'd drug => safe miss


def test_passage_that_merely_mentions_the_entity_is_not_an_answer():
    from app import grounding
    furosemide = vectorstore.Passage(text="Furosemide side effects: Low potassium, dehydration, gout, dizziness.", source="kb", title="Furosemide", score=0.7)
    assert not grounding.covers("What is gout?", furosemide)                   # gout only mentioned in passing
    gout = vectorstore.Passage(text="Gout: a painful joint disease caused by uric acid crystals.", source="web", title="Gout", score=0.7)
    assert grounding.covers("What is gout?", gout)
    ldl = vectorstore.Passage(text="LDL cholesterol targets: LDL below 100 mg/dL is optimal.", source="kb", title="LDL cholesterol targets", score=0.8)
    assert grounding.covers("What does an LDL under 100 mean?", ldl)           # 3-letter acronym is a subject term


def test_patient_specific_question_outside_record_never_goes_to_web(monkeypatch):
    async def llm_up():
        return True

    async def unsure(*a, **k):
        return {"answer": "unknown", "confidence": 0.1, "needs_reference": True}

    async def boom(*a, **k):
        raise AssertionError("web must not be queried for patient-specific questions")
    from app import patients
    p = {"full_name": "A B", "patient_id": "P0000005", "age": 40, "sex": "M", "blood_group": "O+", "city": "X", "conditions": "",
         "medications": "", "allergies": "", "smoker": 0, "bp_sys": 120, "bp_dia": 80, "heart_rate": 70, "bmi": 24.0, "hba1c": 5.2,
         "ldl": 100, "last_visit": "2026-01-01", "recent_encounters": []}
    monkeypatch.setattr(patients, "get", lambda pid: p)
    monkeypatch.setattr(rag.llm, "is_available", llm_up)
    monkeypatch.setattr(rag.llm, "chat_json", unsure)
    monkeypatch.setattr(rag.vectorstore, "search", lambda q, k=None: [])
    monkeypatch.setattr(rag.fallback, "search", boom)
    a = asyncio.run(rag.answer("What did this patient's last MRI show?", "P0000005"))
    assert a.tier == "abstain" and "not_in_record" in a.flags


def test_numbers_must_come_from_sources():
    srcs = ("Levothyroxine: take 30 to 60 minutes before breakfast; separate from iron by 4 hours.",)
    assert rag.ungrounded_numbers("Separate from iron by 4 hours [1].", *srcs) == []
    assert rag.ungrounded_numbers("Take it at least 2 hours apart from iron.", *srcs) == ["2"]
    assert rag.ungrounded_numbers("Maximum is 2,550 mg", "Usual maximum 2000 to 2550 mg per day") == []


def test_invented_number_in_grounded_answer_is_replaced_by_the_reference(monkeypatch):
    async def llm_up():
        return True

    async def invents(*a, **k):
        return {"supported": True, "answer": "Separate them by at least 2 hours [1].", "confidence": 0.9}
    # (a pair the curated interaction table covers, e.g. levothyroxine + iron, is now answered deterministically and never reaches the model;
    #  this test exercises the model's invented-number guard, so it uses one the table does not know)
    p = vectorstore.Passage(text="Levothyroxine and calcium: separate by 4 hours.", source="kb", title="Levothyroxine", score=0.9)
    monkeypatch.setattr(rag.llm, "is_available", llm_up)
    monkeypatch.setattr(rag.llm, "chat_json", invents)
    monkeypatch.setattr(rag.vectorstore, "search", lambda q, k=None: [p])
    a = asyncio.run(rag.answer("How far apart should I take levothyroxine and calcium?"))
    assert "4 hours" in a.answer and any(f.startswith("numeric_mismatch") for f in a.flags)


def test_tier1_answer_with_unverifiable_number_is_rejected(monkeypatch):
    async def llm_up():
        return True

    async def confident(*a, **k):
        return {"answer": "Fever is above 37.5 degrees.", "confidence": 0.95, "needs_reference": False}

    async def no_web(*a, **k):
        return []
    monkeypatch.setattr(rag.llm, "is_available", llm_up)
    monkeypatch.setattr(rag.llm, "chat_json", confident)
    monkeypatch.setattr(rag.vectorstore, "search", lambda q, k=None: [])
    monkeypatch.setattr(rag.fallback, "search", no_web)
    a = asyncio.run(rag.answer("When is a person febrile?"))
    assert a.tier != "llm"


def test_topic_query_keeps_meaningful_digits_and_drops_filler():
    from app import fallback
    assert fallback.topic_query("What causes tension headaches?") == "tension headaches"
    assert fallback.topic_query("What is the difference between type 1 and type 2 diabetes?") == "type 1 type 2 diabetes"
    assert fallback.topic_query("What are the symptoms of hepatitis B?") == "hepatitis B"
    assert fallback.topic_query("What does a high CRP level mean?") == "CRP"
    assert fallback.topic_query("What are the warning signs of colon cancer?") == "colorectal cancer"
    q = fallback.topic_query("Rahul's HbA1c for P0000012 call +91 9876543210", names=["Rahul Sharma"])
    assert "Rahul" not in q and "P0000012" not in q and "9876543210" not in q


def test_label_section_follows_the_question():
    from app import fallback
    assert fallback.label_fields("What are the side effects of prednisone?")[0][0] == "adverse_reactions"
    assert fallback.label_fields("Can I take ibuprofen with prednisone?")[0][0] == "drug_interactions"
    assert fallback.label_fields("Is clopidogrel safe in pregnancy?")[0][0] == "pregnancy"
    assert fallback.label_fields("What is the dose of ciprofloxacin?")[0][0] == "dosage_and_administration"


def test_off_topic_pages_are_dropped_but_named_drug_labels_are_kept():
    from app import fallback
    gout = vectorstore.Passage(text="Gout is a form of arthritis.", source="MedlinePlus", title="Gout", score=0.7)
    stones = vectorstore.Passage(text="Kidney stones are hard deposits.", source="MedlinePlus", title="Kidney Stones", score=0.7)
    label = vectorstore.Passage(text="Warfarin interactions ...", source="openFDA", title="Warfarin - Interactions", score=0.8, direct=True)
    kept = fallback._merge(["kidney", "stones"], [[gout, stones, label]])
    titles = [p.title for p in kept]
    assert "Gout" not in titles and "Kidney Stones" in titles and "Warfarin - Interactions" in titles


def test_authoritative_excerpt_is_returned_when_model_cannot_answer(monkeypatch):
    async def llm_up():
        return True

    async def declines(*a, **k):
        return {"supported": False, "answer": "I could not find this in the references.", "confidence": 0.1}

    async def web(*a, **k):
        return [vectorstore.Passage(text="Tension headaches are due to tight muscles in the neck. They often relate to stress.",
                                    source="MedlinePlus (NIH)", title="Headache", score=0.7)]
    monkeypatch.setattr(rag.llm, "is_available", llm_up)
    monkeypatch.setattr(rag.llm, "chat_json", declines)
    monkeypatch.setattr(rag.vectorstore, "search", lambda q, k=None: [])
    monkeypatch.setattr(rag.fallback, "search", web)
    a = asyncio.run(rag.answer("What causes tension headaches?"))
    assert a.tier == "web" and "verbatim_excerpt" in a.flags and "tight muscles" in a.answer


def test_placeholder_answers_are_rejected():
    assert not rag._usable("<text>") and not rag._usable("") and rag._usable("Metformin is first-line.")
    assert not rag._usable("Gout is a form of arthritis caused by过高尿酸水平导致晶体在关节内沉积。")   # code-switching


def test_tier0_record_lookup():
    from app import patients
    p = {"full_name": "A B", "patient_id": "P0000001", "hba1c": 7.4, "bp_sys": 142, "bp_dia": 88, "blood_group": "O+",
         "medications": "Metformin 500 mg twice daily", "allergies": "", "heart_rate": 70, "bmi": 25.0, "ldl": 100,
         "conditions": "", "last_visit": "2026-01-01", "age": 40, "dob": "1986-01-01", "smoker": 0}
    out = patients.answer_from_record("What is this patient's HbA1c and blood pressure?", p)
    assert "7.4%" in out and "142/88" in out
    assert patients.answer_from_record("What is the normal range for HbA1c?", p) is None     # general knowledge -> not the record
    assert "none known" in patients.answer_from_record("Does she have any allergies?", p)


def test_dosing_detection():
    assert safety.is_dosing_query("What is the maximum dose of paracetamol?")
    assert not safety.is_dosing_query("What causes asthma?")


def test_speech_cleanup():
    assert safety.strip_for_speech("**Take** it [1] daily\n- with food") == "Take it daily with food"


def test_follow_up_detection_does_not_hijack_new_questions():
    hist = [{"role": "user", "content": "What causes tension headaches?"}]
    assert rag._retrieval_query("What is pancreatitis?", hist) == "What is pancreatitis?"
    assert "headaches" in rag._retrieval_query("and how is it treated?", hist)


def test_rag_abstains_when_nothing_found(monkeypatch):
    async def llm_up():
        return True

    async def fake_chat_json(*a, **k):
        return {"answer": "x", "confidence": 0.2, "needs_reference": True, "supported": False}

    async def no_web(*a, **k):
        return []
    monkeypatch.setattr(rag.llm, "is_available", llm_up)
    monkeypatch.setattr(rag.llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(rag.vectorstore, "search", lambda q, k=None: [])
    monkeypatch.setattr(rag.fallback, "search", no_web)
    a = asyncio.run(rag.answer("What is the dose of zorbitrex?"))
    assert a.tier == "abstain" and a.confidence == 0.0


def test_rag_accepts_confident_tier1(monkeypatch):
    async def llm_up():
        return True

    async def fake_chat_json(*a, **k):
        return {"answer": "Asthma is a chronic airway disease.", "confidence": 0.95, "needs_reference": False}
    monkeypatch.setattr(rag.llm, "is_available", llm_up)
    monkeypatch.setattr(rag.llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(rag.vectorstore, "search", lambda q, k=None: [])
    a = asyncio.run(rag.answer("What is asthma?"))
    assert a.tier == "llm"


def test_dosing_questions_never_trusted_to_tier1(monkeypatch):
    async def llm_up():
        return True

    async def fake_chat_json(*a, **k):
        return {"answer": "500 mg", "confidence": 0.99, "needs_reference": False}
    p = vectorstore.Passage(text="Paracetamol dose: 500 to 1000 mg", source="kb", title="Paracetamol", score=0.9)
    monkeypatch.setattr(rag.llm, "is_available", llm_up)
    monkeypatch.setattr(rag.llm, "chat_json", fake_chat_json)
    async def no_web(*a, **k):
        return []
    monkeypatch.setattr(rag.vectorstore, "search", lambda q, k=None: [p])
    monkeypatch.setattr(rag.fallback, "search", no_web)
    a = asyncio.run(rag.answer("What is the dose of paracetamol?"))
    assert a.tier != "llm"


def test_doses_are_quoted_verbatim_without_llm(monkeypatch):
    async def llm_down():
        return False
    p = vectorstore.Passage(text="Levothyroxine dose and dosing: Full replacement is about 1.6 micrograms per kg per day.",
                            source="kb", title="Levothyroxine", score=0.9)
    monkeypatch.setattr(rag.llm, "is_available", llm_down)
    monkeypatch.setattr(rag.vectorstore, "search", lambda q, k=None: [p])
    a = asyncio.run(rag.answer("What is the usual adult dose of levothyroxine?"))
    assert "1.6 micrograms per kg" in a.answer and "verbatim_reference_dose" in a.flags


def test_strong_kb_match_takes_precedence_over_model_memory(monkeypatch):
    calls = []

    async def llm_up():
        return True

    async def fake_chat_json(msgs, **k):
        calls.append(msgs[0]["content"][:40])
        return {"supported": True, "answer": "Grounded answer [1]", "confidence": 0.9}
    p = vectorstore.Passage(text="Malaria red flags: confusion or seizures; jaundice", source="kb", title="Malaria", score=0.85)
    monkeypatch.setattr(rag.llm, "is_available", llm_up)
    monkeypatch.setattr(rag.llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(rag.vectorstore, "search", lambda q, k=None: [p])
    a = asyncio.run(rag.answer("What are the red flag symptoms of malaria?"))
    assert a.tier == "rag" and len(calls) == 1                 # one grounded call; Tier 1 skipped


def test_speech_to_text_drops_hallucinations():
    from app import stt
    assert stt.looks_like_a_loop("I have to say that I have to say that I have to say that")
    assert not stt.looks_like_a_loop("I have a fever and I have a cough")
    assert stt.clean_transcript([("I am having severe headache.", -0.35, 0.03)]) == "I am having severe headache."
    assert stt.clean_transcript([("Thank you very much.", -0.9, 0.1)]) == ""                # stock phrase
    assert stt.clean_transcript([("Okay.", -0.6, 0.05)]) == "Okay."                          # a real one-word answer
    assert stt.clean_transcript([("I'm not going to do this", -1.4, 0.38)]) == ""            # low confidence guess
    assert stt.clean_transcript([("my head hurts", -0.4, 0.01), ("blah blah", -1.5, 0.7)]) == "my head hurts"
