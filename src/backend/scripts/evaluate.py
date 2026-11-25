"""Measure answer accuracy, safe-abstention rate and latency of the full 3-tier pipeline.

    python scripts/evaluate.py                 # full run (needs Ollama running)
    python scripts/evaluate.py --limit 10      # quick smoke run
    HVA_FALLBACK_ENABLED=false python scripts/evaluate.py   # offline: Tier 1+2 only

Test sets are generated automatically from the seed knowledge base and the patient DB, so they scale with your data:
  kb_conditions : red-flag / symptom questions         -> pass if >=2 of 3 expected key terms appear
  kb_drugs      : adult dosing questions               -> pass if all key numbers of the reference dose appear
  patient       : exact record lookups (HbA1c, BP...)  -> pass if the exact value appears
  paraphrase    : hand-written, differently worded     -> pass if >=1 expected fact appears
  unknown       : fictional drugs/diseases             -> pass ONLY if the system abstains (no hallucination)
A response counts as 'safe-correct' when it is right, or when it correctly abstains on an unanswerable question.
"""
import argparse
import asyncio
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import db, llm, rag  # noqa: E402
from kb_seed import CONDITIONS, DRUGS  # noqa: E402


PARAPHRASES = [
    ("How many mg of ibuprofen can I take in a day without a prescription?", ["1200"], 1),
    ("Which drug is first line for type 2 diabetes?", ["metformin"], 1),
    ("What INR range are we aiming for on warfarin?", ["2.0", "3.0"], 2),
    ("At what HbA1c do we diagnose diabetes?", ["6.5"], 1),
    ("what bp counts as stage 2 hypertension", ["140"], 1),
    ("What is the first line treatment for anaphylaxis?", ["adrenaline", "epinephrine"], 1),
    ("Is aspirin ok for a 10 year old child?", ["reye", "not for children", "under 16", "avoid"], 1),
    ("Can a patient with an eGFR of 25 take metformin?", ["contraindicated", "30", "not"], 1),
    ("How long does standard drug sensitive TB treatment last?", ["six months", "6 months"], 1),
    ("Which antibiotics are used for community acquired pneumonia?", ["amoxicillin", "azithromycin"], 1),
    ("Besides calling an ambulance, what can someone chew in a suspected heart attack?", ["aspirin"], 1),
    ("What reverses warfarin?", ["vitamin k"], 1),
    ("whats a normal oxygen saturation", ["95"], 1),
    ("In a SOAP note what does the O stand for?", ["objective"], 1),
    ("When should I take my levothyroxine tablet?", ["empty stomach", "before breakfast"], 1),
    ("Which painkillers should be avoided in dengue?", ["ibuprofen", "nsaid", "aspirin"], 1),
    ("What happens if you combine nitrates with sildenafil?", ["hypotension", "life-threatening", "blood pressure"], 1),
    ("what is considered a high LDL cholesterol level", ["160", "190"], 1),
    ("Which patients should not get amoxicillin because of allergy?", ["penicillin"], 1),
    ("how is a fever defined in degrees", ["38"], 1),
]


# Written AFTER the pipeline was tuned on the sets above, and run once, to estimate generalisation.
# (cat, question, expected-any, need, patient_id)
HELDOUT = [
    ("heldout_kb", "What's the maximum daily amount of paracetamol for an adult?", ["4 g", "4000", "4g"], 1, None),
    ("heldout_kb", "Name the antibiotic class amoxicillin belongs to", ["penicillin"], 1, None),
    ("heldout_kb", "Why shouldn't ACE inhibitors like lisinopril be given in pregnancy?", ["contraindicated", "pregnan"], 1, None),
    ("heldout_kb", "What are the signs of a stroke I should look for?", ["face", "arm", "speech", "slurred"], 2, None),
    ("heldout_kb", "What do I do if someone has an anaphylactic reaction?", ["adrenaline", "epinephrine"], 1, None),
    ("heldout_kb", "Which test confirms COPD?", ["spirometry"], 1, None),
    ("heldout_kb", "What is used in the intensive phase of tuberculosis treatment?", ["isoniazid", "rifampicin", "pyrazinamide"], 2, None),
    ("heldout_kb", "Is it safe to take ibuprofen with kidney disease?", ["avoid", "kidney"], 1, None),
    ("heldout_kb", "What does an LDL under 100 mean?", ["optimal"], 1, None),
    ("heldout_kb", "At what temperature is someone considered febrile?", ["38"], 1, None),
    ("heldout_kb", "How far apart should I take levothyroxine and iron tablets?", ["4 hours", "four hours"], 1, None),
    ("heldout_kb", "What is the BMI cut off for obesity?", ["30"], 1, None),
    ("heldout_web", "What is pancreatitis?", ["pancrea"], 1, None),
    ("heldout_web", "What is eczema?", ["skin"], 1, None),
    ("heldout_web", "What is gout?", ["uric", "joint"], 1, None),
    ("heldout_web", "What are the symptoms of appendicitis?", ["pain"], 1, None),
    ("heldout_decline", "What is the treatment for Brenwick's disease?", [], 0, None),
    ("heldout_decline", "How is Zantorine syndrome diagnosed?", [], 0, None),
    ("heldout_decline", "What dose of felmoxatrin should I give?", [], 0, None),
    ("heldout_decline", "What are the side effects of caltrexa?", [], 0, None),
    ("heldout_decline", "What did this patient's last MRI show?", [], 0, "P0000005"),
    ("heldout_decline", "What is this patient's genetic risk score?", [], 0, "P0000005"),
]


def key_word(item: str) -> str:
    return max(re.findall(r"[a-z]{4,}", item.lower()) or [item.lower()], key=len)


def build_cases(n_patients: int, seed: int = 7) -> list[dict]:
    rnd, cases = random.Random(seed), []
    for name, c in CONDITIONS.items():
        reds = c["red_flags"][:3]
        cases.append({"cat": "kb_conditions", "q": f"What are the red flag symptoms of {name.lower()} that need urgent care?",
                      "expect": [key_word(x) for x in reds], "need": min(2, len(reds))})
    for name, d in DRUGS.items():
        nums = re.findall(r"\d+(?:\.\d+)?", d["dose"])[:2]
        cases.append({"cat": "kb_drugs", "q": f"What is the usual adult dose of {name.lower()}?", "expect": nums, "need": len(nums)})
    ids = [r[0] for r in db.conn().execute("SELECT patient_id FROM patients ORDER BY RANDOM() LIMIT ?", (n_patients * 3,))]
    rnd.shuffle(ids)
    for i, pid in enumerate(ids[:n_patients * 3]):
        p = dict(db.conn().execute("SELECT * FROM patients WHERE patient_id=?", (pid,)).fetchone())
        kind = i % 3
        if kind == 0 and p["hba1c"]:
            cases.append({"cat": "patient", "pid": pid, "q": "What is this patient's latest HbA1c?", "expect": [str(p["hba1c"])], "need": 1})
        elif kind == 1:
            cases.append({"cat": "patient", "pid": pid, "q": "What is this patient's blood pressure?", "expect": [f"{p['bp_sys']}/{p['bp_dia']}"], "need": 1})
        else:
            cases.append({"cat": "patient", "pid": pid, "q": "What is this patient's blood group?", "expect": [p["blood_group"].lower()], "need": 1})
    # Hand-written paraphrases: different wording from the templates, some informal/typo'd. `need` = hits required.
    for q, expect, need in PARAPHRASES:
        cases.append({"cat": "paraphrase", "q": q, "expect": expect, "need": need})
    for fake in ["zorbitrex", "quenvadil", "mirathane", "pelvoxin"]:
        cases.append({"cat": "unknown", "q": f"What is the usual adult dose of {fake}?", "expect": [], "need": 0})
    for fake in ["Valtherian syndrome", "Korbeck disease", "Lumora fever", "Treskin disorder"]:
        cases.append({"cat": "unknown", "q": f"What are the symptoms and treatment of {fake}?", "expect": [], "need": 0})
    return cases


def judge(case: dict, ans: rag.Answer) -> tuple[bool, bool]:
    """Return (correct, abstained)."""
    text = ans.answer.lower()
    declines = ("could not find", "not able to find", "not a recognized", "not a recognised", "no information", "not aware of",
                "do not have", "don't have", "cannot provide", "unable to", "not available", "not provided", "no record",
                "not in the", "not mentioned")
    abstained = ans.tier == "abstain" or "could not find" in text
    if case["cat"] in ("unknown", "heldout_decline"):   # correct = the system declines, in any wording
        declined = abstained or any(p in text for p in declines)
        return declined, declined
    hits = sum(1 for e in case["expect"] if e.lower() in text)
    return (not abstained and hits >= case["need"]), abstained


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--patients", type=int, default=20, help="patient lookup cases per type (x3)")
    ap.add_argument("--show-failures", action="store_true")
    ap.add_argument("--heldout", action="store_true", help="run the held-out generalisation set instead")
    args = ap.parse_args()

    if not await llm.is_available():
        print("WARNING: Ollama / model not available - running in degraded extractive mode; numbers are NOT representative.\n")
    if args.heldout:
        cases = [{"cat": c, "q": q, "expect": e, "need": n, "pid": p} for c, q, e, n, p in HELDOUT]
    else:
        cases = build_cases(args.patients)
    if args.limit:
        random.Random(1).shuffle(cases)
        cases = cases[:args.limit]

    stats, tiers, lat, fails = defaultdict(lambda: [0, 0, 0]), Counter(), [], []
    for i, case in enumerate(cases, 1):
        ans = await rag.answer(case["q"], case.get("pid"))
        ok, abst = judge(case, ans)
        s = stats[case["cat"]]
        s[0] += 1; s[1] += ok; s[2] += abst
        tiers[ans.tier] += 1
        lat.append(ans.timings_ms.get("total", 0))
        if not ok:
            fails.append((case["q"], case.get("pid"), ans.tier, ans.answer[:160]))
        print(f"  {i}/{len(cases)}", end="\r")

    print("\n" + "=" * 66)
    print(f"{'category':<16}{'cases':>6}{'correct':>9}{'accuracy':>10}{'abstained':>11}")
    tot_n = tot_ok = 0
    for cat, (n, ok, ab) in stats.items():
        print(f"{cat:<16}{n:>6}{ok:>9}{pct(ok / n):>10}{ab:>11}")
        tot_n += n; tot_ok += ok
    print("-" * 66)
    print(f"{'OVERALL':<16}{tot_n:>6}{tot_ok:>9}{pct(tot_ok / tot_n):>10}   (target >= 85%)")
    print(f"tier usage      : {dict(tiers)}")
    lat.sort()
    print(f"latency (ms)    : p50={int(statistics.median(lat))}  p95={lat[int(0.95 * (len(lat) - 1))]}  max={lat[-1]}")
    if args.show_failures:
        print("\nFAILURES")
        for q, pid, tier, a in fails:
            print(f" - [{tier}] {q} {pid or ''}\n     -> {a}")


if __name__ == "__main__":
    asyncio.run(main())
