"""Diagnose the online fallback: for each question show what each source returned and what the pipeline finally did.

    python scripts/fallback_probe.py
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import fallback, grounding, rag, safety  # noqa: E402

QUESTIONS = [
    "What is pancreatitis?", "What causes tension headaches?", "What is gout?", "What are the symptoms of appendicitis?",
    "How is celiac disease diagnosed?", "What are the side effects of prednisone?", "What is the dose of ciprofloxacin?",
    "Can I take ibuprofen with prednisone?", "What is psoriasis and how is it treated?", "What causes kidney stones?",
    "What are the symptoms of hepatitis B?", "How does insulin work?", "What is the difference between type 1 and type 2 diabetes?",
    "What are the warning signs of colon cancer?", "Is clopidogrel safe in pregnancy?", "What is vertigo?",
    "What vaccines do adults need?", "What does a high CRP level mean?", "What is the treatment for scabies?",
    "How is sleep apnea treated?",
]


async def main() -> None:
    for q in QUESTIONS:
        sq = fallback.topic_query(q)
        t = time.time()
        web = await fallback.search(q)
        dt = time.time() - t
        cov = [grounding.covers(q, p) for p in web]
        t = time.time()
        a = await rag.answer(q)
        print(f"\nQ: {q}\n   search='{sq}' web={[(p.source[:9], p.title[:28]) for p in web]} covers={cov} ({dt:.1f}s)")
        print(f"   -> [{a.tier}] {a.flags} ({time.time() - t:.1f}s) {a.answer[:150]}")


asyncio.run(main())
