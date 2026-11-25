"""End-to-end smoke test against a running server:  python scripts/smoke_test.py [http://localhost:8000]"""
import json
import sys
import time

import httpx

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000") + "/api"
c = httpx.Client(timeout=300)


def brief(r: dict) -> str:
    return (f"[{r.get('tier')}] conf={r.get('confidence')} flags={r.get('flags')} sources={[s['source'] for s in r.get('sources', [])][:3]}\n"
            f"      {r['answer'][:260]}")


h = c.get(f"{BASE}/health").json()
print("health:", json.dumps(h))

history: list[dict] = []
questions = [
    ("What are the red flag symptoms of dengue fever?", None),
    ("and how is it treated?", None),                                    # follow-up using history
    ("What causes tension headaches?", None),                            # general knowledge -> Tier 1 or web
    ("What is pancreatitis?", None),                                     # not in KB -> Tier 3 online
    ("What is the usual adult dose of metformin?", None),                # verbatim reference dose
    ("I have crushing chest pain and sweating", None),                   # emergency
    ("What is this patient's HbA1c and blood pressure?", "P0000002"),    # Tier 0
    ("What are the symptoms of Valtherian syndrome?", None),             # fictional -> abstain
]
for q, pid in questions:
    t = time.time()
    r = c.post(f"{BASE}/chat", json={"message": q, "patient_id": pid, "history": history[-6:], "speak": True}).json()
    dt = time.time() - t
    history += [{"role": "user", "content": q}, {"role": "assistant", "content": r["answer"]}]
    print(f"\nQ: {q}  ({dt:.1f}s, tts={r.get('tts_engine')})\n   {brief(r)}")
