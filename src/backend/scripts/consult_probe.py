"""End-to-end probe of the Patient Consult against a RUNNING server (real model, real dialogue index, real web + map).

    python scripts/consult_probe.py [http://localhost:8000] [--place Bhubaneswar] [--only A,C]

Each scenario walks the interview, then prints the severity, the knowledge source that answered (with its links and the
reasons earlier sources were skipped), the specialist, and nearby facilities; then checks the expectation and prints PASS/FAIL.
A Pi-hosted 4B model needs about a minute per answer, so a full run takes several minutes.
"""
import argparse
import sys
import time

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("base", nargs="?", default="http://localhost:8000")
ap.add_argument("--place", default="Bhubaneswar", help="typed location used to look up nearby doctors")
ap.add_argument("--only", default="", help="comma list of scenario ids, e.g. A,C")
args = ap.parse_args()
API = args.base.rstrip("/") + "/api"
c = httpx.Client(timeout=600)

# The doctor chooses its own questions, so each scenario gives an opening sentence, answers keyed on words in the question
# (anything else is answered "no"), and optional follow-up questions asked after the assessment.
SCENARIOS = [
    # id, title, opener, {question-keyword: answer}, follow-ups, expected levels, expected sources, expected specialist (a|b = either) or None
    ("A", "Minor cold -> medical model alone",
     "I have a runny nose and sneezing since yesterday, it is mild 2/10",
     {"how old": "I'm 27 and healthy", "allergic": "none", "sneeze": "yes", "running": "yes"}, [],
     {"self_care"}, {"model", "dialogues", "web"}, "General|ENT"),
    ("B", "Fever for days with diabetes -> evidence (doctor conversations / web), see a doctor",
     "I have had fever and body ache for 6 days and it is getting worse, about 6/10",
     {"temperature": "101", "how old": "I'm 45 and I have diabetes", "allergic": "none"}, [],
     {"see_doctor", "urgent"}, {"dialogues", "web"}, "General"),
    ("C", "Emergency -> no model, hospitals",
     "I have crushing chest pain spreading to my left arm and I am sweating", {}, [],
     {"emergency"}, {"none", None}, "Cardio"),
    ("D", "Knee pain for weeks -> orthopaedics",
     "My right knee has been hurting for three weeks, worse when climbing stairs, about 5/10",
     {"how old": "I'm 52, no other conditions", "allergic": "none"}, [],
     {"see_doctor", "self_care"}, {"model", "dialogues", "web"}, "Orthop"),
    ("E", "Follow-up medical questions after an assessment (risky question -> evidence, never a dose from the model)",
     "I have a sore throat for 2 days, mild 3/10",
     {"how old": "I'm 30, healthy", "allergic": "none"}, ["Is it safe to take antibiotics for a sore throat?"],
     {"self_care", "see_doctor"}, {"dialogues", "web", "model"}, None),
    ("F", "Medicine safety: ibuprofen is refused for an older patient with an ulcer, paracetamol offered, doctor referral last",
     "I have a headache and body pain for 2 days, 5/10",
     {"how old": "I'm 70 and I have a stomach ulcer", "allergic": "none"}, ["Can I take ibuprofen?"],
     {"self_care", "see_doctor"}, {"formulary"}, None),
]

results = []
for sid_, title, opener, rules, follow_ups, levels, sources, spec in SCENARIOS:
    if args.only and sid_ not in args.only.split(","):
        continue
    print(f"\n{'=' * 78}\n[{sid_}] {title}")
    try:
        s = c.post(f"{API}/consult/start", params={"speak": "false"}).json()
        sid = s["session_id"]
        print(f"BOT : {s['message'][:110]}")
        out = None
        instants = []
        asked = []

        def say(text, first=False):
            """Send one patient message; fetch the slow explanation too when the server defers it."""
            global out
            body = {"text": text, "speak": False}
            if first and args.place:
                body["place"] = args.place
            t = time.time()
            r = c.post(f"{API}/consult/{sid}/message", json=body)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            out = r.json()
            instant = time.time() - t
            print(f"YOU : {text}")
            print(f"BOT : {out['message'][:330]}  ({instant:.1f}s)" + (f"\n      why: {out['why']}" if out.get("why") else ""))
            if out.get("pending_question"):                    # the slow half: the medical model's explanation
                t = time.time()
                r2 = c.post(f"{API}/consult/{sid}/answer", json={"question": out["pending_question"], "speak": False})
                if r2.status_code != 200:
                    raise RuntimeError(f"answer HTTP {r2.status_code}: {r2.text[:200]}")
                r2 = r2.json()
                out["answer"] = r2["answer"]
                print(f"BOT : {r2['message'][:420]}  (explanation took {time.time() - t:.0f}s)")
                instants.append(instant)

        say(opener, first=True)
        for _ in range(20):                                     # the doctor decides what to ask; we answer it
            if out["stage"] == "advice":
                break
            q = out["question"].lower()
            asked.append(out["question"])
            say(next((a for k, a in rules.items() if k in q), "no"))
        if out["stage"] != "advice":
            raise RuntimeError("the interview never ended")
        assert len(asked) == len(set(asked)), "a question was asked twice"
        print(f"  ({len(asked)} questions asked, none repeated)")
        for fu in follow_ups:
            say(fu)
        td = time.time()
        docs = c.get(f"{API}/consult/{sid}/doctors").json() or {}              # fetched separately: the map can be slow
        print(f"  (nearby lookup took {time.time() - td:.1f}s)")
        a, ans = out.get("assessment"), out.get("answer") or {}
        print(f"  severity   : {a['level']} - {a['label']}  | specialist: {a['specialty']}")
        for r in a["reasons"][:4]:
            print(f"               - {r}")
        print(f"  source     : {ans.get('source')} ({ans.get('source_label')})  confidence={ans.get('confidence')}  by={ans.get('served_by') or '-'}  {ans.get('timings_ms')}")
        for t in ans.get("trail", []):
            print(f"  trail      : {t}")
        for x in ans.get("sources", [])[:3]:
            print(f"  link       : {x.get('url') or '(model knowledge)'}  [{x.get('source')}]")
        print(f"  doctors    : {len(docs.get('results', []))} within {docs.get('radius_km')} km"
              + (f" (error: {docs.get('error')})" if docs.get("error") else "") + (f" - {docs['location']['label'][:50]}" if docs.get("location") else ""))
        for d in docs.get("results", [])[:3]:
            print(f"               {d['name']} ({d['kind']}, {d['distance_km']} km, {d['match']})")
        checks = {
            "severity": a["level"] in levels,
            "source": ans.get("source") in sources,
            "specialist": spec is None or any(x.lower() in a["specialty"].lower() for x in spec.split("|")) or a["level"] == "emergency",
            "nearby doctors found": bool(docs.get("results")) or bool(docs.get("error")),
            "web answers carry links": ans.get("source") != "web" or all(x.get("url") for x in ans.get("sources", [])),
            "assessment shown instantly (<45s, before the model)": all(x < 45 for x in instants),
            "possible causes listed": out.get("assessment", {}).get("possible_causes") is not None,
            "medicines end with a doctor referral": out["assessment"]["level"] == "emergency" or bool(out.get("medicines", {}).get("referral")),
            "no dose from model": ans.get("source") != "model" or not __import__("re").search(r"\d\s?mg", ans.get("text", "")),
        }
        bad = [k for k, v in checks.items() if not v]
        print(f"  => {'PASS' if not bad else 'FAIL: ' + ', '.join(bad)}")
        results.append((sid_, not bad))
    except Exception as e:
        print(f"  => FAIL: {type(e).__name__}: {e}")
        results.append((sid_, False))

print(f"\n{'=' * 78}\n" + "  ".join(f"{i}:{'PASS' if ok else 'FAIL'}" for i, ok in results))
sys.exit(0 if all(ok for _, ok in results) else 1)
