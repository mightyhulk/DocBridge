"""Generate 200,000 SYNTHETIC patient records (no real data) into SQLite.

    python scripts/generate_patients.py            # 200k patients (~1 min)
    python scripts/generate_patients.py --n 5000   # smaller dev set

Deterministic (seeded) so evaluation questions are reproducible. Clinically plausible, not clinically real.
"""
import argparse
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import db  # noqa: E402

FIRST_M = "Aarav Vivaan Aditya Arjun Rohan Karan Rahul Amit Vikram Suresh Ramesh Rajesh Sanjay Manoj Anil Deepak Nikhil Pranav Siddharth Harsh James John Robert Michael William David Daniel Thomas Mohammed Ali Omar Chen Wei Hiroshi Carlos Luis".split()
FIRST_F = "Aadhya Ananya Diya Isha Kavya Meera Neha Pooja Priya Riya Sneha Anjali Divya Lakshmi Sunita Rekha Swati Nandini Mary Patricia Jennifer Linda Elizabeth Susan Sarah Karen Fatima Aisha Mei Yuki Maria Sofia Elena".split()
LAST = "Sharma Verma Patel Singh Kumar Gupta Reddy Nair Iyer Das Sarangi Mishra Panda Behera Mohanty Rao Joshi Mehta Shah Khan Ahmed Smith Johnson Williams Brown Jones Garcia Miller Davis Wilson Chen Wang Tanaka Kim Lopez Martinez".split()
CITIES = "Bhubaneswar Cuttack Mumbai Delhi Bengaluru Hyderabad Chennai Kolkata Pune Ahmedabad Jaipur Lucknow Kochi Bhopal Patna".split()
BLOOD = ["O+"] * 36 + ["A+"] * 28 + ["B+"] * 22 + ["AB+"] * 6 + ["O-"] * 2 + ["A-"] * 2 + ["B-"] * 2 + ["AB-"] * 1
ALLERGIES = ["Penicillin", "Sulfa drugs", "Aspirin", "Ibuprofen", "Latex", "Peanuts", "Shellfish", "Dust mites", "Pollen", "Eggs"]

# condition: (min_age, base_prevalence, [medications])
CONDITIONS = {
    "Hypertension": (30, 0.34, ["Amlodipine 5 mg daily", "Lisinopril 10 mg daily", "Losartan 50 mg daily"]),
    "Type 2 Diabetes": (30, 0.14, ["Metformin 500 mg twice daily", "Metformin 1000 mg twice daily", "Glimepiride 2 mg daily"]),
    "Hyperlipidemia": (35, 0.22, ["Atorvastatin 20 mg nightly", "Atorvastatin 40 mg nightly"]),
    "Asthma": (5, 0.07, ["Salbutamol inhaler as needed", "Budesonide inhaler twice daily"]),
    "COPD": (50, 0.06, ["Tiotropium inhaler daily", "Salbutamol inhaler as needed"]),
    "GERD": (25, 0.12, ["Omeprazole 20 mg daily"]),
    "Hypothyroidism": (25, 0.07, ["Levothyroxine 50 mcg daily", "Levothyroxine 75 mcg daily"]),
    "Osteoarthritis": (45, 0.15, ["Paracetamol 500 mg as needed", "Ibuprofen 400 mg as needed"]),
    "Depression": (18, 0.07, ["Sertraline 50 mg daily"]),
    "Chronic Kidney Disease": (50, 0.06, ["Furosemide 20 mg daily"]),
    "Atrial Fibrillation": (60, 0.05, ["Warfarin 5 mg daily"]),
    "Heart Failure": (60, 0.04, ["Furosemide 40 mg daily", "Lisinopril 5 mg daily"]),
    "Iron-deficiency Anemia": (18, 0.06, ["Ferrous sulfate 200 mg daily"]),
    "Migraine": (15, 0.08, ["Paracetamol 500 mg as needed"]),
}
ACUTE = [("Fever and cough", "Acute upper respiratory infection"), ("Sore throat", "Pharyngitis"),
         ("Burning urination", "Urinary tract infection"), ("Abdominal pain", "Gastroenteritis"),
         ("Back pain", "Low back pain"), ("Headache", "Tension headache"), ("Skin rash", "Contact dermatitis"),
         ("Routine check-up", "No acute findings"), ("Ankle sprain", "Ligament sprain"), ("Diarrhea", "Acute gastroenteritis")]


def make_patient(i: int, rng: random.Random, today: date) -> tuple[tuple, list[tuple]]:
    pid = f"P{i:07d}"
    sex = rng.choice("MF")
    name = f"{rng.choice(FIRST_M if sex == 'M' else FIRST_F)} {rng.choice(LAST)}"
    age = max(1, min(95, int(rng.gauss(42, 20))))
    dob = today - timedelta(days=age * 365 + rng.randint(0, 364))
    conds, meds = [], []
    for c, (min_age, prev, drugs) in CONDITIONS.items():
        if age >= min_age and rng.random() < prev * min(1.6, 0.4 + (age - min_age) / 35):
            conds.append(c)
            meds.append(rng.choice(drugs))
    meds = sorted(set(meds))
    has_dm, has_htn = "Type 2 Diabetes" in conds, "Hypertension" in conds
    bmi = round(max(14.0, min(48.0, rng.gauss(24 + (3 if has_dm else 0), 4.5))), 1)
    sys_ = int(rng.gauss(142 if has_htn else 118, 11)) + (4 if age > 60 else 0)
    dia = int(rng.gauss(88 if has_htn else 76, 7))
    hba1c = round(max(4.5, rng.gauss(7.4, 1.3)), 1) if has_dm else round(max(4.4, rng.gauss(5.3, 0.4)), 1)
    ldl = int(rng.gauss(138 if "Hyperlipidemia" in conds else 102, 24))
    allergies = ", ".join(rng.sample(ALLERGIES, rng.choice([0, 0, 0, 1, 1, 2])))
    last_visit = today - timedelta(days=rng.randint(1, 700))
    row = (pid, name, dob.isoformat(), age, sex, rng.choice(BLOOD), rng.choice(CITIES),
           f"+91-9{rng.randint(100000000, 999999999)}", f"INS{rng.randint(10**8, 10**9 - 1)}",
           ", ".join(conds), "; ".join(meds), allergies, int(rng.random() < (0.18 if age > 18 else 0.0)),
           sys_, dia, int(rng.gauss(76, 10)), bmi, hba1c, ldl, last_visit.isoformat())
    encs = []
    for _ in range(rng.choice([1, 1, 2, 3])):
        d = (today - timedelta(days=rng.randint(1, 900))).isoformat()
        if conds and rng.random() < 0.55:
            c = rng.choice(conds)
            encs.append((pid, d, f"Follow-up for {c}", c, f"Reviewed {c.lower()}; medications continued."))
        else:
            reason, dx = rng.choice(ACUTE)
            encs.append((pid, d, reason, dx, f"Presented with {reason.lower()}; assessed as {dx.lower()}."))
    return row, encs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    db.init_schema()
    c = db.conn()
    c.execute("DELETE FROM patients"); c.execute("DELETE FROM encounters")
    rng, today, t0 = random.Random(args.seed), date(2026, 10, 1), time.time()
    batch_p, batch_e = [], []
    for i in range(1, args.n + 1):
        row, encs = make_patient(i, rng, today)
        batch_p.append(row); batch_e.extend(encs)
        if len(batch_p) >= 10_000:
            c.executemany("INSERT INTO patients VALUES (" + ",".join("?" * 20) + ")", batch_p)
            c.executemany("INSERT INTO encounters(patient_id, visit_date, reason, diagnosis, notes) VALUES (?,?,?,?,?)", batch_e)
            c.commit(); batch_p, batch_e = [], []
            print(f"  {i:>7,} / {args.n:,}", end="\r")
    if batch_p:
        c.executemany("INSERT INTO patients VALUES (" + ",".join("?" * 20) + ")", batch_p)
        c.executemany("INSERT INTO encounters(patient_id, visit_date, reason, diagnosis, notes) VALUES (?,?,?,?,?)", batch_e)
        c.commit()
    c.execute("INSERT INTO patients_fts(patients_fts) VALUES('rebuild')")
    c.commit()
    n_enc = c.execute("SELECT COUNT(*) FROM encounters").fetchone()[0]
    print(f"\nInserted {args.n:,} patients and {n_enc:,} encounters in {time.time() - t0:.0f}s -> {db.get_settings().db_path}")
    for r in c.execute("SELECT patient_id, full_name, age, sex, conditions FROM patients WHERE conditions != '' LIMIT 3"):
        print("  sample:", dict(r))


if __name__ == "__main__":
    main()
