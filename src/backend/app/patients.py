"""Patient record access (SQLite + FTS5). Exact structured data is read via SQL, never guessed by the LLM."""
import re

from . import db

_ID_RE = re.compile(r"P\d{4,}", re.I)


def get(patient_id: str) -> dict | None:
    row = db.conn().execute("SELECT * FROM patients WHERE patient_id=?", (patient_id.upper(),)).fetchone()
    if not row:
        return None
    p = dict(row)
    enc = db.conn().execute(
        "SELECT visit_date, reason, diagnosis, notes FROM encounters WHERE patient_id=? ORDER BY visit_date DESC LIMIT 5",
        (p["patient_id"],)).fetchall()
    p["recent_encounters"] = [dict(e) for e in enc]
    return p


def search(q: str, limit: int = 10) -> list[dict]:
    q = q.strip()
    if not q:
        return []
    if _ID_RE.fullmatch(q):
        rows = db.conn().execute("SELECT patient_id, full_name, age, sex, city FROM patients WHERE patient_id=?",
                                 (q.upper(),)).fetchall()
    else:
        terms = [t for t in re.findall(r"[A-Za-z]+", q)]
        if not terms:
            return []
        match = " ".join(f'"{t}"*' for t in terms)
        rows = db.conn().execute(
            "SELECT p.patient_id, p.full_name, p.age, p.sex, p.city FROM patients_fts f "
            "JOIN patients p ON p.rowid=f.rowid WHERE patients_fts MATCH ? LIMIT ?", (match, limit)).fetchall()
    return [dict(r) for r in rows]


def count() -> int:
    try:
        return db.conn().execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    except Exception:
        return 0


_PATIENT_REF = re.compile(r"\b(this patient|the patient|patient'?s|his|her|their|him|my|latest|current|last|recent|he|she|they)\b", re.I)
_GENERAL = re.compile(r"\b(normal|range|target|threshold|categor|defin|mean|guideline|cutoff|cut-off|stage)\w*", re.I)
_FIELDS = [
    (r"hba1c|\ba1c\b|glycated", lambda p: f"HbA1c is {p['hba1c']}%"),
    (r"blood pressure|\bbp\b", lambda p: f"blood pressure is {p['bp_sys']}/{p['bp_dia']} mmHg"),
    (r"blood (group|type)", lambda p: f"blood group is {p['blood_group']}"),
    (r"heart rate|pulse", lambda p: f"heart rate is {p['heart_rate']} bpm"),
    (r"\bbmi\b|body mass", lambda p: f"BMI is {p['bmi']}"),
    (r"\bldl\b|cholesterol", lambda p: f"LDL cholesterol is {p['ldl']} mg/dL"),
    (r"medication|medicine|prescri|\btaking\b|\bdrugs?\b", lambda p: f"medications: {p['medications'] or 'none recorded'}"),
    (r"allerg", lambda p: f"allergies: {p['allergies'] or 'none known'}"),
    (r"condition|diagnos|medical history|problem list", lambda p: f"conditions: {p['conditions'] or 'none recorded'}"),
    (r"last visit|last seen|recent visit", lambda p: f"last visit was on {p['last_visit']}"),
    (r"\bage\b|how old|date of birth|\bdob\b", lambda p: f"age is {p['age']} (born {p['dob']})"),
    (r"smok", lambda p: f"smoker: {'yes' if p['smoker'] else 'no'}"),
]


def answer_from_record(question: str, p: dict) -> str | None:
    """Tier 0: exact facts about the selected patient come straight from the database, never from model generation."""
    if not _PATIENT_REF.search(question) or _GENERAL.search(question):
        return None
    parts = [fn(p) for pat, fn in _FIELDS if re.search(pat, question, re.I)]
    if not parts:
        return None
    return f"For {p['full_name']} ({p['patient_id']}), " + "; ".join(parts) + "."


def to_context(p: dict) -> str:
    """Compact, LLM-ready rendering of a record (kept short to save tokens/latency)."""
    enc = "; ".join(f"{e['visit_date']}: {e['reason']} -> {e['diagnosis']}" for e in p.get("recent_encounters", [])[:3])
    return (
        f"Patient {p['patient_id']} ({p['full_name']}), {p['age']}y {p['sex']}, blood group {p['blood_group']}, {p['city']}.\n"
        f"Conditions: {p['conditions'] or 'none recorded'}. Medications: {p['medications'] or 'none'}. "
        f"Allergies: {p['allergies'] or 'none known'}. Smoker: {'yes' if p['smoker'] else 'no'}.\n"
        f"Latest vitals/labs: BP {p['bp_sys']}/{p['bp_dia']} mmHg, HR {p['heart_rate']} bpm, BMI {p['bmi']}, "
        f"HbA1c {p['hba1c'] if p['hba1c'] else 'n/a'}%, LDL {p['ldl']} mg/dL. Last visit {p['last_visit']}.\n"
        f"Recent encounters: {enc or 'none'}."
    )
