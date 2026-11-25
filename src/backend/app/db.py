"""SQLite access. One connection per thread; auto-seeds sample patients on fresh/serverless environments."""
import json
import sqlite3
import threading
from datetime import datetime, timezone

from .config import PROJECT_DIR, get_settings

_local = threading.local()
_seeded = False
_seed_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    dob TEXT, age INTEGER, sex TEXT, blood_group TEXT, city TEXT,
    phone TEXT, insurance_id TEXT,
    conditions TEXT, medications TEXT, allergies TEXT,
    smoker INTEGER, bp_sys INTEGER, bp_dia INTEGER, heart_rate INTEGER,
    bmi REAL, hba1c REAL, ldl INTEGER, last_visit TEXT
);
CREATE TABLE IF NOT EXISTS encounters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL, visit_date TEXT, reason TEXT, diagnosis TEXT, notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_enc_patient ON encounters(patient_id);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT, action TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS consult_sessions (
    id TEXT PRIMARY KEY, updated_at REAL, data_json TEXT NOT NULL
);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS patients_fts USING fts5(
    full_name, content='patients', content_rowid='rowid', tokenize='unicode61'
);
"""


def conn() -> sqlite3.Connection:
    c = getattr(_local, "conn", None)
    if c is None:
        s = get_settings()
        s.db_path.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(s.db_path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        _local.conn = c
    return c


def _seed_sample_patients(c: sqlite3.Connection) -> None:
    """Populate synthetic sample patients when starting with an empty DB (e.g. Vercel serverless cold start)."""
    sample = PROJECT_DIR / "sample-data" / "patients.sample.json"
    if not sample.exists():
        return
    if c.execute("SELECT COUNT(*) FROM patients").fetchone()[0] > 0:
        return
    try:
        rows = json.loads(sample.read_text(encoding="utf-8")).get("patients", [])
    except Exception:
        return
    if not rows:
        return
    for p in rows:
        c.execute(
            "INSERT OR IGNORE INTO patients VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p["patient_id"], p["full_name"], p.get("dob"), p.get("age"), p.get("sex"), p.get("blood_group"),
             p.get("city"), p.get("phone"), p.get("insurance_id"), p.get("conditions", ""), p.get("medications", ""),
             p.get("allergies", ""), p.get("smoker", 0), p.get("bp_sys"), p.get("bp_dia"), p.get("heart_rate"),
             p.get("bmi"), p.get("hba1c"), p.get("ldl"), p.get("last_visit")),
        )
        for e in p.get("encounters", []):
            c.execute(
                "INSERT INTO encounters(patient_id, visit_date, reason, diagnosis, notes) VALUES (?,?,?,?,?)",
                (p["patient_id"], e.get("visit_date"), e.get("reason"), e.get("diagnosis"), e.get("notes")),
            )
    c.execute("INSERT INTO patients_fts(patients_fts) VALUES('rebuild')")
    c.commit()


def init_schema() -> None:
    global _seeded
    c = conn()
    c.executescript(SCHEMA)
    c.executescript(FTS_SQL)
    c.commit()
    if not _seeded:
        with _seed_lock:
            if not _seeded:
                _seed_sample_patients(c)
                _seeded = True


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def audit(actor: str, action: str, detail: dict | str = "") -> None:
    """Append-only audit trail. Callers must pass already-masked detail (see safety.mask_pii)."""
    if not isinstance(detail, str):
        detail = json.dumps(detail, ensure_ascii=False)
    c = conn()
    c.execute("INSERT INTO audit_log(ts, actor, action, detail) VALUES (?,?,?,?)",
              (now_iso(), actor, action, detail[:2000]))
    c.commit()
