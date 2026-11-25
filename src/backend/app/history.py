"""A signed-in patient's past consultations. Each one is saved as a snapshot (transcript + latest assessment) under their
Google account id and is only ever returned to that same account."""
import json
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from . import auth, db

router = APIRouter(prefix="/api/history")
_ID = re.compile(r"^[A-Za-z0-9_-]{6,40}$")
_ready = False


def _c():
    global _ready
    c = db.conn()
    if not _ready:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS consult_history (
                id TEXT NOT NULL, owner TEXT NOT NULL, title TEXT, level TEXT, created_at TEXT, updated_at TEXT, data_json TEXT,
                PRIMARY KEY (owner, id)
            );
            CREATE INDEX IF NOT EXISTS idx_history_owner ON consult_history(owner, updated_at DESC);
        """)
        try:
            c.execute("ALTER TABLE consult_history ADD COLUMN kind TEXT DEFAULT 'consult'")      # older databases predate the Clinical Assistant
        except Exception:
            pass
        c.commit()
        _ready = True
    return c


def owner_of(user: dict | None) -> str:
    return (user or {}).get("sub") or (user or {}).get("email") or "demo"      # demo = sign-in switched off


# --- Titles: a saved consultation is named after the SYMPTOMS the patient gave, never after a reply like "No" -------------------
_FILLER = re.compile(
    r"^(?:(?:no|nope|nah|yes|yeah|yep|ok|okay|sure|same|hello|hi|hey|thanks?|thank you|never|none|nothing|uhm+|um+|hmm+|bye(?:-bye)?|"
    r"good (?:morning|day|evening)|have a good day|no changes?|(?:no,? )?(?:i )?(?:do not|don'?t) (?:want|need) to change(?: anything)?|"
    r"(?:yes,? )?i have your details|i have to change .*|i (?:do not|don'?t) know(?: how to change)?)\W*)+$", re.I)
_NOTED = re.compile(r"^Noted:\s*(.+?)(?:\.\s|\.$|$)", re.S)                         # "Noted: severity 10/10, a headache. How long..."
_TOLD = re.compile(r"what you told me [^:]{0,40}:\s*(.+?)(?:\.\s|\.$|$)", re.S)      # "...what you told me earlier today: headache, ..."
_NEG = re.compile(r"^(?:no|not|never|without|denies)\b", re.I)


def _is_filler(t: str | None) -> bool:
    s = re.sub(r"\s+", " ", t or "").strip()
    return not s or s.lower() == "consultation" or len(s.split()) < 2 and len(s) < 4 or bool(_FILLER.match(s))


def _clip(s: str, n: int = 48) -> str:
    s = re.sub(r"\s+", " ", s).strip().rstrip(".!?…")
    if len(s) > n:
        cut = s[:n]
        s = cut[:max(cut.rfind(" "), 20)].rstrip(" ,;") + "…"
    return s[:1].upper() + s[1:]


def derive_title(log: list[dict], result: dict | None = None, current: str | None = None) -> str:
    """Prefer what the patient actually said about the problem; otherwise the symptoms the assistant noted from their answers;
    otherwise the reason behind the assessment."""
    if current and not _is_filler(current):
        return current                                                             # the patient's own title, stored as typed
    for m in log:                                                                  # the symptoms the assistant understood from their answers
        if m.get("role") != "assistant":
            continue
        txt = str(m.get("content", ""))
        mt = _NOTED.match(txt) or _TOLD.search(txt)
        if not mt:
            continue
        parts = [p.strip() for p in re.split(r",\s*", mt.group(1)) if p.strip() and not _NEG.match(p.strip())]
        parts = [re.sub(r"^(?:an?|the)\s+", "", p, flags=re.I) for p in parts]
        parts.sort(key=lambda p: bool(re.match(r"^severity\b", p, re.I)))                           # symptoms first, the score last
        symptoms = [p for p in parts if not re.match(r"^(?:for|since|about|going on)\b", p, re.I)]   # a duration alone is not a symptom
        if any(not re.match(r"^severity\b", p, re.I) for p in symptoms):                              # ...nor is a severity score alone
            return _clip(", ".join(symptoms[:3]))
    for m in [m for m in log if m.get("role") == "user"][:4]:                     # otherwise their own words, from the start of the chat
        t = str(m.get("content", ""))
        if len(t.split()) >= 3 and not _is_filler(t):
            return _clip(t)
    reasons = ((result or {}).get("assessment") or {}).get("reasons") or []
    if reasons:
        return _clip(re.sub(r"^(?:an?|the)\s+", "", re.sub(r"^[^:]{0,40}warning sign:\s*", "", str(reasons[0]), flags=re.I), flags=re.I))
    return "Consultation"


class Snapshot(BaseModel):
    title: str = Field(max_length=120)
    level: str | None = None
    log: list[dict] = Field(max_length=200)
    result: dict | None = None
    progress: float = 0.0
    kind: str = Field("consult", pattern="^(consult|assistant)$")     # which tab the conversation belongs to
    patient: dict | None = None                                      # assistant only: the patient that was selected


@router.get("")
async def list_history(user: dict | None = Depends(auth.require_user)):
    rows = _c().execute("SELECT id, title, level, updated_at, COALESCE(kind, 'consult') AS kind, data_json FROM consult_history WHERE owner=? ORDER BY updated_at DESC LIMIT 100",
                        (owner_of(user),)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        raw = d.pop("data_json", None)
        if d["kind"] == "consult" and _is_filler(d["title"]):          # saved by an older version under a title like "No"
            try:
                data = json.loads(raw or "{}")
                d["title"] = derive_title(data.get("log") or [], data.get("result"))
            except Exception:
                pass
        out.append(d)
    return out


@router.get("/{hid}")
async def get_one(hid: str, user: dict | None = Depends(auth.require_user)):
    r = _c().execute("SELECT * FROM consult_history WHERE owner=? AND id=?", (owner_of(user), hid)).fetchone()
    if not r:
        raise HTTPException(404, "Consultation not found")
    return {"id": r["id"], "title": r["title"], "level": r["level"], "updated_at": r["updated_at"], "kind": r["kind"] or "consult", **json.loads(r["data_json"])}


@router.put("/{hid}")
async def save(hid: str, snap: Snapshot, user: dict | None = Depends(auth.require_user)):
    if not _ID.match(hid):
        raise HTTPException(400, "Bad id")
    data = json.dumps({"log": snap.log, "result": snap.result, "progress": snap.progress, "patient": snap.patient}, ensure_ascii=False)
    if len(data) > 400_000:
        raise HTTPException(413, "Consultation too large to save")
    now = db.now_iso()
    c = _c()
    c.execute("""INSERT INTO consult_history(id, owner, title, level, created_at, updated_at, data_json, kind) VALUES (?,?,?,?,?,?,?,?)
                 ON CONFLICT(owner, id) DO UPDATE SET title=excluded.title, level=excluded.level, updated_at=excluded.updated_at,
                 data_json=excluded.data_json, kind=excluded.kind""", (hid, owner_of(user), derive_title(snap.log, snap.result, snap.title) if snap.kind == "consult" else snap.title,
                                                                              snap.level, now, now, data, snap.kind))
    c.commit()
    return {"ok": True}


@router.delete("/{hid}")
async def delete(hid: str, user: dict | None = Depends(auth.require_user)):
    c = _c()
    c.execute("DELETE FROM consult_history WHERE owner=? AND id=?", (owner_of(user), hid))
    c.commit()
    return {"ok": True}
