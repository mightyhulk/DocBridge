import asyncio
import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import auth, clinical, consult, db, dialogues, doctors, history, llm, medllm, patients, rag, safety, stt, triage, tts, vectorstore
from .config import PROJECT_DIR, get_settings

log = logging.getLogger("hva")
CONSULT_COOKIE = "hva_consult"


async def require_key(x_api_key: str = Header(default="")) -> None:
    key = get_settings().api_key
    if key and x_api_key != key:
        raise HTTPException(401, "Invalid or missing X-API-Key")


def _warmup() -> None:
    """Warm up local indexes in non-serverless environments."""
    try:
        vectorstore.count()
        if stt.available():
            stt.preload()
        tts.engine_name()
    except Exception as e:
        log.warning("warmup incomplete: %s", e)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_schema()
    if not os.environ.get("VERCEL"):
        asyncio.get_running_loop().run_in_executor(None, _warmup)
    yield


app = FastAPI(title="Vitalis AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins, allow_methods=["*"], allow_headers=["*"])
api = APIRouter(prefix="/api", dependencies=[Depends(require_key)])


# ---------- schemas ----------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    patient_id: str | None = None
    history: list[dict] = []
    speak: bool = True


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


# ---------- helpers ----------
async def _speak(text: str) -> dict:
    wav, engine = await asyncio.to_thread(tts.synthesize, text)
    return {"audio_b64": base64.b64encode(wav).decode() if wav else None, "tts_engine": engine}


async def _chat(req: ChatRequest) -> dict:
    db.init_schema()
    ans = await rag.answer(req.message, req.patient_id, req.history)
    out = asdict(ans)
    db.audit("api", "chat", {"tier": ans.tier, "q": safety.mask_pii(req.message), "patient": bool(req.patient_id)})
    if req.speak:
        out.update(await _speak(ans.answer))
    return out


# ---------- routes ----------
@api.get("/health")
async def health():
    db.init_schema()
    return {
        "status": "ok",
        "llm": {"model": get_settings().llm_model, "provider": "huggingface", "available": await llm.is_available()},
        "vector_db": {"documents": await asyncio.to_thread(vectorstore.count)},
        "patients": await asyncio.to_thread(patients.count),
        "stt": {"model": get_settings().whisper_model, "available": stt.available()},
        "tts": {"engine": await asyncio.to_thread(tts.engine_name)},
        "fallback_enabled": get_settings().fallback_enabled,
        "med_llm": await medllm.status(),
        "dialogues": {"cases": await asyncio.to_thread(dialogues.count)},
    }


@api.post("/chat", dependencies=[Depends(auth.require_user)])
async def chat(req: ChatRequest):
    return await _chat(req)


@api.post("/stt", dependencies=[Depends(auth.require_user)])
async def speech_to_text(audio: UploadFile = File(...)):
    data = await audio.read()
    if not data:
        raise HTTPException(400, "Empty audio")
    suffix = "." + (audio.filename or "a.webm").rsplit(".", 1)[-1]
    try:
        return await asyncio.to_thread(stt.transcribe, data, suffix)
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@api.post("/tts", dependencies=[Depends(auth.require_user)])
async def text_to_speech(req: TtsRequest):
    wav, _ = await asyncio.to_thread(tts.synthesize, req.text)
    if wav is None:
        raise HTTPException(503, "No server-side TTS engine available; use browser speech synthesis")
    return Response(wav, media_type="audio/wav")


# ---------- Patient consult (see consult.py) ----------
class ConsultMessage(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    speak: bool = True
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    place: str | None = Field(default=None, max_length=120)
    defer_answer: bool = True


class ConsultAnswer(BaseModel):
    question: str | None = Field(default=None, max_length=2000)
    speak: bool = True


class ConsultLocation(BaseModel):
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    place: str | None = Field(default=None, max_length=120)


def _consult_session(sid: str, request: Request | None = None) -> consult.Session:
    s = consult.get(sid)
    if not s and request is not None:
        raw = request.cookies.get(CONSULT_COOKIE)
        data = auth._unsign(raw) if raw else None
        if data and data.get("id") == sid:
            s = consult.restore_session(data)
    if not s:
        raise HTTPException(404, "Unknown consult session")
    return s


def _with_consult_cookie(payload: dict, s: consult.Session, request: Request) -> JSONResponse:
    consult.save_session(s)
    resp = JSONResponse(payload)
    snap = consult.serialize_session(s)
    snap["exp"] = int(time.time()) + 3600 * 6
    token = auth._sign(snap)
    if len(token) <= 3800:
        resp.set_cookie(CONSULT_COOKIE, token, max_age=3600 * 6, httponly=True, samesite="lax", secure=auth.is_secure_request(request))
    return resp


async def _set_location(s: consult.Session, lat, lon, place) -> None:
    """Remember where the patient is. Coordinates are rounded (~1 km) right here, so nothing finer is ever stored or sent."""
    if lat is not None and lon is not None:
        la, lo = doctors.round_coord(lat), doctors.round_coord(lon)
        same = s.location is not None and (s.location["lat"], s.location["lon"]) == (la, lo)
        s.location = {"lat": la, "lon": lo, "label": s.location["label"] if same else await doctors.reverse_geocode(la, lo)}
    elif place and place.strip():
        try:
            g = await doctors.geocode(place.strip())
        except Exception:
            g = None
        if not g:
            raise HTTPException(404, f"Could not find '{place.strip()}'. Try a city, area or PIN/ZIP code.")
        s.location = {"lat": doctors.round_coord(g["lat"]), "lon": doctors.round_coord(g["lon"]), "label": g["label"]}
    consult.save_session(s)


async def _doctors_for(s: consult.Session) -> dict | None:
    if not s.assessment:
        return None
    if not s.location:
        return {"needs_location": True, "results": [], "specialty": s.assessment.specialty}
    _, tokens = triage.pick_specialty(s.facts.symptom_text or s.facts.text, s.facts.age)
    emergency = s.assessment.level == "emergency"
    found = await doctors.nearby(s.location["lat"], s.location["lon"], tokens, emergency=emergency)
    return {**found, "needs_location": False, "location": s.location,
            "specialty": "Emergency department" if emergency else s.assessment.specialty}


@api.post("/consult/start", dependencies=[Depends(auth.require_user)])
async def consult_start(request: Request, speak: bool = True):
    s, q = consult.start()
    payload = {"session_id": s.id, "stage": "interview", "message": q, "question": q, "progress": 0.0,
               **(await _speak(q) if speak else {})}
    return _with_consult_cookie(payload, s, request)


@api.post("/consult/{sid}/message", dependencies=[Depends(auth.require_user)])
async def consult_message(sid: str, req: ConsultMessage, request: Request):
    s = _consult_session(sid, request)
    if req.lat is not None or req.place:
        await _set_location(s, req.lat, req.lon, req.place)
    out = await consult.handle(s, req.text, with_answer=not req.defer_answer)
    ans = out.get("answer") or {}
    db.audit("api", "consult", {"stage": out["stage"], "level": (out.get("assessment") or {}).get("level"),
                                "source": ans.get("source"), "q": safety.mask_pii(req.text)[:200]})
    out["doctors_pending"] = bool(out["stage"] == "advice" and out.get("assessment") and s.location)
    out["location"] = s.location
    if req.speak and out["message"]:
        out.update(await _speak(safety.strip_for_speech(out["message"])))
    out["session_id"] = s.id
    return _with_consult_cookie(out, s, request)


@api.post("/consult/{sid}/answer", dependencies=[Depends(auth.require_user)])
async def consult_answer(sid: str, req: ConsultAnswer, request: Request):
    """The model-backed half of a consult turn: explanation from the three knowledge sources."""
    s = _consult_session(sid, request)
    try:
        out = await consult.respond(s, req.question or "")
    except ValueError:
        raise HTTPException(409, "Finish the interview first")
    db.audit("api", "consult_answer", {"source": out["answer"]["source"], "flags": out["answer"]["flags"]})
    if req.speak:
        out.update(await _speak(safety.strip_for_speech(out["message"])))
    return _with_consult_cookie(out, s, request)


@api.post("/consult/{sid}/location", dependencies=[Depends(auth.require_user)])
async def consult_location(sid: str, req: ConsultLocation, request: Request):
    s = _consult_session(sid, request)
    if req.lat is None and not req.place:
        raise HTTPException(422, "Send lat/lon or a place name")
    await _set_location(s, req.lat, req.lon, req.place)
    return _with_consult_cookie({"location": s.location, "doctors": await _doctors_for(s)}, s, request)


@api.get("/geocode", dependencies=[Depends(auth.require_user)])
async def geocode_place(q: str):
    """Typed place -> coordinates (rounded), so a location can be chosen before a consultation starts."""
    try:
        g = await doctors.geocode(q.strip())
    except Exception:
        g = None
    if not g:
        raise HTTPException(404, f"Could not find '{q.strip()}'. Try a city, area or PIN/ZIP code.")
    return {"lat": doctors.round_coord(g["lat"]), "lon": doctors.round_coord(g["lon"]), "label": g["label"]}


@api.get("/consult/{sid}/doctors", dependencies=[Depends(auth.require_user)])
async def consult_doctors(sid: str, request: Request):
    return await _doctors_for(_consult_session(sid, request))


@api.get("/patients/search")
async def patient_search(q: str):
    db.init_schema()
    return await asyncio.to_thread(patients.search, q)


@api.get("/patients/{patient_id}")
async def patient_get(patient_id: str):
    db.init_schema()
    p = await asyncio.to_thread(patients.get, patient_id)
    if not p:
        raise HTTPException(404, "Patient not found")
    db.audit("api", "patient_view", {"patient": patient_id})
    return p


@api.get("/patients/{patient_id}/summary")
async def patient_summary(patient_id: str):
    """Chart snapshot for the Clinical Assistant: vitals with status, parsed medicines, and alerts."""
    db.init_schema()
    p = await asyncio.to_thread(patients.get, patient_id)
    if not p:
        raise HTTPException(404, "Patient not found")
    db.audit("api", "patient_summary", {"patient": patient_id})
    return clinical.summarize(p)


@api.get("/clinical/medcheck")
async def clinical_medcheck(patient_id: str, drug: str):
    """Is it safe to add `drug` for this patient?"""
    db.init_schema()
    p = await asyncio.to_thread(patients.get, patient_id)
    if not p:
        raise HTTPException(404, "Patient not found")
    found = clinical.find_drugs(drug)
    if not found:
        return {"known": False, "message": f"'{drug}' is not in the curated drug list. Ask a pharmacist, or check a full interaction reference."}
    res = clinical.medication_check(p, found[0])
    db.audit("api", "medcheck", {"patient": patient_id, "drug": found[0], "verdict": res["verdict"]})
    return {"known": True, **res, "text": clinical.render_medcheck(p, res)}


@api.get("/clinical/interaction")
async def clinical_interaction(a: str, b: str):
    fa, fb = clinical.find_drugs(a), clinical.find_drugs(b)
    if not fa or not fb:
        return {"known": False, "message": "One of the drugs is not in the curated list. Check a full interaction reference."}
    res = clinical.pair_check(fa[0], fb[0])
    return {"known": True, **res, "text": clinical.render_pair(res)}


app.include_router(auth.router)
app.include_router(history.router, dependencies=[Depends(require_key)])
app.include_router(api)

_DIST = PROJECT_DIR / "src" / "frontend" / "dist"
if (_DIST / "index.html").exists():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="ui")
elif not os.environ.get("VERCEL"):
    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": "Vitalis AI API", "ui": "http://localhost:5173 (run `npm run dev` in frontend/)",
                "api_docs": "/docs", "health": "/api/health"}
