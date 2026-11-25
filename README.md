# Vitalis AI: Patient Consult + Clinical Assistant

Web-based voice and text AI for clinics, powered by **Hugging Face Inference API** and ready to deploy on **Vercel**.

| Capability | What it does |
|---|---|
| **Clinical Assistant** | For clinicians: ask clinical questions by voice or text (doses, guidelines, red flags) and, with a patient selected, get a **chart snapshot with alerts**, a **medication safety check** (allergies, interactions, conditions, age) and **care-gap** flags. |
| **Patient Consult** | A patient describes a problem by voice or text → rule-based interview → rule-based **severity** (`self_care` / `see_doctor` / `urgent` / `emergency`) → sourced answer (Hugging Face model, clinical references, and reputable medical websites) → **nearby clinics/hospitals** via OpenStreetMap. |
| **Website-Only Voice** | Browser microphone capture (`voice.js`) → `/api/stt` (Hugging Face Whisper) and spoken replies via browser `speechSynthesis` (or optional local Piper TTS). |

## Repository Structure

```
api/index.py          Vercel Python serverless entrypoint
vercel.json           Vercel build & routing configuration (Vite SPA + FastAPI /api)
requirements.txt      Lightweight Python dependencies for Vercel serverless
src/
  backend/            FastAPI app (app/), build/evaluation scripts (scripts/), requirements.txt
  frontend/           React + Vite web UI (Patient Consult · Clinical Assistant · Sign-in · History)
tests/                pytest unit tests
config/               .env.example (copy to .env for local dev)
deployment/           README.md (Vercel & local deployment guide)
sample-data/          Synthetic patient records (auto-seeded on startup when DB is empty)
```

## Quick Start (Local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/backend/requirements.txt
cp config/.env.example config/.env   # add your HF_TOKEN

# Run backend
cd src/backend
uvicorn app.main:app --reload --port 8000

# Run frontend (new terminal)
cd src/frontend
npm install
npm run dev   # -> http://localhost:5173
```

## Deploy on Vercel

1. Push this repo to GitHub/GitLab/Bitbucket and import it in Vercel.
2. Add `HF_TOKEN` (and optionally `HVA_SESSION_SECRET`, `AUTH_GOOGLE_ID`, `AUTH_GOOGLE_SECRET`, `HVA_PUBLIC_URL`) in Vercel Environment Variables.
3. Deploy — Vercel builds `src/frontend` and serves `/api/*` via `api/index.py`.

See [deployment/README.md](deployment/README.md) for all environment variables and details.

## Tests

```bash
pytest -q
```
