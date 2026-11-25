# Deployment (Website-Only on Vercel + Hugging Face)

## Architecture

- **Frontend:** React + Vite SPA (`src/frontend/`) built to `src/frontend/dist/`
- **Backend:** FastAPI serverless function (`api/index.py` -> `src/backend/app/main.py`)
- **Models:** Hugging Face Inference Router (`https://router.huggingface.co/v1/chat/completions`) and Hugging Face Whisper STT (`https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo`)
- **Voice & Data:** Browser microphone (`MediaRecorder`) + browser `speechSynthesis`, in-repo clinical seed KB + SQLite sample patients (auto-seeded to `/tmp/vitalis-data` on Vercel), and live medical web/API fallbacks (MedlinePlus, openFDA, OpenStreetMap)

---

## Deploy to Vercel

1. Push the repository to GitHub / GitLab / Bitbucket.
2. Import the repository in the [Vercel Dashboard](https://vercel.com). Vercel reads `vercel.json` automatically:
   - Builds `src/frontend` into `src/frontend/dist`
   - Routes `/api/*` to `api/index.py` (installing from root `requirements.txt`)
3. Add **Environment Variables** in Vercel Project Settings:
   - `HF_TOKEN` *(required for LLM & Whisper STT)* — Hugging Face token with **Make calls to Inference Providers** permission
   - `HVA_LLM_MODEL` *(optional, default `meta-llama/Llama-3.3-70B-Instruct`)*
   - `HVA_MED_MODEL` *(optional, default `google/gemma-4-31B-it`)*
   - `HVA_MED_API_URL` *(optional)* — dedicated Hugging Face Inference Endpoint URL if hosting a gated medical model like `google/medgemma-4b-it`
   - `HVA_SESSION_SECRET` *(recommended)* — random secret string for signing session cookies across serverless instances
   - `AUTH_GOOGLE_ID`, `AUTH_GOOGLE_SECRET`, `HVA_PUBLIC_URL` *(optional)* — for Google sign-in (`<HVA_PUBLIC_URL>/api/auth/google/callback`)
   - `HVA_LOCAL_AUTH=true` *(optional)* — enables simple name + email login on the website if Google OAuth is not configured
4. Click **Deploy**.

---

## Local Development

```bash
# 1. Backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/backend/requirements.txt
cp config/.env.example config/.env   # set HF_TOKEN
cd src/backend && uvicorn app.main:app --reload --port 8000

# 2. Frontend (separate terminal)
cd src/frontend
npm install
npm run dev   # http://localhost:5173
```
