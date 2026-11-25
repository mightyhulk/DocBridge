# Demo flow

A 10-minute walkthrough of Vitalis AI. Architecture: [architecture.png](architecture.png). Evidence for every step: [test-evidence/](test-evidence/).

## Before you start (5 minutes)

| Check | How |
|---|---|
| Backend and UI up | `.\deployment\run.ps1`, then open http://localhost:8000 |
| Google sign-in works | The redirect URI `http://localhost:8000/api/auth/google/callback` is registered, and your Google account is a Test user while the consent screen is in Testing mode. |
| Models answer | Ollama is running on the laptop. If the Raspberry Pi is the model host, check the sidebar's "Show system status": medical AI should say `on <pi hostname>`. |
| Pi mic (optional) | `python src/raspberry-pi/pi_mic_agent.py --profile pi` on the Pi, then set the sidebar microphone to "Raspberry Pi". |
| Clean state | Delete old entries from the sidebar History list. |

Open the app in a normal window and keep a second tab ready on `docs/architecture.png`.

## The story

> "Vitalis AI helps a patient work out how urgent a problem is, what is safe to take at home, and which doctor to see. The same system gives clinicians a fast, sourced assistant. It runs on our own machines. Rules decide urgency and doses; the language model only explains."

## Step by step

| # | You do | What the audience should see | Say |
|---|---|---|---|
| 1 | Open the site signed out | Clean sign-in page ([01-login.png](test-evidence/01-login.png)) | "Patients sign in with Google. Nothing about them is stored except their own history." |
| 2 | Click **Continue with Google** | Lands on "Hello, <name>" with the sidebar ([02](test-evidence/02-patient-consult-start.png)) | "Sign-in is server side; the secret never reaches the browser." |
| 3 | Click **I have a fever and a sore throat** (or use the mic) | The first question appears with a one-line *why I am asking* | "It never repeats a question and never asks what you already told it." |
| 4 | Answer: "102 F", "2 days", then "no" to the red-flag questions | Each reply is acknowledged ("Noted: fever of 102°F ...") and the next question changes with your answers | "Questions are chosen by clinical weight, not a fixed script." |
| 5 | When asked about age and conditions say: "I am 39 and I do not have any existing conditions" | No conditions are recorded | "It understands denials, not just keywords." |
| 6 | When asked about allergies say: **"I am not allergic to any medicine."** | Over-the-counter medicines appear, with doses and a referral ([03](test-evidence/03-patient-consult-assessment.png)) | "Negations are handled safely. A denial never blocks a medicine, and an allergy always does." |
| 7 | Read the urgency card and the medicines card | Colour-coded urgency, specialist, label-dose medicines, "see a doctor within 2-3 days" | "Severity and doses are rules, never a model's opinion. Every answer ends with a doctor referral." |
| 8 | Enter a city or press **Use my location** | Nearby specialists with distance, phone, directions | "Only a rounded location and a speciality leave the machine, never the symptoms." |
| 9 | Ask a follow-up: "Is it safe to take ibuprofen?" | Instant medicine-screen answer, then the explanation with its source | "The explanation comes from one of three sources, escalating: the medical model, real doctor-patient cases, then a web search with links." |
| 10 | **Safety moment:** start a new consultation and type "severe chest pain and I cannot breathe" | Red emergency banner, no medicines | "Emergency phrases bypass the model completely." |
| 11 | Switch to **Clinical Assistant**, ask "Does warfarin interact with ibuprofen?" | Answer with a tier tag and sources ([04](test-evidence/04-clinical-assistant-interaction.png)) | "Curated interaction rules, then the model, then references." |
| 12 | Ask "Search the web for the latest ADA guideline on SGLT2 inhibitors in heart failure" | A web answer with page links ([05](test-evidence/05-clinical-assistant-web-search.png)) | "Clinicians can pull in live sources; only anonymised keywords are sent." |
| 13 | Search a patient, then ask "Can I prescribe ibuprofen for this patient?" | Verdict from the patient's allergies, medicines and conditions, plus the chart alerts | "A language model never decides whether a combination is safe." |
| 14 | Point at the sidebar **History** | Both chats are saved; open one and continue ([06](test-evidence/06-history-sidebar.png)) | "Every conversation is saved to your account and private to it." |
| 15 | Show `docs/architecture.png` | The four layers | "Local models and data, anonymised outbound calls only." |

## If something goes wrong

| Problem | What to do |
|---|---|
| Google says `redirect_uri_mismatch` | Register `http://localhost:8000/api/auth/google/callback`, and open the site at `localhost`, not an IP. |
| Google shows "access blocked" | Add your account under Test users on the OAuth consent screen. |
| The explanation is slow (Pi) | Expected: about 1-4 minutes on a Pi 4. The urgency, medicines and doctors appear instantly. Put the laptop first in `HVA_MED_OLLAMA_URLS` for a faster demo. |
| Mic does not respond | Check the sidebar microphone setting. The Pi agent must be running for "Raspberry Pi". |
| No nearby doctors | The public map service can be slow. Click Find again; results are cached once found. |

## What it does not do (say this out loud)

It is a decision-support prototype on synthetic data. It is not a diagnosis and not a medical device. Prescription-only medicines are never suggested.
