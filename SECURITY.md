# Security

Vitalis AI is a **decision-support prototype built on synthetic data**. It is not a medical device and is not hardened for exposure to the
public internet. This document says what protection exists, where the gaps are, and how to report a problem.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Contact the project owner privately, with steps to reproduce, the affected
file or endpoint and the impact. Expect an acknowledgement within a few days. Do not include real patient data in a report.

## Security model

| Area | What is in place |
|---|---|
| **Sign-in** | Google OAuth 2.0 authorization-code flow, server side (`src/backend/app/auth.py`). The client secret never reaches the browser. A random `state` value in an HttpOnly cookie protects the callback against CSRF and replay. Only a verified Google email is accepted. |
| **Session** | A signed cookie (HMAC-SHA256, HttpOnly, SameSite=Lax, `Secure` when served over HTTPS), valid for 7 days, with an expiry inside the signed payload. The signing key is `HVA_SESSION_SECRET` or a random key generated once into `data/session.key`. Tampered or expired cookies are rejected. |
| **Authorization** | Patient routes (`/api/chat`, `/api/stt`, `/api/tts`, `/api/voice/chat`, `/api/consult/*`, `/api/geocode`) and `/api/history` return 401 unless signed in. Saved conversations are keyed to the Google account id and are never returned to another account (covered by tests). |
| **Secrets** | Kept in `config/.env`, which is git-ignored. `config/.env.example` contains no secrets. |
| **API key (optional)** | Setting `HVA_API_KEY` requires an `X-API-Key` header on every `/api` request. |
| **Privacy of outbound requests** | The online tiers send only anonymised topic keywords (`fallback.topic_query` removes names, ids, phone numbers, emails and dates; the selected patient's name is removed). Nearby-doctor search sends a location rounded to about 1 km and a speciality, never symptoms. Questions about "this patient" are answered from the record and never go to the web. |
| **Model safety** | Emergency phrases short-circuit to "call emergency services" without a model. Medicine doses come from a fixed table, never a model. Prescription-only medicines are never suggested. A model's invented numbers are replaced by the source's own words. |
| **Local processing** | Speech-to-text, text-to-speech, the vector indexes, the patient database and the language models run on your own machines. |
| **Audit** | An append-only audit log records actions with PII masked (`db.audit`). |

## Data handled and where it lives

| Data | Location | Notes |
|---|---|---|
| Synthetic patient records | `data/patients.sqlite` | Fictional. No real person. |
| Saved conversations (patient consults and Clinical Assistant chats) | `data/patients.sqlite`, table `consult_history` | **Not encrypted.** A Clinical Assistant chat can contain details from a patient's chart. Treat the file as sensitive and protect the disk. |
| Google profile (name, email, picture) | Only inside the signed session cookie | Not stored on the server. |
| Vector indexes | `data/chroma`, `data/chroma_dialogues` | Public reference text and public datasets. |

## Known limitations (read before any real deployment)

1. **No built-in HTTPS.** Run behind a TLS reverse proxy for anything beyond localhost and set `HVA_PUBLIC_URL` to the https address.
2. **Raspberry Pi mic relay (`/api/remote/*`) is not tied to a signed-in user.** It is protected only by the optional API key and your network. Keep it on a trusted LAN.
3. **No rate limiting or lockout.** Add one at the proxy.
4. **No role separation.** Any signed-in Google account can open the Clinical Assistant. A real deployment needs clinician-only access control and an allow-list of accounts.
5. **History is stored unencrypted** (see above) and has no retention policy beyond manual deletion.
6. **Third-party services** (Google sign-in, DuckDuckGo search, MedlinePlus, openFDA, OpenStreetMap, Hugging Face downloads) see your IP address and the anonymised queries above.
7. The severity and medicine rules are a transparent screen, not a validated clinical tool. A licensed clinician must review all output.

## Operating guidance

- **Rotate any credential that has been shared in chat, email or screenshots.** For Google, create a new client secret in Cloud Console and update `config/.env`.
- Register only the redirect URIs you actually use.
- Keep `config/.env`, `data/` and `logs/` out of version control and backups that leave your control.
- Keep dependencies updated (`pip list --outdated`, `npm audit`).
- Never load real patient data without the approvals your jurisdiction requires (for example HIPAA, GDPR or India's DPDP Act).
