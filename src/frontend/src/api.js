const KEY = import.meta.env.VITE_API_KEY || "";
const headers = (extra = {}) => (KEY ? { "X-API-Key": KEY, ...extra } : extra);

async function handle(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(detail);
  }
  return res.json();
}

const json = (method, path, body) =>
  fetch(`/api${path}`, { method, headers: headers({ "Content-Type": "application/json" }), body: JSON.stringify(body) }).then(handle);

export const api = {
  health: () => fetch("/api/health", { headers: headers() }).then(handle),
  chat: (message, patient_id, history, speak = true) => json("POST", "/chat", { message, patient_id, history, speak }),
  stt: (blob) => {
    const fd = new FormData();
    const ext = blob.type.includes("wav") ? "wav" : blob.type.includes("ogg") ? "ogg" : blob.type.includes("mp4") ? "mp4" : "webm";
    fd.append("audio", blob, `speech.${ext}`);
    return fetch("/api/stt", { method: "POST", headers: headers(), body: fd }).then(handle);
  },
  // Patient consult: interview -> severity -> sourced answer -> nearby specialists
  consultStart: (speak = true) => fetch(`/api/consult/start?speak=${speak}`, { method: "POST", headers: headers() }).then(handle),
  consultMessage: (sid, body) => json("POST", `/consult/${sid}/message`, body),
  consultAnswer: (sid, question, speak = true) => json("POST", `/consult/${sid}/answer`, { question, speak }),
  geocode: (q) => fetch(`/api/geocode?q=${encodeURIComponent(q)}`, { headers: headers() }).then(handle),
  consultDoctors: (sid) => fetch(`/api/consult/${sid}/doctors`, { headers: headers() }).then(handle),
  consultLocation: (sid, body) => json("POST", `/consult/${sid}/location`, body),
  // Clinical Assistant: chart snapshot with alerts, and the medication safety check
  patientSummary: (id) => fetch(`/api/patients/${id}/summary`, { headers: headers() }).then(handle),
  medcheck: (patient_id, drug) => fetch(`/api/clinical/medcheck?patient_id=${encodeURIComponent(patient_id)}&drug=${encodeURIComponent(drug)}`, { headers: headers() }).then(handle),
  searchPatients: (q) => fetch(`/api/patients/search?q=${encodeURIComponent(q)}`, { headers: headers() }).then(handle),
  getPatient: (id) => fetch(`/api/patients/${id}`, { headers: headers() }).then(handle),
};

let current = null;
/** Play server-rendered audio (Piper); fall back to the browser's built-in voice when the server has none. */
export function speak({ audio_b64, text }) {
  stopSpeaking();
  if (audio_b64) {
    current = new Audio(`data:audio/wav;base64,${audio_b64}`);
    current.play().catch(() => {});
  } else if ("speechSynthesis" in window && text) {
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text.replace(/\[\d+\]/g, "")));
  }
}
export function stopSpeaking() {
  if (current) { current.pause(); current = null; }
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
}
/** True while the assistant's voice is playing - the always-on listener must not hear (and answer) itself. */
export function isSpeaking() {
  return !!(current && !current.paused && !current.ended) || ("speechSynthesis" in window && window.speechSynthesis.speaking);
}
// Phones only allow audio after a tap. The first touch unlocks playback so spoken replies (which arrive later, after
// the always-on listener heard a question) are not blocked by the browser.
if (typeof window !== "undefined") {
  const unlock = () => {
    try {
      const a = new Audio("data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=");
      a.play().catch(() => {});
      if ("speechSynthesis" in window) window.speechSynthesis.speak(new SpeechSynthesisUtterance(""));
    } catch {}
    window.removeEventListener("pointerdown", unlock);
  };
  window.addEventListener("pointerdown", unlock);
}

// Google sign-in (patients): the session is an HttpOnly cookie set by the backend
export const history = {
  list: () => fetch("/api/history", { credentials: "same-origin" }).then(handle),
  get: (id) => fetch(`/api/history/${id}`, { credentials: "same-origin" }).then(handle),
  save: (id, snap) => fetch(`/api/history/${id}`, { method: "PUT", credentials: "same-origin", headers: { "Content-Type": "application/json" }, body: JSON.stringify(snap) }).then(handle),
  remove: (id) => fetch(`/api/history/${id}`, { method: "DELETE", credentials: "same-origin" }).then(handle),
};

export const auth = {
  me: () => fetch("/api/auth/me", { credentials: "same-origin" }).then(handle),
  methods: () => fetch("/api/auth/methods", { credentials: "same-origin" }).then(handle),
  localLogin: (name, email) => fetch("/api/auth/local", { method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, email }) }).then(handle),
  logout: () => fetch("/api/auth/logout", { method: "POST", credentials: "same-origin" }).then(handle),
};
