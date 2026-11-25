import React, { useEffect, useRef, useState } from "react";
import { api, history, speak, stopSpeaking } from "../api.js";
import { pickTitle } from "../historyTitle.js";
import { useVoiceInput } from "../voice.js";
import MicButton from "./MicButton.jsx";

const LEVEL_CLASS = { self_care: "ok", see_doctor: "warn", urgent: "bad", emergency: "bad" };
const SOURCE_CLASS = { model: "info", dialogues: "info", web: "warn", formulary: "ok", none: "bad" };
const SOURCE_SHORT = { model: "1 · Medical AI model", dialogues: "2 · Doctor-patient conversations", web: "3 · Web search",
                       formulary: "Medicine safety check", none: "No reliable source" };

/** Over-the-counter medicines, screened against the patient's age, conditions, allergies and regular medicines. Not a prescription. */
function Medicines({ m }) {
  if (!m || !(m.suggestions?.length || m.avoid?.length || m.note || m.self_care?.length)) return null;
  return (
    <div className="answer-card meds">
      <div className="meta">
        <span className="tag ok">Suggested over-the-counter medicines</span>
        <span className="muted small">not a prescription</span>
      </div>
      {m.note && <div className="banner warn">{m.note}</div>}
      {m.suggestions?.length > 0 && m.dose_guidance && <div className="small"><b>How to use these:</b> {m.dose_guidance}</div>}
      {m.suggestions?.map((s) => (
        <div key={s.key} className="med">
          <div><b>{s.name}</b> <span className="muted small">for {s.use}</span></div>
          <div><b>Dose:</b> {s.dose}</div>
          <div><b>Maximum:</b> {s.maximum}</div>
          {s.tips?.length > 0 && <ul className="trail">{s.tips.map((t, i) => <li key={i}>{t}</li>)}</ul>}
        </div>
      ))}
      {m.avoid?.length > 0 && (
        <details open>
          <summary className="small"><b>Not suggested for you</b> (and why)</summary>
          {m.avoid.map((a) => <div key={a.key} className="small med-avoid"><b>{a.name}</b>: {a.reason}</div>)}
        </details>
      )}
      {m.self_care?.length > 0 && <div className="small"><b>Also helpful:</b> {m.self_care.join(" ")}</div>}
      {m.not_prescribed && <div className="muted small">{m.not_prescribed}</div>}
      <div className="muted small">{m.disclaimer}</div>
      {m.referral && <div className="banner bad"><b>See a doctor:</b> {m.referral}</div>}
    </div>
  );
}
const MATCH_TEXT = {
  verified: "Lists this speciality",
  hospital: "Hospital - usually has this department",
  unverified: "Speciality not listed - call to confirm",
};

function host(url) {
  try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return url; }
}

function Msg({ m, user }) {
  const me = m.role === "user";
  const name = me ? (user?.name || "You") : "Health Assistant";
  return (
    <div className={`msg ${m.role} ${m.emergency ? "emergency" : ""}`}>
      {me ? <span className="av me">{(name[0] || "Y").toUpperCase()}</span> : <span className="av bot">✚</span>}
      <div className="msg-main">
        <div className="msg-h"><b>{name}</b><span className="badge">CONSULT</span>{m.t && <span className="time">{new Date(m.t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>}</div>
        <div className="msg-b">{m.content}</div>
        {m.why && <div className="why">Why I am asking: {m.why}</div>}
      </div>
    </div>
  );
}

function Thinking({ since, text }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const s = Math.max(0, Math.round((now - since) / 1000));
  return (
    <div className="bubble assistant muted thinking">
      <span className="spin" />
      <span>{text} {s}s</span>
    </div>
  );
}

export default function Consult({ voiceOut, health, user = null, saved = null, onSaved = () => {} }) {
  const [sid, setSid] = useState(null);
  const [log, setLog] = useState(saved?.log || []);               // a saved (past) consultation opens read-only
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);               // waiting for the instant (rule-based) part
  const [answering, setAnswering] = useState(false);     // waiting for the Hugging Face medical model's explanation
  const [since, setSince] = useState(0);
  const [progress, setProgress] = useState(saved?.progress ?? 0);
  const [result, setResult] = useState(saved?.result || null);       // latest assessment + answer
  const [docs, setDocs] = useState(null);
  const [docsLoading, setDocsLoading] = useState(false);
  const [loc, setLoc] = useState(null);             // {lat, lon, label} the patient shared (before/after the session starts)
  const [place, setPlace] = useState("");
  const [locErr, setLocErr] = useState("");
  const end = useRef(null);
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth" }); }, [log, busy]);

  const say = (r) => voiceOut && requestAnimationFrame(() => speak({ audio_b64: r.audio_b64, text: r.message }));

  // Every consultation gets a run number. A request still in flight from the previous consultation (e.g. the model is
  // still writing its explanation) must not touch the new one - and must not leave the mic/composer disabled.
  const runRef = useRef(0);

  // Save the consultation to the patient's history once an exchange has finished (not while the model is still working).
  const hidRef = useRef("");
  useEffect(() => {
    if (saved || !hidRef.current || busy || !log.some((m) => m.role === "user")) return;
    const title = pickTitle(log);
    const t = setTimeout(() => {
      history.save(hidRef.current, { title, level: result?.assessment?.level || null, log, result, progress })
        .then(onSaved).catch(() => {});
    }, 500);
    return () => clearTimeout(t);
  }, [log, result, busy, answering]);

  const start = async () => {
    const run = ++runRef.current;
    hidRef.current = (crypto.randomUUID?.() || String(Date.now())).replace(/-/g, "").slice(0, 24);
    stopSpeaking();
    setAnswering(false); setDocsLoading(false); setText("");
    setBusy(true); setSince(Date.now()); setResult(null); setDocs(null); setProgress(0); setLog([]); setSid(null);
    try {
      const r = await api.consultStart(voiceOut);
      if (run !== runRef.current) return;
      setSid(r.session_id);
      setLog([{ t: Date.now(), role: "assistant", content: r.message }]);
      say(r);
      return r.session_id;
    } catch (e) {
      if (run === runRef.current) setLog([{ t: Date.now(), role: "assistant", content: "Could not start: " + e.message }]);
    } finally { if (run === runRef.current) setBusy(false); }
  };

  const send = async (t, sidNow = sid) => {
    t = (t || "").trim();
    if (!t || busy || answering || !sidNow) return;
    const run = runRef.current;
    const live = () => run === runRef.current;                 // false once a new consultation has been started
    setText(""); setBusy(true); setSince(Date.now());
    setLog((l) => [...l, { t: Date.now(), role: "user", content: t }]);
    let pending = null;
    try {
      // 1) instant: severity, specialist, medicines and nearby doctors come from rules + the map, not from the model
      const r = await api.consultMessage(sidNow, { text: t, speak: voiceOut, ...(loc ? { lat: loc.lat, lon: loc.lon } : {}) });
      if (!live()) return;
      if (r.message) setLog((l) => [...l, { t: Date.now(), role: "assistant", content: r.message, why: r.why, emergency: r.assessment?.level === "emergency" }]);
      setProgress(r.progress ?? 0);
      if (r.assessment) {
        setResult((prev) => ({ ...(prev || {}), assessment: r.assessment, medicines: r.medicines ?? prev?.medicines,
                               answer: r.pending_question ? null : r.answer ?? prev?.answer }));
      }
      if (r.message) say(r);
      pending = r.pending_question;
      if (r.assessment && (r.first_assessment || r.assessment.changed || !docs)) loadDoctors(run, sidNow);     // in parallel with the explanation
    } catch (e) {
      if (live()) setLog((l) => [...l, { t: Date.now(), role: "assistant", content: "Error: " + e.message }]);
    } finally { if (live()) setBusy(false); }
    if (!pending || !live()) return;
    // 2) slow: the medical model writes the explanation, grounded in one of the three sources
    setAnswering(true); setSince(Date.now());
    try {
      const r2 = await api.consultAnswer(sidNow, pending, voiceOut);
      if (!live()) return;
      setLog((l) => [...l, { t: Date.now(), role: "assistant", content: r2.message }]);
      setResult((prev) => ({ ...(prev || {}), answer: r2.answer }));
      say(r2);
    } catch (e) {
      if (live()) setLog((l) => [...l, { t: Date.now(), role: "assistant", content: "I could not get the explanation: " + e.message }]);
    } finally { if (live()) setAnswering(false); }
  };

  const loadDoctors = async (run = runRef.current, id = sid) => {
    setDocsLoading(true);
    try {
      const d = await api.consultDoctors(id);
      if (run === runRef.current) setDocs(d);
    } catch (e) {
      if (run === runRef.current) setDocs({ results: [], error: e.message });
    } finally { if (run === runRef.current) setDocsLoading(false); }
  };

  const shareLocation = () => {
    setLocErr("");
    if (!navigator.geolocation) return setLocErr("This browser cannot share a location - type your city or PIN code instead.");
    navigator.geolocation.getCurrentPosition(
      (p) => applyLocation({ lat: p.coords.latitude, lon: p.coords.longitude }),
      (e) => setLocErr((e.code === 1 ? "Location permission was denied" : "Could not get your location") + " - type your city or PIN code instead."),
      { enableHighAccuracy: false, timeout: 10000, maximumAge: 300000 });
  };

  const applyLocation = async (body) => {
    setLocErr("");
    if (!sid) {                                      // no session yet: keep it and send it with the first message
      if (body.lat != null) return setLoc({ lat: body.lat, lon: body.lon, label: "your current location" });
      try { setLoc(await api.geocode(body.place)); } catch (e) { setLocErr(e.message); }
      return;
    }
    try {
      if (result?.assessment) setDocsLoading(true);
      const r = await api.consultLocation(sid, body);
      setLoc(r.location);
      if (r.doctors) setDocs(r.doctors);
      setDocsLoading(false);
    } catch (e) { setLocErr(e.message); }
  };

  const quick = async (t) => { const id = await start(); if (id) send(t, id); };

  // Always-on listening: before the consultation starts, speaking the problem starts it; afterwards speech answers the question.
  useVoiceInput((t) => (sid ? send(t) : quick(t)), !saved && !busy && !answering);

  const a = result?.assessment;
  const ans = result?.answer;
  const emergency = a?.level === "emergency";

  return (
    <div className="consult">
      <div className="panel">
        {!!sid && <div className="chat-head"><span className="live-dot" /> <b>Consultation</b><span className="muted small"> · answers are guidance, not a diagnosis</span></div>}
        {!sid && saved ? (
          <>
            <div className="banner warn">This is a saved consultation from {new Date(saved.updated_at).toLocaleString()}. Start a new consultation for a new problem.</div>
            <div className="log">
              {log.map((m, i) => <Msg key={i} m={m} user={user} />)}
            </div>
          </>
        ) : !sid ? (
          <div className="empty">
            <div className="empty-orb">✚</div>
            <h3 className="empty-t">Tell me what's wrong</h3>
            <p className="muted">Describe your symptoms by voice or text. I'll ask a few smart questions, tell you how urgent it is, suggest safe over-the-counter options and find the right doctor near you.</p>
            <button className="primary big" onClick={start} disabled={busy}>Start consultation</button>
            <div className="quick-label">or start with</div>
            <div className="chips center">
              {["I have a fever and a sore throat", "I have a bad headache since morning", "I have a cough and a blocked nose", "My stomach hurts and I feel nauseous"].map((t) => (
                <button key={t} className="chipbtn" onClick={() => quick(t)} disabled={busy}>{t}</button>
              ))}
            </div>
            <div className="feat">
              <div><b>⚡ Instant triage</b><span>Urgency from your symptoms</span></div>
              <div><b>💊 Safe medicines</b><span>Checked for your age & allergies</span></div>
              <div><b>📍 Doctors nearby</b><span>The right specialist</span></div>
            </div>
          </div>
        ) : (
          <>
            {progress < 1 && <div className="progress"><div style={{ width: `${progress * 100}%` }} /></div>}
            <div className="log">
              {log.map((m, i) => <Msg key={i} m={m} user={user} />)}
              {busy && <Thinking since={since} text="Assessing…" />}
              {answering && <Thinking since={since} text="Your urgency level is on the right. The Hugging Face model is now writing an explanation…" />}
              <div ref={end} />
            </div>
            <form className="composer" onSubmit={(e) => { e.preventDefault(); send(text); }}>
              <div className="composer-top"><span className="composer-k">✚ CONSULT</span><span className="muted small">Describe symptoms by voice or text</span></div>
              <div className="composer-row">
              <MicButton onText={send} disabled={busy || answering} />
              <input value={text} onChange={(e) => setText(e.target.value)}
                     placeholder={result ? "Ask a follow-up medical question…" : "Type your answer…"} disabled={busy || answering} />
              <button className="send" aria-label="Send" disabled={busy || answering || !text.trim()}>→</button>
              </div>
            </form>
            <button className="link" onClick={start} disabled={busy}>Start a new consultation</button>
          </>
        )}
      </div>

      <div className="panel">
        <div className="eyebrow">Your location</div>
        <div className="loc">
                <button type="button" onClick={shareLocation}>📍 Use my location</button>
                <input value={place} onChange={(e) => setPlace(e.target.value)} placeholder="or type your city / PIN code"
                       onKeyDown={(e) => e.key === "Enter" && place.trim() && applyLocation({ place: place.trim() })} />
                <button type="button" disabled={!place.trim()} onClick={() => applyLocation({ place: place.trim() })}>Find</button>
              </div>
              {locErr && <div className="err small">{locErr}</div>}
              {loc?.label && <div className="muted small">Searching around: {loc.label} (location rounded to ~1 km; symptoms are never sent)</div>}

        <div className="eyebrow">Assessment</div>
        {!a && <div className="hint">Your urgency level, the recommended specialist and the sources of the answer appear here once the interview is complete.</div>}
        {a && (
          <>
            {emergency && <div className="banner bad">Call your local emergency number (112 / 911 / 108) now or go to the nearest emergency department.</div>}
            <div className={`level-card ${LEVEL_CLASS[a.level]}`}>
              <div className="big">{a.label}</div>
              <div>{a.window}</div>
              <div><b>{emergency ? "Go to:" : "Suggested specialist:"}</b> {emergency ? "Emergency department" : a.specialty}</div>
              {a.reasons?.length > 0 && <ul>{a.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>}
            </div>

            {!emergency && a.possible_causes?.length > 0 && (
              <div className="answer-card">
                <div className="meta"><span className="tag info">What this could be</span><span className="muted small">possibilities from your symptoms, not a diagnosis</span></div>
                {a.possible_causes.map((p, i) => (
                  <div key={i} className="med">
                    <div><b>{p.name}</b> <span className={`tag ${p.likelihood === "most likely" ? "ok" : "warn"}`}>{p.likelihood}</span></div>
                    {p.because?.length > 0 && <div className="muted small">Because of: {p.because.join(", ")}</div>}
                    <div className="small">{p.note}</div>
                  </div>
                ))}
              </div>
            )}
            {!emergency && <Medicines m={result?.medicines} />}
            {answering && !emergency && <Thinking since={since} text="Writing the explanation from the best available source…" />}
            {!answering && ans && ans.source && !emergency && (
              <div className="answer-card">
                <div className="meta">
                  <span className={`tag ${SOURCE_CLASS[ans.source]}`}>{SOURCE_SHORT[ans.source]}</span>
                  {ans.confidence > 0 && <span className="muted">confidence {Math.round(ans.confidence * 100)}%</span>}
                  {ans.served_by && <span className="muted small">model: {host(ans.served_by)}{ans.timings_ms?.model_ms || ans.timings_ms?.dialogue_llm_ms || ans.timings_ms?.web_llm_ms ? ` · ${Math.round((ans.timings_ms.model_ms || ans.timings_ms.dialogue_llm_ms || ans.timings_ms.web_llm_ms) / 1000)}s` : ""}</span>}
                  {ans.flags?.includes("verbatim_excerpt") && <span className="tag warn">quoted from the page</span>}
                  {ans.flags?.includes("verbatim_doctor_reply") && <span className="tag warn">doctor's own words</span>}
                  {ans.flags?.includes("model_unavailable") && <span className="tag warn">medical model offline</span>}
                </div>
                <div className="muted small">{ans.source_label}</div>
                {ans.sources?.length > 0 && (
                  <div className="srcs">
                    <b className="small">{ans.source === "web" ? "Pages I used (open to read more):" : "Where this came from:"}</b>
                    {ans.sources.map((s, i) => (
                      <div key={i} className="src">
                        <b>[{i + 1}] {s.title || s.source}</b> <span className="muted small">{s.source}</span>{" "}
                        {s.url && <a href={s.url} target="_blank" rel="noreferrer">{ans.source === "web" ? s.url : "open dataset"}</a>}
                        {s.snippet && <div className="muted small">{ans.source === "dialogues" ? "Doctor replied: " : ""}{s.snippet}</div>}
                      </div>
                    ))}
                  </div>
                )}
                {ans.trail?.length > 0 && (
                  <details><summary className="small muted">How this answer was found</summary>
                    <ul className="trail">{ans.trail.map((t, i) => <li key={i}>{t}</li>)}</ul>
                  </details>
                )}
              </div>
            )}

            <div>
              <div className="eyebrow">{emergency ? "Nearest hospitals" : `Nearby ${a.specialty}s`}</div>
              {docsLoading && <div className="hint">Looking for {emergency ? "hospitals" : "specialists"} near you…</div>}
              {docs?.needs_location && <div className="hint">Share your location above to see {emergency ? "hospitals" : "specialists"} near you.</div>}
              {docs?.error && <div className="banner warn">Could not reach the map service: {docs.error}. Try again in a moment.</div>}
              {docs && !docs.needs_location && !docs.error && docs.results.length === 0 && <div className="hint">No facilities found within {docs.radius_km} km on the map.</div>}
              {docs?.results?.map((d, i) => (
                <div key={i} className="doc">
                  <div className="name">{d.name} <span className="muted small">({d.kind})</span></div>
                  <div className="row">
                    <span>{d.distance_km} km</span>
                    <span className={`tag ${d.match === "verified" ? "ok" : d.match === "hospital" ? "info" : "warn"}`}>{MATCH_TEXT[d.match]}</span>
                    {d.emergency && <span className="tag bad">emergency dept</span>}
                  </div>
                  {d.address && <div className="muted small">{d.address}</div>}
                  <div className="row small">
                    {d.phone && <a href={`tel:${d.phone.replace(/[^+\d]/g, "")}`}>📞 {d.phone}</a>}
                    {d.website && <a href={d.website} target="_blank" rel="noreferrer">website</a>}
                    <a href={d.directions_url} target="_blank" rel="noreferrer">directions</a>
                    <a href={d.map_url} target="_blank" rel="noreferrer">map</a>
                    {d.opening_hours && <span className="muted">{d.opening_hours}</span>}
                  </div>
                </div>
              ))}
              {docs?.results?.length > 0 && <div className="muted small">From OpenStreetMap: availability and specialities may be incomplete. Please call ahead.</div>}
            </div>
          </>
        )}
        <div className="muted small">
          AI-generated guidance to help you decide what to do next. It is not a diagnosis and does not replace a doctor.
          {health?.med_llm && !health.med_llm.available && " The medical AI model is offline, so answers will be quoted from doctor conversations or web pages."}
        </div>
      </div>
    </div>
  );
}
