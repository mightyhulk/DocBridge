import React, { useEffect, useRef, useState } from "react";
import { api, history, speak } from "../api.js";
import { useVoiceInput } from "../voice.js";
import MicButton from "./MicButton.jsx";
import PatientSnapshot from "./PatientSnapshot.jsx";

const TIER = {
  record: ["Patient record", "ok"], clinical: ["Clinical rules", "ok"], llm: ["Model knowledge", "ok"], rag: ["Knowledge base", "ok"],
  web: ["Web / online source", "warn"], extractive: ["Reference extract", "warn"], abstain: ["No reliable answer", "bad"], emergency: ["EMERGENCY", "bad"],
};

export default function Assistant({ patient, voiceOut, saved = null, onSaved = () => {} }) {
  const [msgs, setMsgs] = useState(saved?.log || []);          // a saved chat reopens here and can be continued
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const end = useRef(null);

  // Every conversation is saved to the signed-in user's history (once an answer has arrived, not while it is being written).
  const hidRef = useRef(saved?.id || "");
  useEffect(() => {
    if (busy || !msgs.some((m) => m.role === "user")) return;
    if (!hidRef.current) hidRef.current = (crypto.randomUUID?.() || String(Date.now())).replace(/-/g, "").slice(0, 24);
    const first = msgs.find((m) => m.role === "user").content;
    const t = setTimeout(() => {
      const who = patient ? { patient_id: patient.patient_id, full_name: patient.full_name } : null;
      history.save(hidRef.current, { kind: "assistant", title: first.slice(0, 60), level: null, log: msgs, patient: who }).then(onSaved).catch(() => {});
    }, 400);
    return () => clearTimeout(t);
  }, [msgs, busy]);
  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth" });   // block body: an effect must not return a non-function
  }, [msgs, busy]);

  const send = async (message) => {
    message = message.trim();
    if (!message || busy) return;
    const history = msgs.filter((m) => !m.meta).slice(-6).map((m) => ({ role: m.role, content: m.content }));
    setMsgs((m) => [...m, { role: "user", content: message }]);
    setText("");
    setBusy(true);
    try {
      const r = await api.chat(message, patient?.patient_id, history, voiceOut);
      // Text and voice come from the same response; start the audio on the frame the text is painted so they coincide.
      setMsgs((m) => [...m, { role: "assistant", content: String(r.answer ?? ""), meta: { ...r, audio_b64: undefined } }]);
      if (voiceOut) requestAnimationFrame(() => speak({ audio_b64: r.audio_b64, text: r.answer }));
    } catch (e) {
      setMsgs((m) => [...m, { role: "assistant", content: "Error: " + e.message, meta: { tier: "abstain" } }]);
    } finally {
      setBusy(false);
    }
  };

  useVoiceInput(send, !busy);       // always-on listening: speech is sent like typed text

  const EXAMPLES = ["Does warfarin interact with ibuprofen?", "What is the usual adult dose of metformin?", "What are the red flag symptoms of dengue fever?",
                    "Is it safe to take ibuprofen with kidney disease?"];

  return (
    <div className={patient ? "assistant-grid" : ""}>
    <div className="panel">
      <div className="log">
        {msgs.length === 0 && (
          <div className="hint">
            Ask a clinical question by voice or text: drug doses and interactions, guidelines, red flags, or select a patient above to check medicines
            against their allergies, medications and conditions and to see what needs attention in their chart.
            {!patient && (
              <div className="chips">
                {EXAMPLES.map((q) => <button key={q} type="button" className="chipbtn" onClick={() => send(q)} disabled={busy}>{q}</button>)}
              </div>
            )}
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={`bubble ${m.role} ${m.meta?.emergency ? "emergency" : ""}`}>
            <div>{m.content}</div>
            {m.meta && m.meta.tier && (
              <div className="meta">
                <span className={`tag ${TIER[m.meta.tier]?.[1]}`}>{TIER[m.meta.tier]?.[0] || m.meta.tier}</span>
                {m.meta.confidence > 0 && <span className="muted">confidence {Math.round(m.meta.confidence * 100)}%</span>}
                {m.meta.timings_ms?.total != null && <span className="muted">{m.meta.timings_ms.total} ms</span>}
                {m.meta.flags?.includes("llm_unavailable") && <span className="tag warn">LLM offline</span>}
                {m.meta.sources?.length > 0 && (
                  <details>
                    <summary>{m.meta.sources.length} source(s)</summary>
                    {m.meta.sources.map((s, j) => (
                      <div key={j} className="src">
                        <b>[{j + 1}] {s.title}</b> <span className="muted">{s.source}</span>{" "}
                        {s.url && <a href={s.url} target="_blank" rel="noreferrer">link</a>}
                        <div className="muted small">{s.snippet}</div>
                      </div>
                    ))}
                  </details>
                )}
                {m.meta.disclaimer && <div className="muted small">{m.meta.disclaimer}</div>}
              </div>
            )}
          </div>
        ))}
        {busy && <div className="bubble assistant muted">Thinking…</div>}
        <div ref={end} />
      </div>
      <form className="composer" onSubmit={(e) => { e.preventDefault(); send(text); }}>
        <div className="composer-top"><span className="composer-k">✚ CLINICAL</span><span className="muted small">Doses, interactions, guidelines · say "search the web" for live sources</span></div>
        <div className="composer-row">
          <MicButton onText={send} disabled={busy} />
          <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Ask a clinical question…" />
          <button className="send" aria-label="Send" disabled={busy || !text.trim()}>→</button>
        </div>
      </form>
    </div>
    {patient && <PatientSnapshot patient={patient} onAsk={send} disabled={busy} />}
    </div>
  );
}
