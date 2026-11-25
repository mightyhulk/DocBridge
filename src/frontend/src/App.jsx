import React, { Fragment, useEffect, useState } from "react";
import { api, auth, history, stopSpeaking } from "./api.js";
import { LEVEL_LABEL, bucket, showTitle, when } from "./historyTitle.js";
import Assistant from "./components/Assistant.jsx";
import Consult from "./components/Consult.jsx";
import Login from "./components/Login.jsx";
import PatientPicker from "./components/PatientPicker.jsx";
import { useVoice, voice } from "./voice.js";

const PILL = {
  starting: "Starting the microphone…", listening: "Listening - just speak", hearing: "Hearing you…", transcribing: "Transcribing…",
  paused: "Paused while I answer", muted: "Muted - tap to listen", "needs-tap": "Tap anywhere to start listening", blocked: "Microphone unavailable",
};

/** Always-visible state of the browser's always-on listener. Tap to mute / unmute (or to retry when blocked). */
function ListenPill() {
  const v = useVoice();
  if (v.status === "off") return null;
  const bad = v.status === "blocked";
  return (
    <div className="listen-wrap">
      <button type="button" className={`listen-pill ${v.status}`} onClick={() => (bad ? voice.start() : voice.setMuted(!v.muted))}
              title={bad ? "Tap to retry" : v.muted ? "Tap to listen again" : "Tap to mute the microphone"}>
        <span className="listen-dot" style={{ transform: `scale(${1 + (v.status === "hearing" ? v.level : 0)})` }} />
        {PILL[v.status] || v.status}
      </button>
      {bad && v.error && <div className="listen-err">{v.error}</div>}
      {v.heard && !bad && v.status !== "muted" && <div className="listen-heard">Heard: “{v.heard}”</div>}
    </div>
  );
}

const TABS = [
  ["consult", "Patient Consult", "🩺", "Describe your symptoms"],
  ["assistant", "Clinical Assistant", "📋", "For clinicians"],
];

function Avatar({ user, size = 38 }) {
  const initial = (user?.name || user?.email || "?").trim()[0]?.toUpperCase();
  return user?.picture
    ? <img className="avatar" style={{ width: size, height: size }} src={user.picture} alt="" referrerPolicy="no-referrer" />
    : <span className="avatar" style={{ width: size, height: size }}>{initial}</span>;
}

export default function App() {
  const [session, setSession] = useState(null);       // null = still checking, else {enabled, user}
  const [tab, setTab] = useState("consult");
  const [consultKey, setConsultKey] = useState(0);    // bumping it starts a fresh consultation
  const [assistKey, setAssistKey] = useState(0);      // bumping it starts a fresh assistant chat
  const [patient, setPatient] = useState(null);
  const [voiceOut, setVoiceOut] = useState(true);
  const [health, setHealth] = useState(null);
  const [showStatus, setShowStatus] = useState(false);
  const [past, setPast] = useState([]);               // this patient's saved consultations
  const [menuOpen, setMenuOpen] = useState(false);     // mobile drawer
  const [viewing, setViewing] = useState(null);       // a saved consultation opened read-only

  useEffect(() => { auth.me().then(setSession).catch(() => setSession({ enabled: false, user: null })); }, []);

  const signedIn = !!session && (!session.enabled || !!session.user);

  useEffect(() => {
    if (!signedIn) return;
    const load = () => api.health().then(setHealth).catch(() => setHealth(null));
    load();
    const t = setInterval(load, 15000);
    return () => clearInterval(t);
  }, [signedIn]);

  // Always-on listening via the browser microphone as soon as the user is signed in.
  useEffect(() => {
    if (!signedIn) { voice.stop(); return; }
    voice.start();
    return () => voice.stop();
  }, [signedIn]);

  const refreshPast = () => history.list().then(setPast).catch(() => {});
  useEffect(() => { if (signedIn) refreshPast(); }, [signedIn]);

  const openPast = async (id) => {
    stopSpeaking(); setMenuOpen(false);
    try {
      const h = await history.get(id);
      setViewing(h);
      if (h.kind === "assistant") { setPatient(h.patient || null); setTab("assistant"); setAssistKey((k) => k + 1); }
      else { setTab("consult"); setConsultKey((k) => k + 1); }
    } catch { refreshPast(); }
  };
  const removePast = async (id) => {
    await history.remove(id).catch(() => {});
    if (viewing?.id === id) { setViewing(null); setConsultKey((k) => k + 1); setAssistKey((k) => k + 1); }
    refreshPast();
  };

  const newThing = () => {
    stopSpeaking(); setViewing(null); setMenuOpen(false);
    if (tab === "assistant") setAssistKey((k) => k + 1); else { setTab("consult"); setConsultKey((k) => k + 1); }
  };

  const signOut = async () => {
    stopSpeaking();
    await auth.logout().catch(() => {});
    setSession((s) => ({ ...s, user: null }));
  };

  if (!session) return <div className="login"><div className="thinking"><span className="spin" /> Loading…</div></div>;
  if (!signedIn) return <Login enabled={session.enabled} />;

  const user = session.user;
  const dot = (ok) => <span className={`dot ${ok ? "ok" : "bad"}`} />;
  const firstName = (user?.name || "").split(" ")[0];

  return (
    <div className="shell">
      <header className="topbar">
        <button className="hamb" aria-label="Open menu" onClick={() => setMenuOpen(true)}>☰</button>
        <span className="brand-mark sm">✚</span><b className="brand-name sm">Vitalis AI</b>
        <button className="topbar-new" aria-label="New" onClick={newThing}>＋</button>
      </header>
      {menuOpen && <div className="backdrop" onClick={() => setMenuOpen(false)} />}
      <aside className={`sidebar ${menuOpen ? "open" : ""}`}>
        <div className="side-brand">
          <span className="brand-mark sm">✚</span>
          <div>
            <div className="brand-name sm">Vitalis AI</div>
            <span className="pill gold">DIGITAL DOCTOR</span>
          </div>
          <button className="close-menu" aria-label="Close menu" onClick={() => setMenuOpen(false)}>×</button>
        </div>

        <button className="new-btn" onClick={newThing}>{tab === "assistant" ? "+ New Chat" : "+ New Consultation"}</button>

        <nav className="side-nav">
          {TABS.map(([id, label, icon, sub]) => (
            <button key={id} className={tab === id ? "on" : ""} onClick={() => { setTab(id); setMenuOpen(false); }}>
              <span className="ico">{icon}</span>
              <span><span className="nav-t">{label}</span><span className="nav-s">{sub}</span></span>
            </button>
          ))}
        </nav>

        <div className="hist">
          <div className="side-label">HISTORY</div>
          <div className="past">
            {past.length === 0 && <div className="who-sub" style={{ padding: "4px 8px" }}>Your conversations are saved here.</div>}
            {past.map((h, i) => {
              const g = bucket(h.updated_at), newGroup = i === 0 || g !== bucket(past[i - 1].updated_at);
              const lvl = h.kind === "assistant" ? "Clinical Assistant" : LEVEL_LABEL[h.level] || "No assessment yet";
              return (
                <Fragment key={h.id}>
                  {newGroup && <div className="past-group">{g}</div>}
                  <div className={`past-item ${viewing?.id === h.id ? "on" : ""}`}>
                    <button className="past-open" onClick={() => openPast(h.id)} title={`${h.title}\n${new Date(h.updated_at).toLocaleString()}`}>
                      {h.kind === "assistant" ? <span className="kind-ico" title="Clinical Assistant chat">💬</span> : <span className={`lvl ${h.level || "none"}`} title={lvl} />}
                      <span className="past-txt">
                        <span className="past-t">{showTitle(h)}</span>
                        <span className="past-m">{lvl} · {when(h.updated_at)}</span>
                      </span>
                    </button>
                    <button className="past-x" aria-label="Delete this consultation" onClick={() => removePast(h.id)}>×</button>
                  </div>
                </Fragment>
              );
            })}
          </div>
        </div>

        <details className="settings">
          <summary>⚙ Settings</summary>
          <div className="side-set">
            <label className="toggle"><input type="checkbox" checked={voiceOut} onChange={(e) => { setVoiceOut(e.target.checked); stopSpeaking(); }} /> Spoken replies</label>
            <button className="link status-toggle" onClick={() => setShowStatus((v) => !v)}>{showStatus ? "Hide" : "Show"} system status</button>
            {showStatus && (
              <div className="status-list">
                {health ? (
                  <>
                    <span>{dot(health.llm.available)} LLM {health.llm.available ? `${health.llm.model} (HF)` : "offline (set HF_TOKEN)"}</span>
                    <span>{dot(health.vector_db.documents > 0)} KB {health.vector_db.documents}</span>
                    <span>{dot(health.stt.available)} voice: {health.stt.available ? health.stt.model : "text / browser"} / {health.tts.engine}</span>
                    <span title={health.med_llm?.endpoint || "set HF_TOKEN to enable Hugging Face model"}>{dot(health.med_llm?.available)} medical AI {health.med_llm?.available ? `${health.med_llm.model} (HF)` : "offline"}</span>
                    <span>{dot(health.dialogues?.cases > 0)} {(health.dialogues?.cases || 0).toLocaleString()} doctor-patient cases</span>
                  </>
                ) : <span>{dot(false)} backend unreachable</span>}
              </div>
            )}
          </div>
        </details>

        <div className="side-user">
          {user ? (
            <>
              <div className="who"><Avatar user={user} size={34} /><div><div className="who-name">{user.name}</div><div className="who-sub">{user.email}</div></div></div>
              <button className="signout" onClick={signOut} title="Sign out" aria-label="Sign out"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><path d="M16 17l5-5-5-5" /><path d="M21 12H9" /></svg></button>
            </>
          ) : <div className="who-sub">Sign-in is off (demo mode)</div>}
        </div>
      </aside>

      <section className="content">
        <ListenPill />
        <div className="content-inner">
          {tab === "consult" && <h2 className="greet">{firstName ? `Hello, ${firstName}.` : "Hello."} <span>How are you feeling today?</span></h2>}
          {tab !== "consult" && <PatientPicker patient={patient} onChange={setPatient} />}
          <main>
            {tab === "consult" && <Consult key={consultKey} user={user} voiceOut={voiceOut} health={health} saved={viewing?.kind === "consult" ? viewing : null} onSaved={refreshPast} />}
            {tab === "assistant" && <Assistant key={assistKey} patient={patient} voiceOut={voiceOut} saved={viewing?.kind === "assistant" ? viewing : null} onSaved={refreshPast} />}
          </main>
          <footer>AI-generated information supports, but never replaces, a licensed clinician. Not for emergencies. Verify critical medical decisions with your doctor.</footer>
        </div>
      </section>
    </div>
  );
}
