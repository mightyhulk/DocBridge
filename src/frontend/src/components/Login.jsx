import React, { useEffect, useState } from "react";
import { api, auth } from "../api.js";

const ERRORS = {
  cancelled: "Sign-in was cancelled.",
  access_denied: "Sign-in was cancelled.",
  state_mismatch: "That sign-in link expired. Please try again.",
  google_unreachable: "Could not reach Google. Check your connection and try again.",
  code_used: "That sign-in link was already used or expired. Click Continue with Google once and wait - do not refresh the page.",
  google_rejected: "Google rejected the sign-in. Check the client secret and redirect address in backend/.env.",
  email_not_verified: "Your Google email is not verified.",
};

function GoogleMark() {
  return (
    <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true">
      <path fill="#FFC107" d="M43.6 20.1H42V20H24v8h11.3C33.7 32.7 29.2 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.8 1.2 8 3l5.7-5.7C34 6.1 29.3 4 24 4 13 4 4 13 4 24s9 20 20 20 20-9 20-20c0-1.3-.1-2.6-.4-3.9z" />
      <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.7 15.1 19 12 24 12c3.1 0 5.8 1.2 8 3l5.7-5.7C34 6.1 29.3 4 24 4 16.3 4 9.7 8.3 6.3 14.7z" />
      <path fill="#4CAF50" d="M24 44c5.2 0 9.9-2 13.4-5.2l-6.2-5.2C29.2 35.1 26.7 36 24 36c-5.2 0-9.6-3.3-11.3-7.9l-6.5 5C9.5 39.6 16.2 44 24 44z" />
      <path fill="#1976D2" d="M43.6 20.1H42V20H24v8h11.3c-.8 2.2-2.2 4.2-4.1 5.6l6.2 5.2C37 39.2 44 34 44 24c0-1.3-.1-2.6-.4-3.9z" />
    </svg>
  );
}

export default function Login({ enabled, onSignedIn = () => window.location.reload() }) {
  const err = new URLSearchParams(window.location.search).get("login_error");
  const [methods, setMethods] = useState(null);          // {google, local}: what this server offers to this browser
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [localErr, setLocalErr] = useState("");
  useEffect(() => { auth.methods().then(setMethods).catch(() => setMethods({ google: enabled, local: false })); }, [enabled]);

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true); setLocalErr("");
    try { await auth.localLogin(name, email); onSignedIn(); }
    catch (ex) { setLocalErr(/pattern|string_pattern/.test(ex.message) ? "Please enter a valid email address." : ex.message); setBusy(false); }
  };

  return (
    <div className="login">
      <div className="login-card">
        <div className="brand-mark">✚</div>
        <h1 className="brand-name">Vitalis AI</h1>
        <p className="login-lead">Describe how you feel by voice or text. Get clear guidance on how urgent it is, safe over-the-counter options, and doctors near you.</p>
        {err && <div className="banner bad">{ERRORS[err] || "Sign-in failed. Please try again."}</div>}
        {methods?.google && <a className="google-btn" href="/api/auth/google/login"><GoogleMark /> Continue with Google</a>}
        {methods?.google && methods?.local && <div className="small muted" style={{ margin: "10px 0" }}>or</div>}
        {methods?.local && (
          <form className="local-login" onSubmit={submit}>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Your name" autoComplete="name" required maxLength={80} />
            <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Your email" type="email" autoComplete="email" required />
            <button className="primary big" disabled={busy || !name.trim() || !email.trim()}>{busy ? "Signing in…" : "Continue"}</button>
            {localErr && <div className="err small">{localErr}</div>}
            <p className="small muted">Your email only keeps your conversation history separate from other people on this network.</p>
          </form>
        )}
        {methods && !methods.google && !methods.local && (
          <div className="banner warn">Sign-in is not available for this address. Open the site at its https:// address, or set HVA_LOCAL_AUTH=true in config/.env.</div>
        )}
        <p className="small muted">AI guidance supports, but never replaces, a licensed doctor. In an emergency call your local emergency number.</p>
      </div>
    </div>
  );
}
