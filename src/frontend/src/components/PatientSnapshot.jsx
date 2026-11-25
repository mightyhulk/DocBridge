import React, { useEffect, useState } from "react";
import { api } from "../api.js";

const SEV_CLASS = { critical: "bad", warning: "warn", info: "info" };
const VITAL_CLASS = { critical: "bad", high: "bad", elevated: "warn", borderline: "warn", low: "warn", normal: "ok", unknown: "" };
const VERDICT = { avoid: ["Not recommended", "bad"], caution: ["Use with caution", "warn"], ok: ["No problems found in these checks", "ok"] };
const FINDING_CLASS = { contraindicated: "bad", major: "bad", moderate: "warn", info: "info" };

/** Chart snapshot for the selected patient: what is wrong in the chart right now, plus a one-click medication safety check. */
export default function PatientSnapshot({ patient, onAsk, disabled }) {
  const [sum, setSum] = useState(null);
  const [err, setErr] = useState("");
  const [drug, setDrug] = useState("");
  const [check, setCheck] = useState(null);
  const [checking, setChecking] = useState(false);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    let live = true;
    setSum(null); setErr(""); setCheck(null); setShowAll(false);
    api.patientSummary(patient.patient_id).then((s) => live && setSum(s)).catch((e) => live && setErr(e.message));
    return () => { live = false; };
  }, [patient.patient_id]);

  const runCheck = async (e) => {
    e?.preventDefault();
    if (!drug.trim() || checking) return;
    setChecking(true); setCheck(null);
    try { setCheck(await api.medcheck(patient.patient_id, drug.trim())); }
    catch (ex) { setCheck({ known: false, message: ex.message }); }
    finally { setChecking(false); }
  };

  if (err) return <div className="panel"><div className="err small">Could not load the chart summary: {err}</div></div>;
  if (!sum) return <div className="panel"><div className="hint">Loading chart summary…</div></div>;

  const alerts = sum.alerts || [];
  const shown = showAll ? alerts : alerts.slice(0, 5);
  const crit = alerts.filter((a) => a.severity === "critical").length;
  const diabetic = sum.conditions.some((c) => /diabet/i.test(c));
  const chips = [
    "Review this patient's medications and chart for problems",
    "Is this patient's blood pressure at target?",
    ...(diabetic ? ["Is this patient's HbA1c at target?"] : ["Is this patient's LDL at target?"]),
  ];

  return (
    <div className="panel snapshot">
      <div className="snap-head">
        <h3>{sum.patient.full_name} <span className="muted small">{sum.patient.patient_id} · {sum.patient.age}y {sum.patient.sex}{sum.smoker ? " · smoker" : ""}</span></h3>
        <span className={`tag ${crit ? "bad" : alerts.length ? "warn" : "ok"}`}>{crit ? `${crit} critical` : alerts.length ? `${alerts.length} to review` : "no alerts"}</span>
      </div>

      {alerts.length > 0 && (
        <div className="alerts">
          {shown.map((a, i) => (
            <div key={i} className={`alert ${SEV_CLASS[a.severity]}`}>
              <b>{a.title}</b>
              <div className="small">{a.detail}</div>
              {a.action && <div className="small"><i>{a.action}</i></div>}
            </div>
          ))}
          {alerts.length > 5 && <button className="link" onClick={() => setShowAll(!showAll)}>{showAll ? "show fewer" : `show all ${alerts.length}`}</button>}
        </div>
      )}

      <div className="vitals">
        {sum.vitals.map((v) => (
          <span key={v.key} className={`vital ${VITAL_CLASS[v.status]}`} title={v.note}>
            <span className="muted small">{v.label}</span> <b>{v.value}{v.unit ? ` ${v.unit}` : ""}</b>
          </span>
        ))}
      </div>

      <div className="small"><b>Conditions:</b> {sum.conditions.join(", ") || "none recorded"}</div>
      <div className="small"><b>Medicines:</b> {sum.medications.length ? sum.medications.map((m) => m.text).join("; ") : "none"}</div>
      <div className="small"><b>Allergies:</b> {sum.allergies.length ? sum.allergies.join(", ") : "none known"}</div>

      <form className="composer" onSubmit={runCheck}>
        <input value={drug} onChange={(e) => setDrug(e.target.value)} placeholder="Check a medicine for this patient (e.g. ibuprofen)…" />
        <button disabled={!drug.trim() || checking}>{checking ? "…" : "Check"}</button>
      </form>
      {check && (
        check.known === false ? <div className="banner warn">{check.message}</div> : (
          <div className="check">
            <span className={`tag ${VERDICT[check.verdict][1]}`}>{VERDICT[check.verdict][0]}: {check.drug}</span>
            {check.findings.length === 0 && <div className="muted small">No allergy, interaction or condition problem in the curated checks. This is not a guarantee: check eGFR, pregnancy status and local guidance.</div>}
            {check.findings.map((f, i) => (
              <div key={i} className={`alert ${FINDING_CLASS[f.severity]}`}>
                <b>{f.title}</b> <span className="muted small">{f.severity}</span>
                <div className="small">{f.detail}</div>
                <div className="small"><i>{f.action}</i></div>
              </div>
            ))}
          </div>
        )
      )}

      <div className="chips">
        {chips.map((c) => <button key={c} type="button" className="chipbtn" disabled={disabled} onClick={() => onAsk(c)}>{c}</button>)}
      </div>
      <div className="muted small">Decision support from curated references (BNF / FDA labels / ACC-AHA / ADA). It covers common drugs and checks only: verify against your formulary and the full record.</div>
    </div>
  );
}
