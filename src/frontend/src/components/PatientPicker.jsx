import React, { useEffect, useState } from "react";
import { api } from "../api.js";

export default function PatientPicker({ patient, onChange }) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState([]);

  useEffect(() => {
    if (q.trim().length < 2) return setResults([]);
    const t = setTimeout(() => api.searchPatients(q).then(setResults).catch(() => setResults([])), 250);
    return () => clearTimeout(t);
  }, [q]);

  const pick = async (id) => {
    onChange(await api.getPatient(id));
    setQ("");
    setResults([]);
  };

  return (
    <div className="picker">
      {patient ? (
        <div className="chip">
          <b>{patient.full_name}</b> · {patient.patient_id} · {patient.age}y {patient.sex}
          <span className="muted"> · {patient.conditions || "no known conditions"}</span>
          <button className="link" onClick={() => onChange(null)}>clear</button>
        </div>
      ) : (
        <>
          <input placeholder="Select patient: search by name or ID (e.g. P0000002)…" value={q} onChange={(e) => setQ(e.target.value)} />
          {results.length > 0 && (
            <ul className="dropdown">
              {results.map((r) => (
                <li key={r.patient_id} onClick={() => pick(r.patient_id)}>
                  {r.full_name} <span className="muted">{r.patient_id} · {r.age}y {r.sex} · {r.city}</span>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
