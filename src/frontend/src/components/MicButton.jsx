import React from "react";
import { useVoice, voice } from "../voice.js";

const LABEL = {
  starting: "Starting the microphone…",
  listening: "Listening - just speak",
  hearing: "Hearing you…",
  transcribing: "Transcribing…",
  paused: "Paused while I answer",
  muted: "Microphone muted - tap to listen again",
  "needs-tap": "Tap anywhere to start listening",
  blocked: "Microphone unavailable",
  off: "Microphone off",
};

/** Browser mic button: shows the always-on listener state and lets the user mute/unmute or retry. */
export default function MicButton() {
  const v = useVoice();
  const live = v.status === "listening" || v.status === "hearing";
  return (
    <span className="mic-wrap">
      <button
        type="button"
        className={`mic ${v.status === "hearing" ? "rec" : live ? "listen" : ""}`}
        onClick={() => (v.status === "blocked" || v.status === "off" ? voice.start() : voice.setMuted(!v.muted))}
        title={v.error || LABEL[v.status] || ""}
        aria-label={LABEL[v.status] || "Microphone"}
      >
        {v.muted || v.status === "blocked" || v.status === "off" ? "🔇" : "🎤"}
      </button>
    </span>
  );
}
