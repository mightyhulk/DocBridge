import { useEffect, useSyncExternalStore } from "react";
import { api, isSpeaking } from "./api.js";

/**
 * Always-on listening. After sign-in the browser keeps the microphone open, detects when the person starts and stops
 * speaking (voice-activity detection on the audio level), sends each utterance to the server's speech-to-text and hands
 * the text to whichever screen is open - exactly as if it had been typed. No button has to be pressed.
 *
 * It pauses itself while the assistant is speaking or thinking, so the assistant never answers its own voice.
 */
const FRAME = 2048;            // samples per analysis frame (~43 ms at 48 kHz)
const START_FRAMES = 3;        // consecutive loud frames that count as "speech started"
const END_SILENCE_S = 1.1;     // this much quiet ends the utterance
const MIN_SPEECH_S = 0.45;     // shorter blips (a cough, a click) are ignored
const MAX_UTTERANCE_S = 25;
const PREROLL_FRAMES = 6;      // keep ~250 ms before the trigger so the first word is not clipped
const COOLDOWN_MS = 700;       // after the assistant's voice ends
const NOISE_WORDS = new Set(["you", "thank you", "thanks", "thanks for watching", "bye", "uh", "um", "hmm", "mm"]);   // typical Whisper hallucinations on near-silence

const state = { status: "off", muted: false, error: "", level: 0, heard: "" };
const subs = new Set();
let snapshot = { ...state };
const emit = () => { snapshot = { ...state }; subs.forEach((f) => f()); };
const set = (patch) => {
  let changed = false;
  for (const k in patch) if (state[k] !== patch[k]) { state[k] = patch[k]; changed = true; }
  if (changed) emit();
};

let input = { fn: null, ready: () => false };        // the open screen registers itself here
let engine = null;
let lastSpeech = 0;

function wavBlob(float32, rate) {
  const pcm = new DataView(new ArrayBuffer(44 + float32.length * 2));
  const w = (o, s) => [...s].forEach((c, i) => pcm.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); pcm.setUint32(4, 36 + float32.length * 2, true); w(8, "WAVEfmt ");
  pcm.setUint32(16, 16, true); pcm.setUint16(20, 1, true); pcm.setUint16(22, 1, true);
  pcm.setUint32(24, rate, true); pcm.setUint32(28, rate * 2, true); pcm.setUint16(32, 2, true); pcm.setUint16(34, 16, true);
  w(36, "data"); pcm.setUint32(40, float32.length * 2, true);
  float32.forEach((v, i) => pcm.setInt16(44 + i * 2, Math.max(-1, Math.min(1, v)) * 32767, true));
  return new Blob([pcm], { type: "audio/wav" });
}

function downsample(buf, from, to = 16000) {
  if (from <= to) return { data: buf, rate: from };
  const ratio = from / to, out = new Float32Array(Math.floor(buf.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const a = Math.floor(i * ratio), b = Math.min(buf.length, Math.floor((i + 1) * ratio));
    let s = 0; for (let j = a; j < b; j++) s += buf[j];
    out[i] = s / Math.max(1, b - a);
  }
  return { data: out, rate: to };
}

async function startEngine() {
  if (engine) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    return set({ status: "blocked", error: window.isSecureContext ? "This browser cannot use the microphone." : "The microphone needs a secure page. Open this site with https:// (accept the certificate warning once)." });
  }
  set({ status: "starting", error: "" });
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
  } catch (e) {
    return set({ status: "blocked", error: e.name === "NotAllowedError" ? "Microphone permission was denied. Allow it in the browser's address bar, then reload." : "Microphone unavailable: " + e.message });
  }
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(FRAME, 1, 1);
  const mute = ctx.createGain(); mute.gain.value = 0;                // the processor must be connected to run, but must not play
  src.connect(proc); proc.connect(mute); mute.connect(ctx.destination);

  const frameS = FRAME / ctx.sampleRate;
  let floor = 0.01, loud = 0, quiet = 0, voiced = 0, speaking = false, chunks = [];
  const preroll = [];
  const reset = () => { speaking = false; loud = 0; quiet = 0; voiced = 0; chunks = []; preroll.length = 0; };

  const finish = async () => {
    const got = chunks, secs = voiced * frameS;
    reset();
    if (secs < MIN_SPEECH_S) return set({ status: "listening" });
    set({ status: "transcribing" });
    const all = new Float32Array(got.reduce((n, c) => n + c.length, 0));
    let o = 0; for (const c of got) { all.set(c, o); o += c.length; }
    const { data, rate } = downsample(all, ctx.sampleRate);
    try {
      const { text } = await api.stt(wavBlob(data, rate));
      const t = (text || "").trim();
      const bare = t.toLowerCase().replace(/[^a-z ]/g, "").trim();
      if (bare.length >= 2 && !(secs < 1.8 && NOISE_WORDS.has(bare))) {
        set({ status: "listening", heard: t });
        input.fn?.(t);
        return;
      }
    } catch (e) {
      if (/sign in/i.test(e.message)) return set({ status: "paused" });
    }
    set({ status: "listening" });
  };

  proc.onaudioprocess = (ev) => {
    const x = ev.inputBuffer.getChannelData(0);
    let sum = 0; for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
    const rms = Math.sqrt(sum / x.length);
    const st = state.status;

    if (ctx.state === "suspended") { ctx.resume().catch(() => {}); return; }
    if (isSpeaking()) lastSpeech = Date.now();
    const busy = state.muted || !input.ready() || Date.now() - lastSpeech < COOLDOWN_MS || st === "transcribing";
    if (busy) {                                              // not our turn to listen: drop anything half-heard
      if (speaking || chunks.length) reset();
      if (st !== "transcribing") set({ status: state.muted ? "muted" : "paused", level: 0 });
      return;
    }
    if (st !== "hearing" && st !== "listening") set({ status: "listening" });

    const threshold = Math.max(0.02, floor * 3);
    const isLoud = rms > threshold;
    if (!speaking) {
      if (!isLoud) floor = floor * 0.97 + rms * 0.03;        // follow the room's background noise
      preroll.push(new Float32Array(x)); if (preroll.length > PREROLL_FRAMES) preroll.shift();
      loud = isLoud ? loud + 1 : 0;
      if (loud >= START_FRAMES) { speaking = true; chunks = [...preroll]; voiced = loud; quiet = 0; set({ status: "hearing" }); }
      return;
    }
    chunks.push(new Float32Array(x));
    if (isLoud) { voiced++; quiet = 0; } else quiet += frameS;
    set({ level: Math.min(1, rms * 8) });
    if (quiet >= END_SILENCE_S || chunks.length * frameS > MAX_UTTERANCE_S) finish();
  };

  const resume = () => ctx.resume().catch(() => {});
  ["pointerdown", "keydown"].forEach((e) => window.addEventListener(e, resume));
  document.addEventListener("visibilitychange", resume);
  stream.getAudioTracks()[0].addEventListener("ended", () => { stopEngine(); set({ status: "blocked", error: "The microphone was disconnected." }); });
  engine = { stream, ctx, proc, src, mute, resume };
  set({ status: ctx.state === "running" ? "listening" : "needs-tap" });
  ctx.resume().then(() => state.status === "needs-tap" && set({ status: "listening" })).catch(() => {});
}

function stopEngine() {
  if (!engine) return;
  const { stream, ctx, proc, src, mute, resume } = engine;
  engine = null;
  proc.onaudioprocess = null;
  [src, proc, mute].forEach((n) => { try { n.disconnect(); } catch {} });
  stream.getTracks().forEach((t) => t.stop());
  ctx.close().catch(() => {});
  ["pointerdown", "keydown"].forEach((e) => window.removeEventListener(e, resume));
  document.removeEventListener("visibilitychange", resume);
  set({ status: "off", level: 0 });
}

export const voice = {
  start: startEngine,
  stop: stopEngine,
  setMuted: (m) => set({ muted: m }),
};

export const useVoice = () => useSyncExternalStore((f) => { subs.add(f); return () => subs.delete(f); }, () => snapshot);

/** A screen calls this to receive what the person says. `ready` is false while it is busy (then the mic is paused). */
export function useVoiceInput(fn, ready) {
  useEffect(() => {
    input = { fn, ready: () => !!ready };
    return () => { if (input.fn === fn) input = { fn: null, ready: () => false }; };
  });
}
