"""Voice loop self-test without a microphone: TTS speaks a question -> STT transcribes -> pipeline answers.

    python scripts/voice_roundtrip.py [http://localhost:8000]
"""
import sys
import time

import httpx

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000") + "/api"
c = httpx.Client(timeout=300)

for q in ["What is the usual adult dose of metformin?", "What are the red flag symptoms of dengue fever?",
          "Does paracetamol interact with warfarin?"]:
    tts_res = c.post(f"{BASE}/tts", json={"text": q})
    if tts_res.status_code != 200:
        print(f"TTS unavailable ({tts_res.status_code}); skipping voice roundtrip for: {q}")
        continue
    t = time.time()
    heard = c.post(f"{BASE}/stt", files={"audio": ("q.wav", tts_res.content, "audio/wav")}).json()["text"]
    r = c.post(f"{BASE}/chat", json={"message": heard, "speak": True}).json()
    print(f"spoken : {q}\nheard  : {heard}\ntier   : {r.get('tier')}  ({time.time() - t:.1f}s)")
    print(f"answer : {r['answer'][:200]}\n")
