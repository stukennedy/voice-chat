# Voice Chat — Local Pipeline with Voice Cloning
#
# Components:
#   1. STT:  whisper.cpp (Metal-accelerated) via subprocess
#   2. LLM:  Ollama (llama3.2:3b) via HTTP API
#   3. TTS:  Pocket-TTS (Python API, voice cloning)
#
# Pipeline: mic -> whisper.cpp -> ollama -> pocket-tts -> speaker
#
# Usage:
#   uv run voice_chat.py
#
# First run: exports voice state from voices/clone_voice.wav -> voices/clone_voice.safetensors
# Subsequent runs: loads safetensors directly (fast)

import subprocess
import json
import time
import sys
import os
import tempfile
import select
import tty
import termios
import threading
import urllib.request
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from pocket_tts import TTSModel, export_model_state

# ── Config ──────────────────────────────────────────────────────────────

WHISPER_CLI = "/opt/homebrew/bin/whisper-cli"
WHISPER_MODEL = os.path.expanduser("~/.local/share/whisper-cpp/ggml-base.en.bin")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

# Voice cloning: uses safetensors if available, otherwise exports from wav
VOICE_WAV = str(Path(__file__).parent / "voices" / "clone_voice.wav")
VOICE_SAFETENSORS = str(Path(__file__).parent / "voices" / "clone_voice.safetensors")
# Fallback preset voice if clone files don't exist
TTS_VOICE_FALLBACK = "alba"

SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 30

SYSTEM_PROMPT = (
    "You are a friendly, concise conversational partner. "
    "Keep responses short and natural — 1-3 sentences unless asked for detail. "
    "No markdown, no bullet points, just spoken-style responses."
)

# ── TTS (Pocket-TTS with voice cloning) ─────────────────────────────────

_tts_model = None
_tts_voice_state = None

def init_tts():
    global _tts_model, _tts_voice_state
    print("🔊 Loading Pocket-TTS model...", flush=True)
    _tts_model = TTSModel.load_model()

    if os.path.exists(VOICE_SAFETENSORS):
        # Fast path: load pre-exported voice state from safetensors
        print(f"🔊 Loading cloned voice from {VOICE_SAFETENSORS}...", flush=True)
        _tts_voice_state = _tts_model.get_state_for_audio_prompt(VOICE_SAFETENSORS)
        print("🔊 TTS ready (cloned voice).", flush=True)
    elif os.path.exists(VOICE_WAV):
        # First run: export voice state from wav
        print(f"🔊 Exporting voice from {VOICE_WAV}...", flush=True)
        _tts_voice_state = _tts_model.get_state_for_audio_prompt(VOICE_WAV)
        export_model_state(_tts_voice_state, VOICE_SAFETENSORS)
        print(f"🔊 Saved voice state to {VOICE_SAFETENSORS}", flush=True)
        print("🔊 TTS ready (cloned voice).", flush=True)
    else:
        # Fallback: use preset voice
        print(f"🔊 No voice clone found, using preset '{TTS_VOICE_FALLBACK}'...", flush=True)
        _tts_voice_state = _tts_model.get_state_for_audio_prompt(TTS_VOICE_FALLBACK)
        print("🔊 TTS ready (preset voice).", flush=True)

def speak(text: str):
    """Generate audio from text and play it."""
    audio = _tts_model.generate_audio(_tts_voice_state, text)
    audio_np = audio.numpy()
    dur = len(audio_np) / _tts_model.sample_rate
    print(f"  🔊 Playing {dur:.1f}s of audio", flush=True)
    sd.play(audio_np, _tts_model.sample_rate)
    sd.wait()

# ── STT (whisper.cpp) ───────────────────────────────────────────────────

def record_audio() -> str:
    """Record audio from mic, save to temp wav, return path."""
    print("🎙️  Recording... (press Enter to stop)", flush=True)

    recording = []
    stop_flag = [False]

    def callback(indata, frames, time_info, status):
        if not stop_flag[0]:
            recording.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=callback)
    stream.start()

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        start = time.time()
        while time.time() - start < MAX_RECORD_SECONDS:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                sys.stdin.read(1)
                break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    stop_flag[0] = True
    stream.stop()
    stream.close()

    if not recording:
        print("  (no audio captured)", flush=True)
        return ""

    audio = np.concatenate(recording, axis=0)
    duration = len(audio) / SAMPLE_RATE
    print(f"  📼 Recorded {duration:.1f}s", flush=True)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_write(tmp.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    return tmp.name

def transcribe(audio_path: str) -> str:
    """Run whisper.cpp on the audio file, return transcript."""
    cmd = [
        WHISPER_CLI,
        "-m", WHISPER_MODEL,
        "-f", audio_path,
        "--language", "en",
        "-nt",
        "-np",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    os.unlink(audio_path)

    text = result.stdout.strip()
    lines = [l for l in text.split("\n") if l.strip() and not l.startswith("whisper_")]
    return " ".join(lines).strip()

# ── LLM (Ollama) ───────────────────────────────────────────────────────

_ollama_context = []

def chat(prompt: str) -> str:
    """Send prompt to Ollama, return response text."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "context": _ollama_context,
        "options": {
            "temperature": 0.7,
            "num_predict": 150,
        }
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    elapsed = time.time() - t0
    text = data.get("response", "").strip()
    _ollama_context.extend(data.get("context", []))

    print(f"  🧠 LLM responded in {elapsed:.2f}s ({len(text)} chars)", flush=True)
    return text

# ── Main Loop ──────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════╗")
    print("║   🦊 Local Voice Chat (fully offline)    ║")
    print("╠══════════════════════════════════════════╣")
    print("║  Press Enter to talk, type 'quit' to exit ║")
    print("╚══════════════════════════════════════════╝")
    print()

    init_tts()

    # Check ollama is running
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
    except Exception:
        print("⚠️  Ollama not running. Starting it...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)

    turn = 0
    while True:
        try:
            user_input = input(f"\n[{turn}] Press Enter to record (or type 'quit'): ")
            if user_input.strip().lower() in ("quit", "exit", "q"):
                print("👋 Bye!")
                break
            if user_input.strip():
                user_text = user_input.strip()
            else:
                audio_path = record_audio()
                if not audio_path:
                    continue
                t0 = time.time()
                user_text = transcribe(audio_path)
                stt_time = time.time() - t0
                print(f"  📝 STT ({stt_time:.2f}s): \"{user_text}\"", flush=True)
                if not user_text:
                    print("  (empty transcript, skipping)")
                    continue

            response = chat(user_text)
            print(f"  💬 Response: \"{response}\"", flush=True)

            speak(response)

            turn += 1

        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

if __name__ == "__main__":
    main()
