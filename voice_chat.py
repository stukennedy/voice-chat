# Voice Chat — Local Pipeline with Voice Cloning + Streaming TTS
#
# Components:
#   1. Turn:  Silero VAD + Smart Turn v3 — hands-free end-of-turn detection
#   2. STT:   whisper.cpp (Metal-accelerated) via subprocess
#   3. LLM:   Ollama (llama3.2:3b) — streamed token-by-token
#   4. TTS:   Pocket-TTS — sentence-chunked streaming, plays while LLM still generates
#
# Pipeline: mic -> vad + smart-turn -> whisper.cpp -> ollama (stream)
#           -> sentence buffer -> pocket-tts -> speaker
#
# Turn taking is half-duplex: the mic closes while the bot speaks, so there is
# no barge-in and no echo. See turn_taking.py.
#
# Usage:
#   uv run voice_chat.py                 # hands-free
#   uv run voice_chat.py --push-to-talk  # press Enter to start/stop recording
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
import queue
import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from pocket_tts import TTSModel, export_model_state

from turn_taking import TurnListener

# ── Config ──────────────────────────────────────────────────────────────

WHISPER_CLI = "/opt/homebrew/bin/whisper-cli"
WHISPER_MODEL = os.path.expanduser("~/.local/share/whisper-cpp/ggml-base.en.bin")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

VOICE_WAV = str(Path(__file__).parent / "voices" / "clone_voice.wav")
VOICE_SAFETENSORS = str(Path(__file__).parent / "voices" / "clone_voice.safetensors")
TTS_VOICE_FALLBACK = "alba"

SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 30

# Spoken phrases that end a hands-free session
QUIT_PHRASES = ("quit", "exit", "goodbye", "good bye", "bye", "stop chatting")

SYSTEM_PROMPT = (
    "You are a friendly, concise conversational partner. "
    "Keep responses short and natural — 1-3 sentences unless asked for detail. "
    "No markdown, no bullet points, just spoken-style responses."
)

# Sentence chunking: split on these terminators
SENTENCE_ENDINGS = ".!?;:,\n"
# Minimum chunk length before we allow a split (avoid tiny fragments)
MIN_CHUNK_CHARS = 15
# Max chunk size (forces a split even without punctuation)
MAX_CHUNK_CHARS = 200

# ── TTS ─────────────────────────────────────────────────────────────────

_tts_model = None
_tts_voice_state = None

def init_tts():
    global _tts_model, _tts_voice_state
    print("🔊 Loading Pocket-TTS model...", flush=True)
    _tts_model = TTSModel.load_model()

    if os.path.exists(VOICE_SAFETENSORS):
        print(f"🔊 Loading cloned voice from safetensors...", flush=True)
        _tts_voice_state = _tts_model.get_state_for_audio_prompt(VOICE_SAFETENSORS)
        print("🔊 TTS ready (cloned voice).", flush=True)
    elif os.path.exists(VOICE_WAV):
        print(f"🔊 Exporting voice from wav...", flush=True)
        _tts_voice_state = _tts_model.get_state_for_audio_prompt(VOICE_WAV)
        export_model_state(_tts_voice_state, VOICE_SAFETENSORS)
        print("🔊 TTS ready (cloned voice).", flush=True)
    else:
        print(f"🔊 Using preset '{TTS_VOICE_FALLBACK}'...", flush=True)
        _tts_voice_state = _tts_model.get_state_for_audio_prompt(TTS_VOICE_FALLBACK)
        print("🔊 TTS ready (preset voice).", flush=True)

def generate_speech_chunk(text: str) -> np.ndarray:
    """Generate audio for a text chunk. Runs in a thread."""
    audio = _tts_model.generate_audio(_tts_voice_state, text)
    return audio.numpy()

def play_audio(audio_np: np.ndarray):
    """Play audio and wait for completion."""
    sd.play(audio_np, _tts_model.sample_rate)
    sd.wait()

# ── Sentence chunker ────────────────────────────────────────────────────

def chunk_sentences(text_stream, on_chunk=None):
    """
    Consume text incrementally, yield complete sentence chunks.
    text_stream: iterable of strings (tokens from LLM stream).
    Yields (chunk_text, is_final) tuples.
    """
    buffer = ""
    for token in text_stream:
        buffer += token
        # Check if we should flush
        should_flush = False

        # Flush on sentence-ending punctuation (if chunk is long enough)
        if len(buffer) >= MIN_CHUNK_CHARS:
            for ending in SENTENCE_ENDINGS:
                if buffer.rstrip().endswith(ending):
                    should_flush = True
                    break

        # Force flush if buffer is too long
        if len(buffer) >= MAX_CHUNK_CHARS:
            should_flush = True

        if should_flush:
            chunk = buffer.strip()
            buffer = ""
            if chunk:
                yield chunk

    # Flush remaining
    if buffer.strip():
        yield buffer.strip()

# ── LLM streaming ───────────────────────────────────────────────────────

_ollama_context = []

def stream_llm(prompt: str):
    """
    Stream LLM tokens from Ollama. Yields token strings.
    Updates _ollama_context with the final context.
    """
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": True,
        "context": _ollama_context,
        "options": {
            "temperature": 0.7,
            "num_predict": 200,
        }
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    full_response = ""

    with urllib.request.urlopen(req, timeout=60) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line:
                continue
            data = json.loads(line)
            token = data.get("response", "")
            full_response += token
            if token:
                yield token

            if data.get("done", False):
                # Update context for conversation memory
                ctx = data.get("context", [])
                if ctx:
                    _ollama_context.clear()
                    _ollama_context.extend(ctx)
                break

def chat_streaming(prompt: str):
    """
    Stream LLM output, chunk into sentences, generate TTS in parallel.
    Plays audio chunks as they're ready — first sentence starts speaking
    while LLM is still generating the rest.
    """
    t0 = time.time()
    first_audio_played = False

    # Create a token generator
    token_stream = stream_llm(prompt)

    # Chunk tokens into sentences
    sentence_stream = chunk_sentences(token_stream)

    # Pipeline: for each sentence chunk, generate TTS in a worker thread
    # while continuing to consume the LLM stream
    audio_queue = queue.Queue(maxsize=4)  # limit backlog

    def tts_worker():
        """Generate TTS for each chunk, push audio to queue."""
        for chunk in sentence_stream:
            t_gen = time.time()
            audio = generate_speech_chunk(chunk)
            gen_time = time.time() - t_gen
            dur = len(audio) / _tts_model.sample_rate
            audio_queue.put((chunk, audio, gen_time, dur))

        # Sentinel: signal end of stream
        audio_queue.put(None)

    # Start TTS worker thread
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    # Main thread: play audio chunks as they arrive
    chunks_played = 0
    full_text = ""

    while True:
        item = audio_queue.get(timeout=60)
        if item is None:
            break

        chunk, audio, gen_time, dur = item
        full_text += chunk + " "

        if not first_audio_played:
            total_ttfb = time.time() - t0
            print(f"  🎵 First audio: {total_ttfb:.2f}s (chunk: {gen_time:.2f}s gen, {dur:.1f}s audio)", flush=True)
            first_audio_played = True
        else:
            print(f"  🎵 Chunk: {gen_time:.2f}s gen, {dur:.1f}s audio", flush=True)

        # Print the chunk as it's spoken
        print(f"  💬 \"{chunk}\"", flush=True)

        play_audio(audio)
        chunks_played += 1

    tts_thread.join(timeout=5)

    total = time.time() - t0
    print(f"  🧠 Total: {total:.2f}s, {chunks_played} chunks", flush=True)

    return full_text.strip()

# ── STT ────────────────────────────────────────────────────────────────

def write_temp_wav(audio: np.ndarray) -> str:
    """Write mono float32 audio to a temp 16-bit wav for whisper.cpp."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_write(tmp.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    return tmp.name

def record_audio() -> str:
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

    return write_temp_wav(audio)

def transcribe(audio_path: str) -> str:
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

# ── Main ───────────────────────────────────────────────────────────────

def ensure_ollama():
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
    except Exception:
        print("⚠️  Ollama not running. Starting it...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)

def is_quit(text: str) -> bool:
    cleaned = text.lower().strip().strip(".!?,")
    return cleaned in QUIT_PHRASES

BANNER_WIDTH = 56

def banner(hands_free: bool):
    mode = "hands-free" if hands_free else "push-to-talk"
    hint = ("Just start talking — say 'goodbye' or Ctrl-C to exit"
            if hands_free else
            "Press Enter to talk, type 'quit' to exit")

    def row(text, indent=2, emoji_cells=0):
        # Emoji render two columns wide but count as one character
        pad = BANNER_WIDTH - indent - len(text) - emoji_cells
        return "║" + " " * indent + text + " " * max(pad, 0) + "║"

    print()
    print("╔" + "═" * BANNER_WIDTH + "╗")
    print(row(f"🦊 Local Voice Chat ({mode}, offline)", indent=3, emoji_cells=1))
    print("╠" + "═" * BANNER_WIDTH + "╣")
    print(row(hint))
    print("╚" + "═" * BANNER_WIDTH + "╝")
    print()

def next_prompt_hands_free(listener) -> str | None:
    """Capture one spoken turn and transcribe it. None means 'nothing usable'."""
    audio = listener.listen()
    if audio is None:
        return None

    t0 = time.time()
    text = transcribe(write_temp_wav(audio))
    print(f"  📝 STT ({time.time() - t0:.2f}s): \"{text}\"", flush=True)
    return text or None

def next_prompt_push_to_talk(turn: int) -> str | None:
    """Legacy path: Enter to record, Enter again to stop. Typed text also works."""
    typed = input(f"\n[{turn}] Press Enter to record (or type 'quit'): ").strip()
    if is_quit(typed):
        return "__quit__"
    if typed:
        return typed

    audio_path = record_audio()
    if not audio_path:
        return None

    t0 = time.time()
    text = transcribe(audio_path)
    print(f"  📝 STT ({time.time() - t0:.2f}s): \"{text}\"", flush=True)
    if not text:
        print("  (empty transcript, skipping)")
        return None
    return text

def main():
    hands_free = "--push-to-talk" not in sys.argv
    banner(hands_free)

    init_tts()
    listener = TurnListener() if hands_free else None
    ensure_ollama()

    turn = 0
    while True:
        try:
            if hands_free:
                user_text = next_prompt_hands_free(listener)
            else:
                user_text = next_prompt_push_to_talk(turn)

            if user_text is None:
                continue
            if user_text == "__quit__" or is_quit(user_text):
                print("👋 Bye!")
                break

            # Stream LLM → chunk → TTS (plays while generating).
            # In hands-free mode the mic is already closed, so this can't
            # hear itself.
            print("  🧠 Thinking + speaking...", flush=True)
            chat_streaming(user_text)

            turn += 1

        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
