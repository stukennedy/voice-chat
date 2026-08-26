# Voice Chat — Local Pipeline with Voice Cloning + Streaming TTS
#
# Components:
#   1. Turn:  Silero VAD + Smart Turn v3 — hands-free end-of-turn detection
#   2. STT:   whisper.cpp (Metal-accelerated) via subprocess — multilingual
#   3. LLM:   Ollama (qwen2.5:7b) — streamed token-by-token
#   4. TTS:   Pocket-TTS — sentence-chunked streaming, plays while LLM still generates
#
# Pipeline: mic -> vad + smart-turn -> whisper.cpp -> ollama (stream)
#           -> sentence buffer -> pocket-tts -> speaker
#
# Turn taking is half-duplex: the mic closes while the bot speaks, so there is
# no barge-in and no echo. See turn_taking.py.
#
# Languages: english, french, german, italian, portuguese, spanish.
# Pocket-TTS ships one set of weights per language, so the language is detected from
# the first utterance and then locked for the rest of the session.
#
# Usage:
#   uv run voice_chat.py                        # hands-free, auto-detect language
#   uv run voice_chat.py --push-to-talk         # press Enter to start/stop recording
#   uv run voice_chat.py --lang de              # pinned to German
#   uv run voice_chat.py --lang de --text "..." # non-interactive smoke test (no mic, no LLM)
#
# First run per language: exports voice state from voices/clone_voice.wav
#   -> voices/clone_voice.<language>.safetensors
# Subsequent runs: loads that safetensors directly (fast)

import argparse
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
# Multilingual weights (no .en suffix). Use ggml-small.bin for better non-English
# accuracy at roughly 2x the transcription time.
WHISPER_MODEL = os.path.expanduser("~/.local/share/whisper-cpp/ggml-base.bin")
OLLAMA_URL = "http://localhost:11434/api/generate"

# One model for every language. A 3B model is faster per token, but only the first sentence
# is latency-critical (after that, playback is slower than generation), and the gap there is
# ~80ms — not worth evicting and reloading gigabytes of weights on a language switch.
OLLAMA_MODEL = "qwen2.5:7b"

# Whisper ISO code -> (Pocket-TTS config name, human-readable name)
# Pocket-TTS has 6-layer and 24-layer variants of most languages; the 6-layer ones are
# ~4x cheaper and are what keeps latency near the English numbers. French is 24-layer only.
LANGUAGES = {
    "en": ("english", "English"),
    "fr": ("french_24l", "French"),
    "de": ("german", "German"),
    "it": ("italian", "Italian"),
    "pt": ("portuguese", "Portuguese"),
    "es": ("spanish", "Spanish"),
}
DEFAULT_LANGUAGE = "en"

# Preset voice per Pocket-TTS language, used when there is no cloned voice available.
# Preset voice states live under languages/<lang>/embeddings/ in the model repo, so a
# preset name from one language does not exist in another.
PRESET_VOICE = {
    "english": "alba",
    "french_24l": "estelle",
    "german": "juergen",
    "italian": "giovanni",
    "portuguese": "rafael",
    "spanish": "lola",
}

VOICES_DIR = Path(__file__).parent / "voices"
VOICE_WAV = VOICES_DIR / "clone_voice.wav"

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

# ── Language helpers ────────────────────────────────────────────────────

def tts_language_for(lang_code: str) -> str:
    """Pocket-TTS config name for a whisper language code."""
    return LANGUAGES[lang_code][0]

def language_name(lang_code: str) -> str:
    return LANGUAGES[lang_code][1]

def voice_safetensors_for(tts_lang: str) -> Path:
    """Cloned voice states are encoded by the loaded model, so each language needs its own."""
    return VOICES_DIR / f"clone_voice.{tts_lang}.safetensors"

def resolve_language(detected: str) -> str:
    """Map a detected whisper code onto a supported language, falling back to the default."""
    if detected in LANGUAGES:
        return detected
    print(
        f"  ⚠️  Detected language '{detected}' is not supported by Pocket-TTS "
        f"(supported: {', '.join(LANGUAGES)}) — falling back to {language_name(DEFAULT_LANGUAGE)}.",
        flush=True,
    )
    return DEFAULT_LANGUAGE

# ── TTS ─────────────────────────────────────────────────────────────────

_tts_model = None
_tts_voice_state = None
_tts_language = None  # Pocket-TTS config name currently loaded

def _resolve_voice_state(tts_lang: str):
    """Cloned state for this language if we have one, else clone it, else a preset voice."""
    cached = voice_safetensors_for(tts_lang)

    if cached.exists():
        print(f"🔊 Loading cloned voice ({tts_lang}) from safetensors...", flush=True)
        return _tts_model.get_state_for_audio_prompt(str(cached))

    preset = PRESET_VOICE[tts_lang]

    if not VOICE_WAV.exists():
        print(f"🔊 No {VOICE_WAV.name} found — using preset voice '{preset}'.", flush=True)
        return _tts_model.get_state_for_audio_prompt(preset)

    # Cloning needs the gated kyutai/pocket-tts weights. Without an HF login, Pocket-TTS
    # silently falls back to the no-cloning weights, and only fails when we clone.
    if not _tts_model.has_voice_cloning:
        print(
            f"  ⚠️  Voice cloning weights unavailable for {tts_lang} — using preset '{preset}'.\n"
            f"      Accept the terms at https://huggingface.co/kyutai/pocket-tts and run "
            f"`uvx hf auth login`, then re-run to clone your voice into this language.",
            flush=True,
        )
        return _tts_model.get_state_for_audio_prompt(preset)

    print(f"🔊 Exporting cloned voice for {tts_lang} from wav...", flush=True)
    state = _tts_model.get_state_for_audio_prompt(str(VOICE_WAV))
    export_model_state(state, str(cached))
    return state

def init_tts(lang_code: str):
    """Load the Pocket-TTS weights for a language. No-op if already loaded."""
    global _tts_model, _tts_voice_state, _tts_language

    tts_lang = tts_language_for(lang_code)
    if _tts_language == tts_lang:
        return

    t0 = time.time()
    print(f"🔊 Loading Pocket-TTS model ({tts_lang})...", flush=True)
    _tts_model = TTSModel.load_model(language=tts_lang)
    _tts_language = tts_lang
    _tts_voice_state = _resolve_voice_state(tts_lang)
    print(f"🔊 TTS ready — {language_name(lang_code)} ({time.time() - t0:.1f}s).", flush=True)

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
_ollama_context_lang = None

def system_prompt_for(lang_code: str) -> str:
    """Small models drift back to English mid-conversation without an explicit instruction."""
    return f"{SYSTEM_PROMPT} Always reply in {language_name(lang_code)}."

def stream_llm(prompt: str, lang_code: str):
    """
    Stream LLM tokens from Ollama. Yields token strings.
    Updates _ollama_context with the final context.
    """
    global _ollama_context_lang

    # The system prompt is baked into the context on the first call, so a language change
    # (e.g. a typed English turn followed by a spoken German one) needs a fresh context.
    if _ollama_context_lang != lang_code:
        _ollama_context.clear()
        _ollama_context_lang = lang_code

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt_for(lang_code),
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

def speak_stream(sentence_stream, t0=None):
    """
    Generate TTS for each sentence chunk in a worker thread and play chunks as they
    become ready, so audio starts before the text stream has finished.
    """
    if t0 is None:
        t0 = time.time()
    first_audio_played = False

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

def chat_streaming(prompt: str, lang_code: str):
    """
    Stream LLM output, chunk into sentences, generate TTS in parallel.
    Plays audio chunks as they're ready — first sentence starts speaking
    while LLM is still generating the rest.
    """
    t0 = time.time()
    token_stream = stream_llm(prompt, lang_code)
    sentence_stream = chunk_sentences(token_stream)
    return speak_stream(sentence_stream, t0)

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

def transcribe(audio_path: str, language: str = "auto"):
    """
    Transcribe an audio file. `language` is a whisper code, or "auto" to detect.
    Returns (text, detected_language_code).
    """
    cmd = [
        WHISPER_CLI,
        "-m", WHISPER_MODEL,
        "-f", audio_path,
        "--language", language,
        "-nt",
        "-np",
        "-oj",
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    # whisper.cpp writes the JSON result alongside the input file.
    json_path = audio_path + ".json"
    text, detected = "", language

    try:
        with open(json_path) as f:
            data = json.load(f)
        # "params.language" echoes the request ("auto"); the detection is in "result.language".
        detected = data.get("result", {}).get("language", language)
        text = " ".join(
            seg.get("text", "") for seg in data.get("transcription", [])
        ).strip()
    except (OSError, json.JSONDecodeError) as e:
        print(f"  ⚠️  Could not read whisper output: {e}", flush=True)
    finally:
        for path in (json_path, audio_path):
            try:
                os.unlink(path)
            except OSError:
                pass

    return text, detected

# ── Main ───────────────────────────────────────────────────────────────

def speak_once(lang_code: str, text: str):
    """Non-interactive smoke test: speak text in a language, no mic and no LLM."""
    init_tts(lang_code)
    print(f"  💬 Speaking {language_name(lang_code)}: \"{text}\"", flush=True)
    speak_stream(chunk_sentences(iter([text])))

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

def next_prompt_hands_free(listener, language: str):
    """Capture one spoken turn and transcribe it. None means 'nothing usable'."""
    audio = listener.listen()
    if audio is None:
        return None

    t0 = time.time()
    text, detected = transcribe(write_temp_wav(audio), language)
    print(f"  📝 STT ({time.time() - t0:.2f}s): \"{text}\"", flush=True)
    if not text:
        return None
    return text, detected

def next_prompt_push_to_talk(turn: int, language: str):
    """Legacy path: Enter to record, Enter again to stop. Typed text also works."""
    typed = input(f"\n[{turn}] Press Enter to record (or type 'quit'): ").strip()
    if is_quit(typed):
        return "__quit__", None
    if typed:
        # Typed input carries no audio, so there is nothing to detect a language from.
        return typed, None

    audio_path = record_audio()
    if not audio_path:
        return None

    t0 = time.time()
    text, detected = transcribe(audio_path, language)
    print(f"  📝 STT ({time.time() - t0:.2f}s): \"{text}\"", flush=True)
    if not text:
        # Don't lock the session to a language guessed from silence or noise.
        print("  (empty transcript, skipping)")
        return None
    return text, detected

def main():
    parser = argparse.ArgumentParser(description="Local streaming voice chat.")
    parser.add_argument(
        "--push-to-talk",
        action="store_true",
        help="Press Enter to start/stop recording instead of hands-free turn detection.",
    )
    parser.add_argument(
        "--lang",
        choices=sorted(LANGUAGES),
        help="Pin the language instead of detecting it from the first utterance.",
    )
    parser.add_argument(
        "--text",
        help="Speak this text and exit (smoke test). Requires --lang.",
    )
    args = parser.parse_args()

    if args.text:
        if not args.lang:
            parser.error("--text requires --lang")
        speak_once(args.lang, args.text)
        return

    hands_free = not args.push_to_talk
    banner(hands_free)

    # Language is locked for the session because switching means reloading TTS weights.
    session_lang = args.lang
    if session_lang:
        print(f"🌍 Language pinned to {language_name(session_lang)}.", flush=True)
    else:
        print("🌍 Language will be detected from your first utterance.", flush=True)
    print(f"🧠 LLM: {OLLAMA_MODEL} (all languages)", flush=True)
    print(f"📝 STT: {os.path.basename(WHISPER_MODEL)}", flush=True)

    # Load English up front so English sessions start speaking immediately; a different
    # detected language swaps the model once, on the first turn.
    init_tts(session_lang or DEFAULT_LANGUAGE)
    listener = TurnListener() if hands_free else None
    ensure_ollama()

    turn = 0
    while True:
        try:
            language = session_lang or "auto"
            if hands_free:
                prompt = next_prompt_hands_free(listener, language)
            else:
                prompt = next_prompt_push_to_talk(turn, language)

            if prompt is None:
                continue
            user_text, detected = prompt
            if user_text == "__quit__" or is_quit(user_text):
                print("👋 Bye!")
                break

            # `detected` is None for typed input, which leaves the session unlocked.
            if session_lang is None and detected is not None:
                session_lang = resolve_language(detected)
                print(
                    f"  🌍 Detected {language_name(session_lang)} — locked for this session.",
                    flush=True,
                )
                init_tts(session_lang)
            turn_lang = session_lang or DEFAULT_LANGUAGE

            # Stream LLM → chunk → TTS (plays while generating).
            # In hands-free mode the mic is already closed, so this can't
            # hear itself.
            print(f"  🧠 Thinking + speaking [{OLLAMA_MODEL}]...", flush=True)
            chat_streaming(user_text, turn_lang)

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
