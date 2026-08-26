# 🦊 Local Voice Chat

Fully offline voice chat pipeline running on Apple Silicon (M4 Mac mini). No API keys, no cloud, fully private.

- **Turn taking:** [Silero VAD](https://github.com/snakers4/silero-vad) + [Smart Turn v3](https://huggingface.co/pipecat-ai/smart-turn-v3) — hands-free, ~15ms CPU
- **STT:** [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (Metal-accelerated) — ~0.3s for short utterances
- **LLM:** [Ollama](https://ollama.com) (qwen2.5:7b) — ~50 tok/s warm, all six languages
- **TTS:** [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts) — 6x realtime, 200ms to first audio, voice cloning

**Total round-trip latency: ~700ms–1s** for a short exchange.

Speaks **English, French, German, Italian, Portuguese and Spanish** — see [Languages](#languages).

## Prerequisites

```bash
# 1. whisper.cpp (via Homebrew)
brew install whisper-cpp

# 2. Whisper model — base multilingual (148MB, detects and transcribes all 6 languages)
mkdir -p ~/.local/share/whisper-cpp
curl -L -o ~/.local/share/whisper-cpp/ggml-base.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
# Optional: ggml-small.bin (466MB) — better non-English accuracy, ~2x slower

# 3. Ollama + models
brew install ollama  # or download from ollama.com
ollama pull qwen2.5:7b    # 4.7GB — handles all six languages

# 4. HuggingFace auth — the voice cloning weights are gated
#    a) Accept the terms at: https://huggingface.co/kyutai/pocket-tts
#    b) Log in locally. This opens a browser device-code flow (it prints a code
#       to enter at hf.co/oauth/device) and blocks until you authorise:
uvx hf auth login
#    Non-interactive alternative, with a read token from
#    https://huggingface.co/settings/tokens:
uvx hf auth login --token hf_xxxxx

# Skipping this is fine — the app runs and falls back to preset voices.
```

## Install

```bash
git clone https://github.com/stukennedy/voice-chat.git
cd voice-chat
uv sync
```

## Languages

Pocket-TTS ships **one set of weights per language**, so the language is detected from your
first utterance with `whisper -l auto` and then **locked for the session** — switching means
loading different TTS weights and a different cloned voice state.

| Code | Language | Pocket-TTS model | Preset voice (fallback) |
|------|----------|------------------|--------------|
| `en` | English | `english` (6-layer) | `alba` |
| `de` | German | `german` (6-layer) | `juergen` |
| `it` | Italian | `italian` (6-layer) | `giovanni` |
| `pt` | Portuguese | `portuguese` (6-layer) | `rafael` |
| `es` | Spanish | `spanish` (6-layer) | `lola` |
| `fr` | French | `french_24l` (24-layer) | `estelle` |

**French is the slow one.** Kyutai only publishes a 24-layer model for French — 4x the layers
of the others, which roughly triples time-to-first-audio (~1.0s vs ~0.3s). Every other
language has a 6-layer model.

Anything outside these six (whisper knows ~99 languages) falls back to English with a warning.

With a HuggingFace login your cloned voice is used for **every** language, each exported to
its own state file on first use. The preset column is what you get without one.

### Pinning or testing a language

```bash
uv run voice_chat.py                       # auto-detect from first utterance
uv run voice_chat.py --lang it             # skip detection, pin to Italian
uv run voice_chat.py --lang it --push-to-talk   # combines with either turn-taking mode

# Smoke test a language without a mic or the LLM:
uv run voice_chat.py --lang fr --text "Bonjour, comment allez-vous?"
```

**Detection keys on pronunciation, not vocabulary.** Whisper classifies how the speech
sounds, so correctly-pronounced Italian is detected from as little as 1.6s, while Italian
spoken with a strong English accent is detected as English however long the sample. Two
things that do *not* help, both measured: a longer utterance, and `ggml-small.bin` — it
returned identical results to `ggml-base.bin` on every sample tested.

If a language isn't detected reliably in your voice, pin it with `--lang`. That skips
detection entirely and is the intended fix — the language is locked for the session either
way, so pinning costs you nothing.

## Voice Cloning

The script uses a voice sample in `voices/clone_voice.wav` (18s, 24kHz mono). The first time
you use each language it exports a `.safetensors` voice state for that language, so later
runs load in a fraction of a second.

### Use your own voice

Replace `voices/clone_voice.wav` with your own clean recording:

```bash
# Convert any audio to the right format (24kHz mono 16-bit WAV)
ffmpeg -i your_recording.wav -ar 24000 -ac 1 -sample_fmt s16 voices/clone_voice.wav

# Delete the cached safetensors so they re-export
rm voices/clone_voice.*.safetensors
```

**Cloned voice states are per-language.** `get_state_for_audio_prompt` encodes your wav
through the currently loaded model, so a state exported against the English weights is
meaningless to the German ones. Each language gets its own file, exported on first use:

```
voices/clone_voice.wav                     # your recording — the source of truth
voices/clone_voice.english.safetensors     # exported on first English run
voices/clone_voice.german.safetensors      # exported on first German run
...
```

Bear in mind an English speaker's cloned timbre reading French will carry an English accent.

**Tips for good cloning:**
- 10–30 seconds of clean single-speaker speech
- No background noise or music
- Natural conversational tone
- If rough, clean it first with [Adobe Podcast Enhance](https://podcast.adobe.com/enhance) (free)

### Use a preset voice instead

Delete the wav and the safetensors in `voices/` and each language falls back to its preset
(see the [Languages](#languages) table). To pick different ones, edit `PRESET_VOICE` in
`voice_chat.py`.

Preset voice states are stored **per language** in the model repo, so a name from one
language does not exist in another — `alba` is English-only, `giovanni` is Italian-only.
The full English catalogue: `alba`, `anna`, `azelma`, `charles`, `cosette`, `eponine`,
`fantine`, `george`, `jane`, `jean`, `javert`, `marius`, `mary`, `michael`, `paul`,
`peter_yearsley`, `stuart_bell`, `vera`.

**Without a HuggingFace login, cloning is unavailable** — the gated `kyutai/pocket-tts`
weights can't be downloaded, Pocket-TTS silently falls back to the no-cloning weights, and
the app uses the preset voice with a warning. Accept the terms and run `uvx hf auth login`
to enable cloning.

### Export a voice via CLI

```bash
# Export voice state from a wav
uvx pocket-tts export-voice --voice ./your_voice.wav --output ./voices/your_voice.safetensors

# Generate speech with it
uvx pocket-tts generate --voice ./voices/your_voice.safetensors --text "Hello world"
```

## Run

```bash
uv run voice_chat.py                 # hands-free (default)
uv run voice_chat.py --push-to-talk  # Enter to start, Enter to stop
```

Hands-free:

1. **Just start talking** → Silero VAD opens a turn within ~64ms
2. **Stop talking** → after 200ms of silence, Smart Turn v3 decides whether
   you're actually finished or just pausing mid-sentence; if unsure it keeps
   listening and re-checks every 200ms
3. whisper.cpp transcribes and detects the language (~0.3s)
4. On the first turn the language is locked, and its TTS model loaded if it isn't already
5. qwen2.5:7b generates a response, instructed to reply in that language (~500ms)
6. Pocket-TTS speaks the response in the cloned voice (~200ms to first audio)
7. Audio plays through speakers, mic closed
8. Say "goodbye" (or Ctrl-C) to exit

Push-to-talk follows the same flow, with Enter to start and stop recording instead of
steps 1–2. Typed input also works there; it carries no audio, so it never locks a language.

## Turn taking

Two models cooperate, which is how Smart Turn is designed to be used:

- **Silero VAD** runs on every 32ms frame (~3ms) and answers "is there sound?"
- **Smart Turn v3** runs only *during* silence (~13ms) and answers "are they
  finished?" — from the raw waveform, not the transcript. It's a Whisper-tiny
  encoder plus a linear head, 8MB int8 ONNX, downloaded on first run.

The distinction matters: a 400ms pause mid-sentence and a 400ms pause at the
end of a turn look identical to a VAD. Smart Turn scores the first around 0.01
and the second around 0.97, so you don't get interrupted mid-thought and you
don't wait out a fixed timeout when you're plainly done.

Turn taking is **half-duplex** — the mic stream is closed while the bot speaks.
That rules out barge-in, but it also means TTS output can never be fed back
into the detector, so no echo cancellation is needed and speakers work fine.

Tunables live at the top of `turn_taking.py`:

| Setting | Default | Effect |
|---------|---------|--------|
| `SILENCE_PROBE_MS` | `200` | Silence before the first Smart Turn check — the floor on response latency |
| `PROBE_INTERVAL_MS` | `200` | How often to re-check while silence continues |
| `MAX_SILENCE_MS` | `3000` | Hard fallback; ends the turn however Smart Turn votes |
| `SMART_TURN_THRESHOLD` | `0.5` | Raise to interrupt less, lower to respond faster |
| `VAD_THRESHOLD` | `0.5` | Raise in a noisy room |
| `PREROLL_SECONDS` | `0.3` | Audio kept from before speech onset, so word one isn't clipped |

## Configuration

Edit `voice_chat.py` to change:

| Setting | Default | Options |
|---------|---------|---------|
| `OLLAMA_MODEL` | `qwen2.5:7b` | Any Ollama model that speaks your languages |
| `WHISPER_MODEL` | `ggml-base.bin` | `ggml-small.bin` (better non-English, ~2x slower) |
| `LANGUAGES` | 6 languages | Maps whisper codes → Pocket-TTS models; swap `german` for `german_24l` etc. |
| `PRESET_VOICE` | per-language | Fallback voice when no clone is available |
| `DEFAULT_LANGUAGE` | `en` | Used for typed input and unsupported detections |
| `MAX_RECORD_SECONDS` | `30` | Max recording length |
| `SENTENCE_ENDINGS` | `.!?` | Where chunks split; `,;:` are excluded on purpose (see below) |
| `TTS_TARGET_RMS` | `0.06` | Loudness every chunk is matched to |
| `SYSTEM_PROMPT` | concise conversational | Edit to change LLM personality |

## Performance (M4 Mac mini, 16GB)

| Component | Metric |
|-----------|--------|
| Silero VAD | ~3ms per 32ms frame |
| Smart Turn v3 (int8 ONNX) | ~13ms per check, 8MB model |
| whisper.cpp (base multilingual) | ~0.2s for a 6s utterance, including `-l auto` detection |
| qwen2.5:7b (warm) | ~50 tok/s, 0.26s to first sentence |
| Pocket-TTS, 6-layer (en/de/it/pt/es) | 0.05–0.14s to first audio, warm, short sentence |
| Pocket-TTS, 24-layer (fr) | 0.47s to first audio — ~4x the others, matching 4x the layers |
| Model load (Pocket-TTS) | 20–55s first run (downloads weights), 0.16–0.18s cached (0.47s fr) |
| Voice state load (safetensors) | <0.5s |
| Voice state export (from wav) | ~1s for 18s clip, once per language |

Time to first audio scales with chunk length; the figures above are a short sentence. A
full-length opening chunk in normal conversation lands around 0.2–0.5s.

Each language's weights download once (~20s for a 6-layer model, ~55s for French) and are
cached in `~/.cache/huggingface`. After that, switching language costs well under a second.

**Why one LLM for every language.** A 3B model generates roughly twice the tokens per second
of `qwen2.5:7b`, but that barely matters here: TTS playback is slower than generation, so
after the first sentence the LLM always has slack. Measured time to first sentence was 0.18s
for `llama3.2:3b` against 0.26s for `qwen2.5:7b` — an 80ms difference. Using a separate model
per language would trade that 80ms for a 30–60s model eviction and reload the first time you
switch language, so the pipeline uses one model throughout.

### Chunking and loudness

The LLM's output is split into chunks and spoken as it streams. Two details matter:

- **Chunks split on `.!?` only.** Pocket-TTS renders each chunk as a complete utterance
  with sentence-final prosody, so splitting on commas put a 5–7dB fade on every comma —
  audible as the voice dipping mid-sentence. The one exception is the *first* chunk of a
  response, which may also split on `,;:` so speech starts promptly when the opening
  sentence is long. After that, playback outpaces generation and there's nothing to gain.
- **Chunks are loudness-matched.** Generated independently, chunk level varied by up to
  12dB. Each is now normalised to `TTS_TARGET_RMS`, measured over voiced samples only so
  a chunk padded with silence isn't amplified to compensate.
- **Playback uses one long-lived output stream.** `sd.play()` opens and closes a CoreAudio
  stream per call, measured at 390–640ms of dead air *per chunk* regardless of audio
  length — seconds of silence punched into a multi-chunk response. Writing into a single
  persistent stream brings that to ~0.04s total. `drain_playback()` then waits out the
  device latency before the mic reopens, which is what keeps half-duplex honest.

## Architecture

```
mic → sounddevice (16kHz) → Silero VAD (32ms frames)
                                  ↓ silence
                            Smart Turn v3 (ONNX) → turn complete?
                                  ↓ yes
                            whisper.cpp (Metal) → text + detected language
                                  ↓
                            language locked for session
                                  ↓
  speaker ← Pocket-TTS (CPU, 24kHz, per-language model) ← Ollama (qwen2.5:7b)
     (mic closed while speaking — half-duplex)
```

All processing is local. No network calls except Ollama on localhost:11434.

## License

MIT
