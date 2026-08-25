# 🦊 Local Voice Chat

Fully offline voice chat pipeline running on Apple Silicon (M4 Mac mini). No API keys, no cloud, fully private.

- **STT:** [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (Metal-accelerated) — ~0.3s for short utterances
- **LLM:** [Ollama](https://ollama.com) (llama3.2:3b) — ~37 tok/s warm
- **TTS:** [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts) — 6x realtime, 200ms to first audio, voice cloning

**Total round-trip latency: ~700ms–1s** for a short exchange.

## Prerequisites

```bash
# 1. whisper.cpp (via Homebrew)
brew install whisper-cpp

# 2. Whisper model — base.en (141MB, good for realtime English)
mkdir -p ~/.local/share/whisper-cpp
curl -L -o ~/.local/share/whisper-cpp/ggml-base.en.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"

# 3. Ollama + a model
brew install ollama  # or download from ollama.com
ollama pull llama3.2:3b   # 2GB, ~37 tok/s on M4
# Optional: ollama pull qwen3:8b  (smarter but slower, uses more RAM)

# 4. HuggingFace auth (for voice cloning model — gated access)
#    a) Accept terms at: https://huggingface.co/kyutai/pocket-tts
#    b) Log in locally:
pip install huggingface-hub
huggingface-cli login  # paste a read token from https://huggingface.co/settings/tokens
```

## Install

```bash
git clone https://github.com/stukennedy/voice-chat.git
cd voice-chat
uv sync
```

## Voice Cloning

The script uses a voice sample in `voices/clone_voice.wav` (18s, 24kHz mono). On first run, it exports a `.safetensors` voice state for fast subsequent loads.

### Use your own voice

Replace `voices/clone_voice.wav` with your own clean recording:

```bash
# Convert any audio to the right format (24kHz mono 16-bit WAV)
ffmpeg -i your_recording.wav -ar 24000 -ac 1 -sample_fmt s16 voices/clone_voice.wav

# Delete the cached safetensors so it re-exports
rm voices/clone_voice.safetensors
```

**Tips for good cloning:**
- 10–30 seconds of clean single-speaker speech
- No background noise or music
- Natural conversational tone
- If rough, clean it first with [Adobe Podcast Enhance](https://podcast.adobe.com/enhance) (free)

### Use a preset voice instead

Edit `voice_chat.py` and change `TTS_VOICE_FALLBACK` to one of:
`alba`, `anna`, `azelma`, `charles`, `cosette`, `eponine`, `estelle`, `fantine`,
`george`, `giovanni`, `jane`, `jean`, `javert`, `juergen`, `lola`, `marius`,
`mary`, `michael`, `paul`, `peter_yearsley`, `rafael`, `stuart_bell`, `vera`

Then delete the safetensors file and the wav file in `voices/`.

### Export a voice via CLI

```bash
# Export voice state from a wav
uvx pocket-tts export-voice --voice ./your_voice.wav --output ./voices/your_voice.safetensors

# Generate speech with it
uvx pocket-tts generate --voice ./voices/your_voice.safetensors --text "Hello world"
```

## Run

```bash
uv run voice_chat.py
```

1. **Press Enter** → starts recording from mic
2. **Press Enter again** → stops recording
3. whisper.cpp transcribes (~0.3s)
4. llama3.2:3b generates a response (~500ms)
5. Pocket-TTS speaks the response in the cloned voice (~200ms to first audio)
6. Audio plays through speakers
7. Type `quit` to exit

## Configuration

Edit `voice_chat.py` to change:

| Setting | Default | Options |
|---------|---------|---------|
| `OLLAMA_MODEL` | `llama3.2:3b` | `qwen3:8b` (smarter, slower), any Ollama model |
| `WHISPER_MODEL` | `ggml-base.en.bin` | `ggml-small.en.bin` (better accuracy, ~2x slower) |
| `TTS_VOICE_FALLBACK` | `alba` | Any preset voice name |
| `MAX_RECORD_SECONDS` | `30` | Max recording length |
| `SYSTEM_PROMPT` | concise conversational | Edit to change LLM personality |

## Performance (M4 Mac mini, 16GB)

| Component | Metric |
|-----------|--------|
| whisper.cpp (base.en) | 20x realtime (5.5s audio → 0.28s) |
| llama3.2:3b (warm) | ~37 tok/s |
| Pocket-TTS | 6.1x realtime (5.5s audio → 0.9s), 200ms to first chunk |
| Model load (Pocket-TTS) | ~10s first run, cached after |
| Voice state load (safetensors) | <1s |
| Voice state export (from wav) | ~1s for 18s clip |

## Architecture

```
mic → sounddevice (16kHz) → whisper.cpp (Metal) → text
                                                       ↓
    speaker ← Pocket-TTS (CPU, 24kHz) ← Ollama (llama3.2:3b)
```

All processing is local. No network calls except Ollama on localhost:11434.

## License

MIT
