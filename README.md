# 🦊 Local Voice Chat

Fully offline voice chat pipeline running on Apple Silicon (M4 Mac mini).

- **STT:** whisper.cpp (Metal-accelerated) — ~0.3s for short utterances
- **LLM:** Ollama (llama3.2:3b) — ~37 tok/s warm
- **TTS:** [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts) — 6x realtime, 200ms to first audio

Total round-trip latency: ~700ms–1s for a short exchange. No API keys, no cloud, fully private.

## Setup

```bash
# Dependencies
brew install whisper-cpp
ollama pull llama3.2:3b
ollama serve  # if not already running

# Install Python deps
cd ~/Dev/voice-chat
uv sync

# Run
uv run voice_chat.py
```

## Usage

1. Press **Enter** to start recording
2. Press **Enter** again to stop
3. Listen to the response

## Voice Cloning

Pass a clean `.wav` file as `TTS_VOICE` in the script, or use `--voice` with the CLI:

```python
TTS_VOICE = "/path/to/your_voice.wav"
```

```bash
uvx pocket-tts generate --voice /path/to/voice.wav --text "Hello world"
```

Export a voice state for fast reloading:

```python
from pocket_tts import TTSModel, export_model_state
model = TTSModel.load_model()
state = model.get_state_for_audio_prompt("./your_voice.wav")
export_model_state(state, "./your_voice.safetensors")
```

## Config

Edit `voice_chat.py` to change:
- `OLLAMA_MODEL` — swap to `qwen3:8b` for smarter (slower) responses
- `TTS_VOICE` — any built-in voice name or path to a `.wav`
- `WHISPER_MODEL` — swap to `small.en` for better accuracy
- `TTS_QUANTIZE` — int8 quantization for lower memory
