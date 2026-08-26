# Turn Taking — hands-free end-of-turn detection
#
# Two models cooperate, as Smart Turn is designed to be used:
#   1. Silero VAD    — cheap per-frame speech/silence gate (32ms frames, ~3ms)
#   2. Smart Turn v3 — semantic endpointing, run *during* silence to decide
#                      whether that silence is a real turn boundary or just a
#                      mid-sentence pause (8MB int8 ONNX, ~13ms on CPU)
#
# Silero says "they stopped making noise". Smart Turn says "they stopped talking".
#
# Half-duplex: the mic stream is opened per turn and closed before the bot
# speaks, so TTS output is never fed back into the detector.

import queue
import time
from collections import deque

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import torch
from huggingface_hub import hf_hub_download
from silero_vad import load_silero_vad
from transformers import WhisperFeatureExtractor

# ── Config ──────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000

# Silero requires exactly 512-sample frames at 16kHz (32ms)
VAD_FRAME = 512
FRAME_MS = VAD_FRAME / SAMPLE_RATE * 1000

VAD_THRESHOLD = 0.5        # speech probability above which a frame is speech
SPEECH_START_FRAMES = 2    # consecutive speech frames needed to open a turn
PREROLL_SECONDS = 0.3      # audio retained from before speech onset

SILENCE_PROBE_MS = 200     # first Smart Turn check after this much silence
PROBE_INTERVAL_MS = 200    # re-check cadence while the silence continues
MAX_SILENCE_MS = 3000      # hard fallback: end the turn however Smart Turn votes

SMART_TURN_THRESHOLD = 0.5  # p(complete) at or above which the turn is over
SMART_TURN_WINDOW = 8 * SAMPLE_RATE  # model sees the last 8s of the turn

MAX_TURN_SECONDS = 30
SETTLE_FRAMES = 3          # frames discarded after opening the stream
KEEP_TRAILING_MS = 300     # trailing silence kept before handing audio to STT
MIN_SPEECH_MS = 250        # turns with less real speech than this are noise, not words

# ── Smart Turn v3 ───────────────────────────────────────────────────────

class SmartTurn:
    """Semantic endpointing: p(the speaker has finished their turn)."""

    REPO = "pipecat-ai/smart-turn-v3"
    WEIGHTS = "smart-turn-v3.2-cpu.onnx"

    def __init__(self):
        path = hf_hub_download(self.REPO, self.WEIGHTS)

        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(path, sess_options=opts)
        self._features = WhisperFeatureExtractor(chunk_length=8)

        # Warm up so the first real turn isn't the one that pays for it
        self.probability(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def probability(self, audio: np.ndarray) -> float:
        """
        audio: mono float32 at 16kHz, the current turn so far.
        Returns the model's sigmoid output — 1.0 = definitely complete.
        """
        # The model wants exactly 8s with the speech at the *end*, so long
        # input is cropped from the front and short input is front-padded.
        if len(audio) > SMART_TURN_WINDOW:
            audio = audio[-SMART_TURN_WINDOW:]
        elif len(audio) < SMART_TURN_WINDOW:
            pad = np.zeros(SMART_TURN_WINDOW - len(audio), dtype=np.float32)
            audio = np.concatenate([pad, audio])

        inputs = self._features(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=SMART_TURN_WINDOW,
            truncation=True,
            do_normalize=True,
        )
        features = np.expand_dims(inputs.input_features.squeeze(0).astype(np.float32), axis=0)
        return float(self._session.run(None, {"input_features": features})[0][0][0])

# ── Turn state machine ──────────────────────────────────────────────────

class TurnBuilder:
    """
    Frame-by-frame end-of-turn state machine. Does no audio I/O, so it can be
    driven from a mic stream or from a recorded file in tests.

    feed() returns None while the turn is still in progress, or the turn's
    audio once the turn is judged complete.
    """

    def __init__(self, speech_probability, smart_turn: "SmartTurn", verbose: bool = True):
        self._is_speech_prob = speech_probability
        self._smart_turn = smart_turn
        self.verbose = verbose
        self.reset()

    def reset(self):
        self._preroll = deque(maxlen=max(1, int(PREROLL_SECONDS * 1000 / FRAME_MS)))
        self._turn: list[np.ndarray] = []
        self._speech_run = 0
        self.speaking = False
        self.silence_ms = 0.0
        self._next_probe = SILENCE_PROBE_MS
        self._speech_ms = 0.0
        self._probes = 0

    def feed(self, frame: np.ndarray) -> np.ndarray | None:
        is_speech = self._is_speech_prob(frame) >= VAD_THRESHOLD

        if not self.speaking:
            # Hold a rolling pre-roll so the first word isn't clipped
            self._preroll.append(frame)
            self._speech_run = self._speech_run + 1 if is_speech else 0
            if self._speech_run >= SPEECH_START_FRAMES:
                self.speaking = True
                self._turn = list(self._preroll)
                self.silence_ms, self._next_probe = 0.0, SILENCE_PROBE_MS
                if self.verbose:
                    print("  🗣️  Speech detected", flush=True)
            return None

        self._turn.append(frame)

        if is_speech:
            self._speech_ms += FRAME_MS
            self.silence_ms, self._next_probe = 0.0, SILENCE_PROBE_MS
        else:
            self.silence_ms += FRAME_MS

            if self.silence_ms >= MAX_SILENCE_MS:
                finished = self._finish(reason=f"{self.silence_ms:.0f}ms silence timeout")
                if finished is not None:
                    return finished
                return None

            if self.silence_ms >= self._next_probe:
                audio = np.concatenate(self._turn)
                t0 = time.time()
                p = self._smart_turn.probability(audio)
                took_ms = (time.time() - t0) * 1000
                if p >= SMART_TURN_THRESHOLD:
                    finished = self._finish(
                        reason=f"p={p:.2f} after {self.silence_ms:.0f}ms silence, {took_ms:.0f}ms"
                    )
                    if finished is not None:
                        return finished
                    return None
                self._probes += 1
                if self.verbose:
                    # Rewrite one line rather than scrolling a probe per 200ms
                    print(
                        f"\r  ⏳ still talking (p={p:.2f}, "
                        f"{self.silence_ms / 1000:.1f}s silence)\033[K",
                        end="",
                        flush=True,
                    )
                self._next_probe = self.silence_ms + PROBE_INTERVAL_MS

        if len(self._turn) * VAD_FRAME >= MAX_TURN_SECONDS * SAMPLE_RATE:
            return self._finish(reason=f"max {MAX_TURN_SECONDS}s reached")

        return None

    def _finish(self, reason: str) -> np.ndarray | None:
        audio = np.concatenate(self._turn)

        if self.verbose and self._probes:
            print()  # close off the rewritten "still talking" line

        # A door slam or a cough can clear the VAD without being speech.
        # Drop it rather than sending whisper something to hallucinate over.
        if self._speech_ms < MIN_SPEECH_MS:
            print(f"  🔇 Ignoring {self._speech_ms:.0f}ms blip (not speech)", flush=True)
            self.reset()
            return None

        # Trim the dead air we spent deliberating — whisper is both slower and
        # more prone to hallucinating on long trailing silence.
        excess_ms = self.silence_ms - KEEP_TRAILING_MS
        if excess_ms > 0:
            audio = audio[: -int(excess_ms / 1000 * SAMPLE_RATE)]

        print(f"  ⏹️  Turn end ({len(audio) / SAMPLE_RATE:.1f}s, {reason})", flush=True)
        return audio

# ── Listener ────────────────────────────────────────────────────────────

class TurnListener:
    """
    Blocking hands-free capture. listen() returns the audio of one spoken turn
    once Smart Turn judges it complete, or None if nothing was said within
    wait_seconds.

    The mic stream is opened per call and closed on return, which is what makes
    this half-duplex: while the bot is speaking there is no open input stream,
    so TTS output can never be mistaken for the user taking a turn.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        print("👂 Loading Silero VAD...", flush=True)
        self._vad = load_silero_vad()
        print("👂 Loading Smart Turn v3...", flush=True)
        self._smart_turn = SmartTurn()
        self._builder = TurnBuilder(self._speech_probability, self._smart_turn, verbose)
        print("👂 Turn detection ready.", flush=True)

    def _speech_probability(self, frame: np.ndarray) -> float:
        with torch.no_grad():
            return self._vad(torch.from_numpy(frame), SAMPLE_RATE).item()

    def listen(self, wait_seconds: float | None = None) -> np.ndarray | None:
        self._vad.reset_states()
        self._builder.reset()

        frames: queue.Queue = queue.Queue()

        def callback(indata, _frames, _time, _status):
            frames.put(indata[:, 0].copy())

        settle = SETTLE_FRAMES
        started = time.time()
        print("🎙️  Listening...", flush=True)

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=VAD_FRAME,
            callback=callback,
        )
        with stream:
            while True:
                try:
                    frame = frames.get(timeout=0.5)
                except queue.Empty:
                    frame = None

                if frame is not None:
                    # Drop the stream's startup transient
                    if settle > 0:
                        settle -= 1
                        continue

                    audio = self._builder.feed(frame)
                    if audio is not None:
                        return audio

                if (
                    wait_seconds
                    and not self._builder.speaking
                    and time.time() - started > wait_seconds
                ):
                    print("  (nothing heard)", flush=True)
                    return None
