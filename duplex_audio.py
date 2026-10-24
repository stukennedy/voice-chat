# Duplex Audio — speaker and mic open together, with echo cancellation
#
# Barge-in needs the mic open while the bot speaks, and an open mic next to a
# speaker hears the bot. WebRTC's echo canceller (via livekit) subtracts the
# speaker signal from the mic signal before VAD and STT see it.
#
# One CoreAudio stream carries both directions, so each callback hands over the
# mic block and the speaker block for the same 10ms of wall-clock time. That
# keeps the echo canceller's reference aligned with what the mic picked up.

import collections
import queue
import threading

import numpy as np
import sounddevice as sd
from livekit import rtc
from scipy.signal import firwin, lfilter, resample_poly

DEVICE_RATE = 48000
BLOCK = DEVICE_RATE // 100  # the echo canceller only accepts 10ms frames
CAPTURE_RATE = 16000        # what VAD, Smart Turn and whisper want
DECIMATE = DEVICE_RATE // CAPTURE_RATE


class _Downsampler:
    """48kHz -> 16kHz, carrying filter state across blocks so block edges don't click."""

    def __init__(self):
        self._taps = firwin(63, 7000, fs=DEVICE_RATE)
        self._state = np.zeros(len(self._taps) - 1)

    def __call__(self, block: np.ndarray) -> np.ndarray:
        filtered, self._state = lfilter(self._taps, 1.0, block, zi=self._state)
        return filtered[::DECIMATE].astype(np.float32)


def _to_frame(block: np.ndarray) -> rtc.AudioFrame:
    pcm = (np.clip(block, -1.0, 1.0) * 32767).astype(np.int16)
    return rtc.AudioFrame(pcm.tobytes(), DEVICE_RATE, 1, len(pcm))


def _from_frame(frame: rtc.AudioFrame) -> np.ndarray:
    return np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32767


class DuplexAudio:
    """
    play() queues speech; frames yields echo-cancelled 16kHz mic audio in
    frame_size chunks for as long as the stream is open.
    """

    def __init__(self, frame_size: int, echo_cancellation: bool = True):
        self._frame_size = frame_size
        self._apm = (
            rtc.AudioProcessingModule(echo_cancellation=True, high_pass_filter=True)
            if echo_cancellation else None
        )
        self._downsample = _Downsampler()
        self._pending = np.zeros(0, dtype=np.float32)

        self._lock = threading.Lock()
        self._playback: collections.deque[np.ndarray] = collections.deque()
        self._offset = 0         # read position within _playback[0]
        self.queued_samples = 0  # running totals at DEVICE_RATE, for "how much was heard"
        self.played_samples = 0

        self._raw: queue.Queue = queue.Queue()
        self.frames: queue.Queue = queue.Queue()

        self._stream = sd.Stream(
            samplerate=DEVICE_RATE,
            blocksize=BLOCK,
            channels=(1, 1),
            dtype="float32",
            callback=self._callback,
        )
        threading.Thread(target=self._process, daemon=True).start()

    def start(self):
        self._stream.start()
        if self._apm is not None:
            in_latency, out_latency = self._stream.latency
            self._apm.set_stream_delay_ms(int((in_latency + out_latency) * 1000))

    # ── Playback ────────────────────────────────────────────────────────

    def play(self, audio: np.ndarray, sample_rate: int):
        if sample_rate != DEVICE_RATE:
            g = np.gcd(sample_rate, DEVICE_RATE)
            audio = resample_poly(audio, DEVICE_RATE // g, sample_rate // g)
        audio = audio.astype(np.float32)
        with self._lock:
            self._playback.append(audio)
            self.queued_samples += len(audio)

    def stop(self):
        """Drop everything not yet handed to the device."""
        with self._lock:
            self._playback.clear()
            self._offset = 0
            self.queued_samples = self.played_samples

    def is_playing(self) -> bool:
        with self._lock:
            return bool(self._playback)

    @property
    def echo_converged(self) -> bool:
        """The canceller adapts to the room from the bot's own speech; ~1s is enough."""
        return self._apm is None or self.played_samples >= DEVICE_RATE

    @property
    def output_latency(self) -> float:
        return self._stream.latency[1]

    def _pull(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < n and self._playback:
                head = self._playback[0]
                take = min(n - filled, len(head) - self._offset)
                out[filled:filled + take] = head[self._offset:self._offset + take]
                filled += take
                self._offset += take
                if self._offset == len(head):
                    self._playback.popleft()
                    self._offset = 0
            self.played_samples += filled
        return out

    # ── Capture ─────────────────────────────────────────────────────────

    def _callback(self, indata, outdata, frames, _time, _status):
        out = self._pull(frames)
        outdata[:, 0] = out
        self._raw.put((indata[:, 0].copy(), out))

    def _process(self):
        # The echo canceller is not thread-safe, so reference and capture are
        # both fed from this one thread, in callback order.
        while True:
            mic, speaker = self._raw.get()
            if self._apm is not None and len(mic) == BLOCK:
                self._apm.process_reverse_stream(_to_frame(speaker))
                frame = _to_frame(mic)
                self._apm.process_stream(frame)
                mic = _from_frame(frame)

            self._pending = np.concatenate([self._pending, self._downsample(mic)])
            while len(self._pending) >= self._frame_size:
                self.frames.put(self._pending[:self._frame_size])
                self._pending = self._pending[self._frame_size:]
