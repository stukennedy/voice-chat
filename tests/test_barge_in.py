import threading
import time

import numpy as np
import pytest

import duplex_audio
import turn_taking
import voice_chat
from duplex_audio import BLOCK, DEVICE_RATE, DuplexAudio

FRAME = turn_taking.VAD_FRAME


def speechlike(seconds: float, seed: int) -> np.ndarray:
    """Broadband noise with a ~4Hz syllable envelope — enough for the echo canceller to lock on to."""
    rng = np.random.default_rng(seed)
    n = int(seconds * DEVICE_RATE)
    t = np.arange(n) / DEVICE_RATE
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t + rng.uniform(0, np.pi)) ** 2
    return (rng.standard_normal(n) * envelope * 0.05).astype(np.float32)


def run_through(audio: DuplexAudio, mic: np.ndarray, speaker: np.ndarray) -> np.ndarray:
    blocks = len(mic) // BLOCK
    for i in range(blocks):
        audio._raw.put((mic[i * BLOCK:(i + 1) * BLOCK], speaker[i * BLOCK:(i + 1) * BLOCK]))
    expected = blocks * BLOCK // duplex_audio.DECIMATE // FRAME
    return np.concatenate([audio.frames.get(timeout=10) for _ in range(expected)])


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x ** 2).mean()))


def test_echo_is_removed_and_the_user_is_not():
    bot = speechlike(6, seed=1)
    echo = np.concatenate([np.zeros(int(0.03 * DEVICE_RATE), np.float32), bot])[:len(bot)] * 0.5

    cleaned = run_through(DuplexAudio(FRAME), echo, bot)
    raw = run_through(DuplexAudio(FRAME, echo_cancellation=False), echo, bot)
    settled = slice(len(cleaned) // 3, None)  # after the canceller has adapted
    assert rms(cleaned[settled]) < rms(raw[settled]) / 10  # > 20dB of echo removed

    user = speechlike(3, seed=2)
    alone = run_through(DuplexAudio(FRAME), user, np.zeros_like(user))
    assert rms(alone[len(alone) // 3:]) > 0.5 * rms(user)


def test_capture_is_resampled_to_16k_frames():
    tone = np.sin(2 * np.pi * 440 * np.arange(DEVICE_RATE) / DEVICE_RATE).astype(np.float32) * 0.1
    out = run_through(DuplexAudio(FRAME, echo_cancellation=False), tone, np.zeros_like(tone))
    assert len(out) == DEVICE_RATE // BLOCK * BLOCK // 3 // FRAME * FRAME
    spectrum = np.abs(np.fft.rfft(out[1600:]))
    peak_hz = np.argmax(spectrum) * turn_taking.SAMPLE_RATE / len(out[1600:])
    assert abs(peak_hz - 440) < 5


def test_playback_accounting_and_stop():
    audio = DuplexAudio(FRAME)
    audio.play(np.ones(24000, np.float32), 24000)  # 1s at 24k -> 48000 device samples
    assert audio.is_playing() and audio.queued_samples == DEVICE_RATE

    audio._pull(BLOCK * 10)
    assert audio.played_samples == BLOCK * 10
    assert not audio.echo_converged

    audio.stop()
    assert not audio.is_playing()
    assert audio.queued_samples == audio.played_samples
    assert not audio._pull(BLOCK).any()  # silence once stopped


class FakeSmartTurn:
    def probability(self, _audio):
        return 0.0


def test_barge_in_needs_sustained_speech():
    probs = iter([1.0] * 5 + [0.0] + [1.0] * turn_taking.BARGE_IN_START_FRAMES)
    started = []
    builder = turn_taking.TurnBuilder(lambda _f: next(probs), FakeSmartTurn(), verbose=False)
    builder.start_frames = turn_taking.BARGE_IN_START_FRAMES
    builder.on_speech_start = lambda: started.append(True)

    frame = np.zeros(FRAME, np.float32)
    for _ in range(6):  # a short burst then a gap: not enough to interrupt
        builder.feed(frame)
    assert not started
    for _ in range(turn_taking.BARGE_IN_START_FRAMES):
        builder.feed(frame)
    assert started and builder.speaking


class AlwaysDone:
    def probability(self, _audio):
        return 1.0


def test_an_interruption_is_not_split_by_a_dropout():
    builder = turn_taking.TurnBuilder(lambda f: float(f[0]), AlwaysDone(), verbose=False)
    builder.start_frames = turn_taking.BARGE_IN_START_FRAMES
    builder.min_turn_ms = turn_taking.BARGE_IN_MIN_TURN_MS

    def feed(prob, frames):
        return [builder.feed(np.full(FRAME, prob, np.float32)) for _ in range(frames)]

    feed(1.0, turn_taking.BARGE_IN_START_FRAMES)
    builder.min_turn_ms = 0  # the floor changing mid-turn must not lift the hold
    gap = round(400 / turn_taking.FRAME_MS)
    assert not any(r is not None for r in feed(0.0, gap))  # would end a normal turn
    assert builder.speaking
    feed(1.0, 10)
    assert any(r is not None for r in feed(0.0, gap))  # past the hold, silence ends it


def test_a_short_word_counts_its_onset_as_speech():
    builder = turn_taking.TurnBuilder(lambda f: float(f[0]), AlwaysDone(), verbose=False)
    frames = [1.0] * round(288 / turn_taking.FRAME_MS) + [0.0] * 20  # ~288ms "Bye."
    ended = [builder.feed(np.full(FRAME, p, np.float32)) for p in frames]
    assert any(r is not None for r in ended)


@pytest.fixture
def fake_tts(monkeypatch):
    class Model:
        sample_rate = 24000
    monkeypatch.setattr(voice_chat, "_tts_model", Model())
    monkeypatch.setattr(voice_chat, "generate_speech_chunk", lambda text: np.ones(12000, np.float32))
    audio = DuplexAudio(FRAME, echo_cancellation=False)
    monkeypatch.setattr(voice_chat, "_duplex", audio)
    return audio


def test_interrupting_stops_speech_but_lets_the_llm_finish(fake_tts):
    interrupted = threading.Event()
    consumed = []

    def sentences():
        for i in range(6):
            consumed.append(i)
            if i == 2:
                while fake_tts.queued_samples < DEVICE_RATE:
                    time.sleep(0.005)
                fake_tts._pull(DEVICE_RATE)  # the first two chunks (0.5s each) are heard
                interrupted.set()
            yield f"sentence {i}."
            time.sleep(0.01)

    heard, cut_short = voice_chat.speak_stream(sentences(), interrupted=interrupted)

    assert cut_short
    assert heard == "sentence 0. sentence 1."
    assert not fake_tts.is_playing()
    voice_chat.wait_for_previous_response()
    assert consumed == list(range(6))


def test_uninterrupted_reply_is_heard_in_full(fake_tts):
    done = threading.Event()

    def player():
        while not done.is_set():
            fake_tts._pull(BLOCK * 50)
            time.sleep(0.005)

    threading.Thread(target=player, daemon=True).start()
    heard, cut_short = voice_chat.speak_stream(iter(["one two three.", "four five six."]),
                                               interrupted=threading.Event())
    done.set()
    assert not cut_short
    assert heard == "one two three. four five six."


def test_interruption_note():
    assert "only heard you say: \"Hi there.\"" in voice_chat.interruption_note("Hi there.")
    assert "before you started" in voice_chat.interruption_note("")
