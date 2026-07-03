"""MicSensor is the audio twin of WebcamSensor. The capture itself needs
hardware, but WAV framing is pure and worth pinning: a malformed header means
the model gets an unreadable clip."""

import io
import wave

from saccade.sensors.mic import MicSensor


def test_wav_framing_is_well_formed():
    sensor = MicSensor(sample_rate=16000)
    pcm = b"\x01\x00" * 16000  # 1s of int16 mono silence-ish
    data = sensor._wav(pcm)
    with wave.open(io.BytesIO(data), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2  # int16
        assert w.getframerate() == 16000
        assert w.getnframes() == 16000  # 2 bytes/sample -> 16000 frames
        assert w.readframes(16000) == pcm  # payload round-trips intact


def test_default_index_is_none():
    # -1 in config means "system default"; the sensor takes None for that.
    assert MicSensor().index is None
    assert MicSensor(index=2).index == 2
