"""Local transcription is what makes hearing vendor-independent. The contract:
the transcript reaches every backend, and when we transcribe here the audio does
not leave with it."""

import asyncio

from saccade.schema import Frame, heard_text
from saccade.sensors.mic import MicSensor


class FakeTranscriber:
    def __init__(self, text="someone knocked on the door"):
        self.text = text

    async def transcribe(self, wav):
        self.seen = wav
        return self.text


def _mic_with(transcriber, monkeypatch):
    """A MicSensor whose recording is faked: no PortAudio in CI."""
    monkeypatch.setattr("saccade.sensors.mic.require_audio", lambda: None)
    sensor = MicSensor(index=0, fps=1.0, transcriber=transcriber)
    monkeypatch.setattr(sensor, "_record", lambda: b"\x00\x00" * 16000)
    return sensor


async def _first(sensor):
    async for frame in sensor.stream():
        return frame
    raise AssertionError("stream yielded nothing")


def test_transcribing_locally_does_not_send_the_audio(monkeypatch):
    """The whole reason to transcribe here. Attaching the wav *as well* would
    upload the room anyway and give up the point of doing it locally."""
    frame = asyncio.run(_first(_mic_with(FakeTranscriber(), monkeypatch)))
    assert frame.text == "someone knocked on the door"
    assert frame.audio is None


def test_without_a_transcriber_the_audio_is_still_sent(monkeypatch):
    """Unchanged path: hand the clip to a backend that accepts audio."""
    frame = asyncio.run(_first(_mic_with(None, monkeypatch)))
    assert frame.audio and frame.text == ""


def test_silence_still_yields_a_frame(monkeypatch):
    """An ambient mic is mostly nothing. A silent clip must still tick, or the
    loop's cadence stalls and Glance never learns that nothing was said."""
    frame = asyncio.run(_first(_mic_with(FakeTranscriber(""), monkeypatch)))
    assert frame.text == "" and frame.audio is None


def test_transcript_is_rendered_for_every_backend():
    rendered = heard_text([Frame(ts=0.0, text="the kettle is boiling")])
    assert "the kettle is boiling" in rendered


def test_no_text_renders_nothing():
    """Backends append this unconditionally, so a vision-only run must get an
    empty string rather than a stray header."""
    assert heard_text([Frame(ts=0.0, image=b"jpeg")]) == ""
