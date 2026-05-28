"""Speakers are swappable outputs. The loop must drive an async Speaker.say the
same way it drove a sync print, and the TTS speaker must write a real wav."""

import asyncio
import wave
from types import SimpleNamespace

from saccade import loop as looplib
from saccade.focus import Focus
from saccade.glance import Glance
from saccade.memory import Memory
from saccade.backends.stub import StubBackend
from saccade.schema import Frame
from saccade.speakers.print import PrintSpeaker
from saccade.speakers.gemini_tts import GeminiTTSSpeaker


def _mem(tmp_path):
    return Memory(str(tmp_path / "ep.jsonl"), str(tmp_path / "prefs.md"))


def test_print_speaker_say_is_awaitable():
    asyncio.run(PrintSpeaker().say("hello"))  # no error, returns None


def test_loop_awaits_an_async_speaker(tmp_path):
    """A salient frame should drive Speaker.say through to completion — proving
    the loop awaits an async on_action, not just calls it and drops the coroutine."""
    memory = _mem(tmp_path)
    memory.observe_frame(Frame(ts=0.0, meta={"scene": "the mug is empty"}))

    said = []

    class RecordingSpeaker:
        async def say(self, text):
            said.append(text)

    asyncio.run(
        looplib._tick(
            Glance(StubBackend("glance")), Focus(StubBackend("focus")), memory, 6,
            RecordingSpeaker().say,
        )
    )
    assert len(said) == 1 and said[0]  # the async say() was awaited to completion


def test_gemini_tts_writes_a_wav(tmp_path):
    pcm = b"\x00\x01" * 12000  # 24000 bytes of fake 16-bit mono PCM
    fake_resp = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(
            parts=[SimpleNamespace(inline_data=SimpleNamespace(data=pcm))]
        ))]
    )

    class FakeModels:
        async def generate_content(self, **kwargs):
            self.kwargs = kwargs
            return fake_resp

    fake_models = FakeModels()
    speaker = GeminiTTSSpeaker("m", "Kore", str(tmp_path / "utt"))
    speaker._client = SimpleNamespace(aio=SimpleNamespace(models=fake_models))

    asyncio.run(speaker.say("there is someone behind the plants"))

    wavs = list((tmp_path / "utt").glob("*.wav"))
    assert len(wavs) == 1
    with wave.open(str(wavs[0]), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == len(pcm) // 2
    # the line we asked for actually reached the model
    assert fake_models.kwargs["contents"] == "there is someone behind the plants"
