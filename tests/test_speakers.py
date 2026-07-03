"""Speakers are swappable outputs. The loop must drive an async Speaker.say the
same way it drove a sync print, and the TTS speaker must write a real wav."""

import asyncio
import urllib.request
import wave
from types import SimpleNamespace

import pytest

from saccade import loop as looplib
from saccade.backends.stub import StubBackend
from saccade.focus import Focus
from saccade.glance import Glance
from saccade.memory import Memory
from saccade.schema import Frame
from saccade.speakers.gemini_tts import GeminiTTSSpeaker
from saccade.speakers.home_assistant import HomeAssistantSpeaker, _lan_ip
from saccade.speakers.print import PrintSpeaker


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
            Glance(StubBackend("glance")),
            Focus(StubBackend("focus")),
            memory,
            6,
            RecordingSpeaker().say,
        )
    )
    assert len(said) == 1 and said[0]  # the async say() was awaited to completion


def test_gemini_tts_writes_a_wav(tmp_path):
    pytest.importorskip("google.genai")  # optional dep: synthesize() needs SDK types
    pcm = b"\x00\x01" * 12000  # 24000 bytes of fake 16-bit mono PCM
    fake_resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(inline_data=SimpleNamespace(data=pcm))]
                )
            )
        ]
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


class _FakeTTS:
    """Stands in for GeminiTTSSpeaker: writes a wav, returns its path."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    async def synthesize(self, text):
        path = self.out_dir / "clip.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(b"\x00\x01" * 100)
        return path


def test_ha_speaker_serves_clip_and_posts_play_media(tmp_path):
    tts = _FakeTTS(tmp_path / "utt")
    spk = HomeAssistantSpeaker(
        tts,
        "http://ha.local:8123",
        "tok",
        "media_player.den",
        serve_host="127.0.0.1",
        serve_port=0,  # ephemeral — no port collision
    )

    posted = {}
    spk._play_media = lambda url: posted.setdefault("url", url)  # no real HA call

    asyncio.run(spk.say("dinner is ready"))

    # play_media got a URL pointing at our own server + the synthesized clip
    assert posted["url"].startswith("http://127.0.0.1:") and posted["url"].endswith("/clip.wav")
    # and that URL is actually serveable — the file is reachable over HTTP
    body = urllib.request.urlopen(posted["url"], timeout=5).read()
    assert body.startswith(b"RIFF") and body.endswith(b"\x00\x01" * 100)  # wav with our pcm
    spk._server.shutdown()


def _stub_tts(tmp_path, **kw):
    """A GeminiTTSSpeaker whose synthesize() writes a wav without touching the SDK."""
    spk = GeminiTTSSpeaker("m", "Kore", str(tmp_path / "utt"), **kw)

    async def fake_synthesize(text):
        path = spk.out_dir / "clip.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(b"\x00\x01" * 100)
        return path

    spk.synthesize = fake_synthesize
    return spk


def test_say_routes_to_device_when_index_set(tmp_path, monkeypatch):
    """out_index >= 0 plays to that specific device, not the OS-default play_cmd."""
    spk = _stub_tts(tmp_path, play_cmd="afplay", out_index=3)
    played = {}
    monkeypatch.setattr(spk, "_play_to_device", lambda path: played.setdefault("path", path))
    asyncio.run(spk.say("hi"))
    assert played["path"].name == "clip.wav"  # device playback fired


def test_say_falls_back_to_play_cmd_without_index(tmp_path, monkeypatch):
    """out_index -1 (default) leaves the device path alone — play_cmd/OS default."""
    spk = _stub_tts(tmp_path, play_cmd="", out_index=-1)
    hit = {"dev": False}
    monkeypatch.setattr(spk, "_play_to_device", lambda path: hit.__setitem__("dev", True))
    asyncio.run(spk.say("hi"))  # play_cmd empty -> just synthesize + print
    assert hit["dev"] is False  # the device path must not fire


def test_lan_ip_is_an_address():
    ip = _lan_ip()
    assert ip.count(".") == 3 and all(p.isdigit() for p in ip.split("."))
