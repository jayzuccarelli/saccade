"""Speakers are swappable outputs. The loop must drive an async Speaker.say the
same way it drove a sync print, and the TTS speaker must write a real wav."""

import asyncio
import sys
import time
import urllib.error
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
from saccade.speakers import _playback
from saccade.speakers.gemini_tts import GeminiTTSSpeaker
from saccade.speakers.home_assistant import HomeAssistantSpeaker, _lan_ip
from saccade.speakers.piper import PiperError, PiperSpeaker
from saccade.speakers.print import PrintSpeaker


def _mem(tmp_path):
    return Memory(str(tmp_path / "ep.jsonl"), str(tmp_path / "prefs.md"))


def test_print_speaker_say_is_awaitable():
    asyncio.run(PrintSpeaker().say("hello"))  # no error, returns None


def test_loop_awaits_an_async_speaker(tmp_path):
    """A salient frame should drive Speaker.say through to completion, proving
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
        "media_player.living_room",
        serve_host="127.0.0.1",
        serve_port=0,  # ephemeral: no port collision
    )

    posted = {}
    spk._play_media = lambda url: posted.setdefault("url", url)  # no real HA call

    asyncio.run(spk.say("dinner is ready"))

    # play_media got a URL pointing at our own server + the synthesized clip
    assert posted["url"].startswith("http://127.0.0.1:") and posted["url"].endswith("/clip.wav")
    # and that URL is actually serveable: the file is reachable over HTTP
    body = urllib.request.urlopen(posted["url"], timeout=5).read()
    assert body.startswith(b"RIFF") and body.endswith(b"\x00\x01" * 100)  # wav with our pcm
    spk._server.shutdown()


def test_ha_speaker_does_not_expose_a_directory_index(tmp_path):
    """The clip dir accumulates every line saccade has ever spoken, so fetching one
    clip by name must work while browsing the directory must not."""
    tts = _FakeTTS(tmp_path / "utt")
    spk = HomeAssistantSpeaker(
        tts,
        "http://ha.local:8123",
        "tok",
        "media_player.living_room",
        serve_host="127.0.0.1",
        serve_port=0,
    )
    posted = {}
    spk._play_media = lambda url: posted.setdefault("url", url)
    asyncio.run(spk.say("dinner is ready"))

    root = posted["url"].rsplit("/", 1)[0] + "/"
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(root, timeout=5)
    assert excinfo.value.code == 404
    # the named clip is still reachable: we closed the index, not the door
    assert urllib.request.urlopen(posted["url"], timeout=5).read().startswith(b"RIFF")
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
    monkeypatch.setattr(
        _playback, "_to_device", lambda path, idx: bool(played.update(path=path, idx=idx)) or True
    )
    asyncio.run(spk.say("hi"))
    assert played["path"].name == "clip.wav" and played["idx"] == 3  # device playback fired


def test_say_falls_back_to_play_cmd_without_index(tmp_path, monkeypatch):
    """out_index -1 (default) leaves the device path alone: play_cmd/OS default."""
    spk = _stub_tts(tmp_path, play_cmd="", out_index=-1)
    hit = {"dev": False}
    monkeypatch.setattr(_playback, "_to_device", lambda path, idx: hit.__setitem__("dev", True))
    asyncio.run(spk.say("hi"))  # play_cmd empty -> just synthesize + print
    assert hit["dev"] is False  # the device path must not fire


def _fake_piper(monkeypatch, returncode=0, stderr=b""):
    """Stand in for the piper subprocess, capturing the argv it was called with."""
    seen: dict[str, list[str]] = {}

    class FakeProc:
        async def communicate(self):
            return b"", stderr

    async def fake_exec(*cmd, **kw):
        seen["cmd"] = list(cmd)
        proc = FakeProc()
        proc.returncode = returncode
        return proc

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    return seen


def test_piper_runs_as_a_subprocess_not_an_import(tmp_path, monkeypatch):
    """saccade is MIT and piper-tts is GPL, so it gets run, never linked. It also
    has to be *this* interpreter: the venv running saccade is the one with piper
    in it, and on Windows there may be no console script on PATH."""
    seen = _fake_piper(monkeypatch)
    path = asyncio.run(PiperSpeaker("en_US-lessac-medium", str(tmp_path)).synthesize("hello"))
    assert seen["cmd"][:3] == [sys.executable, "-m", "piper"]
    assert "en_US-lessac-medium" in seen["cmd"]
    assert path.suffix == ".wav"


def test_piper_guards_text_that_starts_with_a_dash(tmp_path, monkeypatch):
    """Regression: without the `--` delimiter an utterance like '-5 degrees
    outside' is parsed as a flag and piper exits instead of speaking."""
    seen = _fake_piper(monkeypatch)
    asyncio.run(PiperSpeaker("v", str(tmp_path)).synthesize("-5 degrees outside"))
    assert seen["cmd"][-2:] == ["--", "-5 degrees outside"]


def test_piper_not_installed_names_the_install_command(tmp_path, monkeypatch):
    _fake_piper(monkeypatch, returncode=1, stderr=b"No module named piper\n")
    with pytest.raises(PiperError, match="pip install piper-tts"):
        asyncio.run(PiperSpeaker("v", str(tmp_path)).synthesize("hi"))


def test_piper_missing_voice_names_the_download_command(tmp_path, monkeypatch):
    """The other failure people actually hit: piper installed, no voice pulled."""
    _fake_piper(monkeypatch, returncode=1, stderr=b"cannot find voice en_US-lessac-medium\n")
    with pytest.raises(PiperError, match="download_voices en_US-lessac-medium"):
        asyncio.run(PiperSpeaker("en_US-lessac-medium", str(tmp_path)).synthesize("hi"))


def test_piper_kills_a_hung_process_instead_of_waiting_forever(tmp_path, monkeypatch):
    """The loop awaits the speaker, so an unbounded wait here stops the agent
    watching the room (silently, and until someone restarts it). A truncated .onnx
    hangs rather than erroring, which is how you reach this in the wild."""
    from saccade.speakers import piper as piper_mod

    real_exec = asyncio.create_subprocess_exec

    async def hang(*args, **kw):
        return await real_exec(sys.executable, "-c", "__import__('time').sleep(30)", **kw)

    monkeypatch.setattr(piper_mod, "SYNTH_TIMEOUT_S", 0.5)
    monkeypatch.setattr("asyncio.create_subprocess_exec", hang)
    started = time.monotonic()
    with pytest.raises(PiperError, match="didn't finish"):
        asyncio.run(PiperSpeaker("v", str(tmp_path)).synthesize("hi"))
    assert time.monotonic() - started < 10  # it gave up, it didn't ride out the sleep


def test_playback_kills_a_hung_player(tmp_path, monkeypatch, capsys):
    """Same stall, one layer out: a wedged `afplay` holds the loop open forever.
    Losing an utterance is the acceptable outcome; losing the agent isn't."""
    monkeypatch.setattr(_playback, "PLAY_TIMEOUT_S", 0.5)
    clip = tmp_path / "utt.wav"
    clip.write_bytes(b"RIFF")
    started = time.monotonic()
    asyncio.run(_playback.play(clip, f"{sys.executable} -c __import__('time').sleep(30)", -1))
    assert time.monotonic() - started < 10
    assert "hung" in capsys.readouterr().out


def test_ha_speaker_synthesizes_with_piper_by_default(tmp_path):
    """Playing on a media_player shouldn't be the one output still forcing an API
    key. The HA speaker wraps whatever synthesizes; by default that's local Piper."""
    from saccade.__main__ import make_speaker
    from saccade.config import Config

    spk = make_speaker(Config(speaker="home_assistant", tts_dir=str(tmp_path / "utt")))
    assert isinstance(spk, HomeAssistantSpeaker)
    assert isinstance(spk.tts, PiperSpeaker)
    # and it must not also play locally: HA is the thing making the sound
    assert spk.tts.play_cmd == "" and spk.tts.out_index == -1


def test_ha_speaker_still_takes_gemini_when_asked(tmp_path):
    pytest.importorskip("google.genai")  # constructing GeminiTTSSpeaker needs the SDK
    from saccade.__main__ import make_speaker
    from saccade.config import Config

    spk = make_speaker(
        Config(speaker="home_assistant", ha_tts="gemini_tts", tts_dir=str(tmp_path / "utt"))
    )
    assert isinstance(spk.tts, GeminiTTSSpeaker)


def test_lan_ip_is_an_address():
    ip = _lan_ip()
    assert ip.count(".") == 3 and all(p.isdigit() for p in ip.split("."))


class _FakeSd:
    """Just enough sounddevice to exercise the channel matching."""

    def __init__(self, max_out):
        self.max_out = max_out
        self.played = None

    def query_devices(self, idx):
        return {"max_output_channels": self.max_out}

    def play(self, data, samplerate, device):
        self.played = data

    def wait(self):
        pass


def _play_wav(monkeypatch, tmp_path, max_out, channels=1, frames=None):
    # numpy lives in the `audio` extra, and the base harness is tested without it.
    np = pytest.importorskip("numpy")

    clip = tmp_path / "clip.wav"
    pcm = np.zeros(100 * channels, dtype=np.int16) if frames is None else frames
    with wave.open(str(clip), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(pcm.tobytes())
    fake = _FakeSd(max_out)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    ok = _playback._to_device(clip, 0)
    return ok, fake


def test_a_mono_clip_is_widened_to_what_the_device_takes(tmp_path, monkeypatch):
    """Piper writes mono and plenty of CoreAudio outputs will only open a stereo
    stream, so every utterance died with `Invalid number of channels [-9998]`:
    the agent watched the room all day and never made a sound."""
    ok, fake = _play_wav(monkeypatch, tmp_path, max_out=2)
    assert ok
    assert fake.played.shape[1] == 2


def test_a_mono_device_gets_mono(tmp_path, monkeypatch):
    ok, fake = _play_wav(monkeypatch, tmp_path, max_out=1)
    assert ok
    assert fake.played.ndim == 1 or fake.played.shape[1] == 1


def test_an_index_that_cannot_output_falls_back(tmp_path, monkeypatch, capsys):
    """A stale SACCADE_AUDIO_OUT_INDEX pointing at a mic shouldn't cost you every
    spoken line; the OS default is right there."""
    ok, _ = _play_wav(monkeypatch, tmp_path, max_out=0)
    assert not ok
    assert "no output" in capsys.readouterr().out


def test_a_dead_device_index_still_reaches_play_cmd(tmp_path, monkeypatch):
    """The fallback has to actually be taken, or the warning is just a warning."""
    ran = []
    monkeypatch.setattr(_playback, "_to_device", lambda path, idx: False)

    async def fake_exec(*cmd, **kw):
        ran.append(cmd)

        class P:
            async def wait(self):
                return 0

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    asyncio.run(_playback.play(tmp_path / "x.wav", "afplay", 7))
    assert ran and ran[0][0] == "afplay"


def test_every_channel_pair_lands_on_what_the_device_wants(tmp_path, monkeypatch):
    """Scaling by `wants // channels` quietly didn't: 2 into 3 stayed 2, and
    PortAudio refuses that exactly like it refused the mono stream."""
    for channels, wants in ((1, 2), (2, 3), (6, 2), (2, 1), (1, 1)):
        ok, fake = _play_wav(monkeypatch, tmp_path, max_out=wants, channels=channels)
        assert ok
        got = fake.played.shape[1] if fake.played.ndim > 1 else 1  # 1-D is mono
        assert got == wants, f"{channels} -> {wants} gave {got}"


def test_a_loud_downmix_does_not_wrap_around(tmp_path, monkeypatch):
    """Averaging int16 in int16 overflows: two channels near full scale summed to
    a negative number, so the loudest moment came out as a click."""
    np = pytest.importorskip("numpy")
    loud = np.full(200, 30000, dtype=np.int16)  # 100 frames, 2 channels, both hot
    ok, fake = _play_wav(monkeypatch, tmp_path, max_out=1, channels=2, frames=loud)
    assert ok
    assert fake.played.min() > 0, "a positive signal came out negative"
