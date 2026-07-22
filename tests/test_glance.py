"""Glance (peripheral, low-acuity) downscales its own input. Focus does not;
it always gets full resolution, which is why downscaling lives here, not in the
sensor."""

import asyncio
import io

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from saccade.glance import Glance  # noqa: E402
from saccade.memory import Memory  # noqa: E402
from saccade.schema import Frame, Percept, Window  # noqa: E402


class _CapturingBackend:
    def __init__(self):
        self.frames = None
        self.prompt = None

    async def complete(self, prompt, frames, schema=None):
        self.frames = frames
        self.prompt = prompt
        return '{"summary":"x","tags":[],"salience":0.1,"escalate":false,"state_delta":""}'


def _jpeg(w, h) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (w, h), (100, 100, 100)).save(out, "JPEG")
    return out.getvalue()


def _mem(tmp_path):
    return Memory(str(tmp_path / "e.jsonl"), str(tmp_path / "p.md"))


def test_glance_downscales_before_sending(tmp_path):
    big = _jpeg(1920, 1080)
    be = _CapturingBackend()
    g = Glance(be, max_dim=768)
    asyncio.run(
        g.perceive(Window(frames=[Frame(ts=0.0, image=big, mime="image/jpeg")]), _mem(tmp_path))
    )
    sent = be.frames[0].image
    assert max(Image.open(io.BytesIO(sent)).size) == 768
    assert len(sent) < len(big)


def test_glance_sends_original_when_off(tmp_path):
    big = _jpeg(1920, 1080)
    be = _CapturingBackend()
    g = Glance(be, max_dim=0)
    asyncio.run(
        g.perceive(Window(frames=[Frame(ts=0.0, image=big, mime="image/jpeg")]), _mem(tmp_path))
    )
    assert be.frames[0].image is big  # untouched


def test_glance_passes_audio_frames_through(tmp_path):
    """Downscaling is image-only: an audio-only frame (mic sensor) must reach the
    backend with its clip intact, not get dropped by the shrink path."""
    be = _CapturingBackend()
    g = Glance(be, max_dim=768)
    clip = b"RIFFwavbytes"
    asyncio.run(
        g.perceive(
            Window(frames=[Frame(ts=0.0, audio=clip, audio_mime="audio/wav")]), _mem(tmp_path)
        )
    )
    assert be.frames[0].audio is clip  # untouched
    assert be.frames[0].image is None


def test_glance_marks_already_escalated_percepts_in_context(tmp_path):
    """An ongoing event Glance already escalated is marked [escalated] in its
    recent context, so it has an anchor not to re-flag the same thing each tick:
    the over-escalation seen on a motionless person on the couch."""
    memory = _mem(tmp_path)
    memory.observe(Percept(ts=0.0, summary="person on the couch", escalate=True))
    memory.observe(Percept(ts=1.0, summary="person on the couch", escalate=False))

    recent = Glance(_CapturingBackend())._recent(memory)
    assert "[escalated] person on the couch" in recent  # the one it flagged is marked
    assert recent.count("[escalated]") == 1  # the non-escalated one is left unmarked

    be = _CapturingBackend()  # and the marked context actually reaches the model
    asyncio.run(
        Glance(be).perceive(
            Window(frames=[Frame(ts=2.0, image=_jpeg(64, 64), mime="image/jpeg")]), memory
        )
    )
    assert "[escalated] person on the couch" in be.prompt


def test_the_trace_gets_what_was_sent_not_what_was_captured(tmp_path):
    """The question a trace answers is what the model saw. Recording the camera's
    original while the model got a downscaled copy would let the two disagree
    exactly when it matters."""
    import asyncio
    import json

    from saccade.trace import Trace

    sent = {}

    class _Backend:
        async def complete(self, prompt, frames, schema=None):
            sent["frames"] = frames
            return '{"summary": "s", "tags": [], "salience": 0.1, "escalate": false, "state_delta": "", "next_glance_s": 1.0}'

    trace = Trace(tmp_path)
    g = Glance(_Backend(), max_dim=0, trace=trace)
    mem = Memory(str(tmp_path / "e.jsonl"), str(tmp_path / "p.md"))
    asyncio.run(g.perceive(Window(frames=[Frame(ts=1.0, image=b"asis")]), mem))
    assert (trace.root / "000001_glance_0.jpg").read_bytes() == b"asis"
    meta = json.loads((trace.root / "000001_glance.json").read_text())
    assert meta["images"] == 1
