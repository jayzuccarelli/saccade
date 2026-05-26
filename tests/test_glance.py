"""Glance (peripheral, low-acuity) downscales its own input. Focus does not —
it always gets full resolution, which is why downscaling lives here, not in the
sensor."""

import asyncio
import io

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from saccade.glance import Glance  # noqa: E402
from saccade.memory import Memory  # noqa: E402
from saccade.schema import Frame, Window  # noqa: E402


class _CapturingBackend:
    def __init__(self):
        self.frames = None

    async def complete(self, prompt, frames, schema=None):
        self.frames = frames
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
    asyncio.run(g.perceive(Window(frames=[Frame(ts=0.0, image=big, mime="image/jpeg")]), _mem(tmp_path)))
    sent = be.frames[0].image
    assert max(Image.open(io.BytesIO(sent)).size) == 768
    assert len(sent) < len(big)


def test_glance_sends_original_when_off(tmp_path):
    big = _jpeg(1920, 1080)
    be = _CapturingBackend()
    g = Glance(be, max_dim=0)
    asyncio.run(g.perceive(Window(frames=[Frame(ts=0.0, image=big, mime="image/jpeg")]), _mem(tmp_path)))
    assert be.frames[0].image is big  # untouched
