"""The trace is the evidence locker: what the model was shown, on disk, per tick.

It exists because "cover the lens and it still describes a man" has three
unrelated causes that look identical from the terminal. The files settle it.
"""

import json
from pathlib import Path

from saccade.schema import Frame
from saccade.trace import Trace


def _tick(t: Trace, n: int = 1) -> None:
    for _ in range(n):
        t.record(
            "glance",
            [Frame(ts=1.0, image=b"jpegbytes", text="hello there")],
            '{"summary": "a hand over the lens"}',
        )


def test_a_tick_leaves_the_image_and_the_reply(tmp_path: Path):
    t = Trace(tmp_path)
    _tick(t)
    jpg = t.root / "000001_glance_0.jpg"
    assert jpg.read_bytes() == b"jpegbytes"
    meta = json.loads((t.root / "000001_glance.json").read_text())
    assert meta["heard"] == ["hello there"]
    assert "hand over the lens" in meta["reply"]


def test_an_audio_only_tick_still_records(tmp_path: Path):
    """No image is itself evidence: it says the camera contributed nothing."""
    t = Trace(tmp_path)
    t.record("glance", [Frame(ts=1.0, audio=b"wav", text="just sound")], "{}")
    meta = json.loads((t.root / "000001_glance.json").read_text())
    assert meta["images"] == 0
    assert meta["audio_bytes"] == 3


def test_old_ticks_are_pruned(tmp_path: Path):
    """A 768px jpeg a second is ~300MB an hour; the trace is a window, not an
    archive."""
    t = Trace(tmp_path, keep=5)
    _tick(t, 12)
    stems = {p.name.split("_", 1)[0] for p in t.root.iterdir()}
    assert stems == {f"{i:06d}" for i in range(8, 13)}


def test_two_runs_do_not_share_a_directory(tmp_path: Path, monkeypatch):
    """Numbering restarts per process, so a shared dir would interleave runs and
    the pruner would eat the wrong files."""
    monkeypatch.setattr("saccade.trace.time.time", lambda: 111)
    a = Trace(tmp_path)
    monkeypatch.setattr("saccade.trace.time.time", lambda: 222)
    b = Trace(tmp_path)
    assert a.root != b.root
