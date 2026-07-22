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


def test_two_runs_do_not_share_a_directory_even_in_the_same_second(tmp_path: Path, monkeypatch):
    """Ctrl-C and rerun lands inside one second, and numbering restarts per
    process, so a shared dir means the new run overwrites the old run's evidence:
    the one failure a trace must not have. Review's catch."""
    monkeypatch.setattr("saccade.trace.time.time", lambda: 111)
    a = Trace(tmp_path)
    _tick(a)
    b = Trace(tmp_path)
    _tick(b)
    assert a.root != b.root
    assert (a.root / "000001_glance.json").exists()
    assert (b.root / "000001_glance.json").exists()


def test_old_runs_are_swept_when_a_new_one_starts(tmp_path: Path, monkeypatch):
    """The per-run cap bounds one run; this bounds the habit. A debugging
    afternoon is a dozen Ctrl-C-and-reruns, each leaving up to `keep` ticks
    behind, and nobody comes back to sweep them."""
    times = iter([100, 200, 300, 400])
    monkeypatch.setattr("saccade.trace.time.time", lambda: next(times))
    a = Trace(tmp_path)
    _tick(a)
    b = Trace(tmp_path)
    c = Trace(tmp_path)
    d = Trace(tmp_path)
    assert not a.root.exists()
    assert b.root.exists() and c.root.exists() and d.root.exists()


def test_the_sweep_only_eats_what_the_trace_wrote(tmp_path: Path, monkeypatch):
    """rmtree on a name pattern had better be a strict pattern: a user's own
    files don't become deletable by living in the trace dir."""
    (tmp_path / "run-keepsake").mkdir()
    (tmp_path / "notes.txt").write_text("mine")
    times = iter([100, 200, 300, 400])
    monkeypatch.setattr("saccade.trace.time.time", lambda: next(times))
    for _ in range(4):
        Trace(tmp_path)
    assert (tmp_path / "run-keepsake").exists()
    assert (tmp_path / "notes.txt").read_text() == "mine"


def test_a_burst_of_same_second_starts_never_sweeps_the_new_run(tmp_path: Path, monkeypatch):
    """Review's catch, reproduced before fixing: same-second dirs shared a sort
    key, so the sweep picked its victim by directory-listing order, and on one
    filesystem that was the run about to write. The run just created is exempt:
    being about to write is what makes it newest, and no name tie-break says that."""
    monkeypatch.setattr("saccade.trace.time.time", lambda: 111)
    traces = [Trace(tmp_path) for _ in range(6)]
    assert traces[-1].root.exists()
    _tick(traces[-1])
    assert (traces[-1].root / "000001_glance.json").exists()
    assert sum(t.root.exists() for t in traces) == 3
