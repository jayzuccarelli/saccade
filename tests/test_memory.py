"""The three stores, each doing its one job."""

import json

from saccade.memory import EpisodicMemory, SemanticMemory, SensoryMemory, WorkingMemory
from saccade.schema import Frame, Percept


def test_sensory_is_a_bounded_frame_ring():
    s = SensoryMemory(maxlen=3)
    for i in range(5):
        s.observe(Frame(ts=float(i)))
    assert [int(f.ts) for f in s.recent(3)] == [2, 3, 4]  # last 3, oldest dropped
    assert len(s.recent(10)) == 3  # never more than it holds


def test_working_is_a_bounded_ring():
    w = WorkingMemory(maxlen=2)
    for s in ("a", "b", "c"):
        w.observe(Percept(ts=0.0, summary=s))
    assert [p.summary for p in w.recent()] == ["b", "c"]  # oldest dropped
    assert "b" in w.summary() and "c" in w.summary()


def test_episodic_appends_jsonl(tmp_path):
    path = tmp_path / "ep.jsonl"
    e = EpisodicMemory(str(path))
    e.record("action", {"message": "m"})
    e.record("action", {"message": "n"})
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["message"] == "m"


def test_semantic_reads_file_or_default(tmp_path):
    path = tmp_path / "prefs.md"
    path.write_text("be quiet during focus")
    assert SemanticMemory(str(path)).text() == "be quiet during focus"
    assert "no preferences" in SemanticMemory(str(tmp_path / "missing.md")).text()
