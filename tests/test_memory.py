"""The three stores, each doing its one job."""

import json

from saccade.memory import EpisodicMemory, SemanticMemory, WorkingMemory
from saccade.schema import Percept


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
