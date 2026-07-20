"""Focus is given what it recently said, so it can self-regulate (not repeat)."""

import asyncio

from saccade.focus import Focus
from saccade.memory import Memory
from saccade.schema import Percept, Window


class _CapturingBackend:
    def __init__(self):
        self.prompt = None

    async def complete(self, prompt, frames, schema=None):
        self.prompt = prompt
        return '{"reasoning":"r","speak":false,"message":""}'


def test_focus_prompt_includes_recent_utterances(tmp_path):
    memory = Memory(str(tmp_path / "ep.jsonl"), str(tmp_path / "p.md"))
    memory.episodic.record("action", {"message": "want a hand with that?"})
    be = _CapturingBackend()
    asyncio.run(Focus(be).reason(Percept(ts=0.0, summary="x"), Window(frames=[]), memory))
    assert "want a hand with that?" in be.prompt  # Focus can see what it just said


def test_focus_prompt_handles_no_history(tmp_path):
    memory = Memory(str(tmp_path / "ep.jsonl"), str(tmp_path / "p.md"))
    be = _CapturingBackend()
    asyncio.run(Focus(be).reason(Percept(ts=0.0, summary="x"), Window(frames=[]), memory))
    assert "haven't said anything recently" in be.prompt


def test_focus_drops_utterances_older_than_the_window(tmp_path):
    """A line said in a prior session (older than the window) must not mute a
    fresh run: episodic persists on disk, so without a time bound 'recent'
    leaks across runs. That was the 'I walked in and heard nothing' bug."""
    memory = Memory(str(tmp_path / "ep.jsonl"), str(tmp_path / "p.md"))
    memory.episodic.record("action", {"message": "there is someone behind the plants"})
    memory.episodic.tail[-1]["ts"] -= 10_000  # said hours ago, well past the window
    be = _CapturingBackend()
    asyncio.run(
        Focus(be, recent_said_window_s=180).reason(
            Percept(ts=0.0, summary="x"), Window(frames=[]), memory
        )
    )
    assert "behind the plants" not in be.prompt  # stale line is filtered out
    assert "haven't said anything recently" in be.prompt
