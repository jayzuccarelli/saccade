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
