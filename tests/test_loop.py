"""The loop: escalation triggers Focus, and one bad tick never kills the agent."""

import asyncio

from saccade import loop as looplib
from saccade.focus import Focus
from saccade.glance import Glance
from saccade.memory import Memory
from saccade.backends.stub import StubBackend
from saccade.schema import Frame


class _ScriptedSensor:
    def __init__(self, scenes):
        self.scenes = scenes

    async def stream(self):
        for s in self.scenes:
            yield Frame(ts=0.0, image=None, meta={"scene": s})


class _BoomBackend:
    async def complete(self, prompt, frames, schema=None):
        raise RuntimeError("boom")


def _run(sensor, glance, focus, memory):
    actions: list[str] = []
    asyncio.run(looplib.run(sensor, glance, focus, memory, on_action=actions.append))
    return actions


def test_salient_scene_triggers_an_action(tmp_path):
    sensor = _ScriptedSensor(["person typing", "the mug is empty"])
    glance = Glance(StubBackend("glance"))
    focus = Focus(StubBackend("focus"))
    memory = Memory(str(tmp_path / "ep.jsonl"), str(tmp_path / "prefs.md"))
    actions = _run(sensor, glance, focus, memory)
    assert len(actions) == 1  # only the salient scene escalated
    assert "empty" in actions[0]


class _CapturingFocusBackend:
    def __init__(self):
        self.n_frames = None

    async def complete(self, prompt, frames, schema=None):
        self.n_frames = len(frames)
        return '{"reasoning":"r","speak":false,"message":""}'


def test_focus_receives_a_clip_not_one_frame(tmp_path):
    # 4 scenes, the last is salient -> Focus fires with a clip of recent frames
    sensor = _ScriptedSensor(["a", "b", "c", "the mug is empty"])
    glance = Glance(StubBackend("glance"))
    cap = _CapturingFocusBackend()
    focus = Focus(cap)
    memory = Memory(str(tmp_path / "ep.jsonl"), str(tmp_path / "prefs.md"))
    asyncio.run(looplib.run(sensor, glance, focus, memory, on_action=[].append, focus_clip_frames=3))
    assert cap.n_frames == 3  # got the last 3 buffered frames, not just 1


def test_loop_survives_a_failing_backend(tmp_path):
    sensor = _ScriptedSensor(["the mug is empty"])
    glance = Glance(_BoomBackend())  # blows up every tick
    focus = Focus(StubBackend("focus"))
    memory = Memory(str(tmp_path / "ep.jsonl"), str(tmp_path / "prefs.md"))
    actions = _run(sensor, glance, focus, memory)  # must NOT raise
    assert actions == []
