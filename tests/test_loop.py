"""Loop: a glance reasons over a clip; the agent survives bad ticks and stops
cleanly when a finite source ends. The per-tick logic is tested directly (no
timing); the parallel capture/shutdown is tested through run()."""

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


class _CapturingFocusBackend:
    def __init__(self):
        self.n_frames = None

    async def complete(self, prompt, frames, schema=None):
        self.n_frames = len(frames)
        return '{"reasoning":"r","speak":true,"message":"hi"}'


class _BoomBackend:
    async def complete(self, prompt, frames, schema=None):
        raise RuntimeError("boom")


def _mem(tmp_path):
    return Memory(str(tmp_path / "ep.jsonl"), str(tmp_path / "prefs.md"))


# --- per-tick logic: deterministic, no timing ---

def test_tick_salient_frame_triggers_action(tmp_path):
    memory = _mem(tmp_path)
    memory.observe_frame(Frame(ts=0.0, meta={"scene": "the mug is empty"}))
    actions = []
    asyncio.run(looplib._tick(Glance(StubBackend("glance")), Focus(StubBackend("focus")), memory, 6, actions.append))
    assert len(actions) == 1


def test_tick_quiet_frame_stays_silent(tmp_path):
    memory = _mem(tmp_path)
    memory.observe_frame(Frame(ts=0.0, meta={"scene": "person typing"}))
    actions = []
    asyncio.run(looplib._tick(Glance(StubBackend("glance")), Focus(StubBackend("focus")), memory, 6, actions.append))
    assert actions == []


def test_tick_gives_focus_a_clip_not_one_frame(tmp_path):
    memory = _mem(tmp_path)
    for s in ["a", "b", "c", "the mug is empty"]:
        memory.observe_frame(Frame(ts=0.0, meta={"scene": s}))
    cap = _CapturingFocusBackend()
    asyncio.run(looplib._tick(Glance(StubBackend("glance")), Focus(cap), memory, 3, [].append))
    assert cap.n_frames == 3  # last 3 buffered frames, not just the latest


# --- run(): parallel capture, clean shutdown, resilience ---

def test_run_terminates_on_finite_source(tmp_path):
    sensor = _ScriptedSensor(["a", "b", "the mug is empty"])
    glance = Glance(StubBackend("glance"))
    focus = Focus(StubBackend("focus"))
    actions = []
    # completes (doesn't hang) once the source is exhausted
    asyncio.run(looplib.run(sensor, glance, focus, _mem(tmp_path), on_action=actions.append, glance_fps=0))


def test_run_survives_a_failing_backend(tmp_path):
    sensor = _ScriptedSensor(["the mug is empty"])
    glance = Glance(_BoomBackend())
    focus = Focus(StubBackend("focus"))
    actions = []
    asyncio.run(looplib.run(sensor, glance, focus, _mem(tmp_path), on_action=actions.append, glance_fps=0))
    assert actions == []  # glance kept failing; no crash, no action
