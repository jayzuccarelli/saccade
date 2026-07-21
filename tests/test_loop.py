"""Loop: a glance reasons over a clip; the agent survives bad ticks and stops
cleanly when a finite source ends. The per-tick logic is tested directly (no
timing); the parallel capture/shutdown is tested through run()."""

import asyncio
import os

from saccade import loop as looplib
from saccade.backends.stub import StubBackend
from saccade.focus import Focus
from saccade.glance import Glance
from saccade.loop import _next_interval
from saccade.memory import Memory
from saccade.schema import Frame, Percept


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


class _FlappingBackend:
    """Fails every other call: an intermittent fault, not a dead backend."""

    def __init__(self):
        self.calls = 0

    async def complete(self, prompt, frames, schema=None):
        self.calls += 1
        if self.calls % 2:
            raise RuntimeError("boom")
        return '{"reasoning":"r","speak":false,"message":""}'


def _mem(tmp_path):
    return Memory(str(tmp_path / "ep.jsonl"), str(tmp_path / "prefs.md"))


# --- per-tick logic: deterministic, no timing ---


def test_tick_salient_frame_triggers_action(tmp_path):
    memory = _mem(tmp_path)
    memory.observe_frame(Frame(ts=0.0, meta={"scene": "the mug is empty"}))
    actions = []
    asyncio.run(
        looplib._tick(
            Glance(StubBackend("glance")), Focus(StubBackend("focus")), memory, 6, actions.append
        )
    )
    assert len(actions) == 1


def test_tick_quiet_frame_stays_silent(tmp_path):
    memory = _mem(tmp_path)
    memory.observe_frame(Frame(ts=0.0, meta={"scene": "person typing"}))
    actions = []
    asyncio.run(
        looplib._tick(
            Glance(StubBackend("glance")), Focus(StubBackend("focus")), memory, 6, actions.append
        )
    )
    assert actions == []


def test_tick_gives_focus_a_clip_not_one_frame(tmp_path):
    memory = _mem(tmp_path)
    for s in ["a", "b", "c", "the mug is empty"]:
        memory.observe_frame(Frame(ts=0.0, meta={"scene": s}))
    cap = _CapturingFocusBackend()
    asyncio.run(looplib._tick(Glance(StubBackend("glance")), Focus(cap), memory, 3, [].append))
    assert cap.n_frames == 3  # last 3 buffered frames, not just the latest


# --- adaptive cadence: model-suggested interval, clamped ---


def _p(next_glance_s):
    return Percept(ts=0.0, next_glance_s=next_glance_s)


def test_next_interval_is_fixed_floor_without_adaptive():
    # adaptive off: the model's suggestion is ignored, cadence stays the floor.
    assert _next_interval(_p(9.0), floor=1.0, ceiling=15.0, adaptive=False) == 1.0


def test_next_interval_uses_suggestion_within_bounds():
    assert _next_interval(_p(6.0), floor=1.0, ceiling=15.0, adaptive=True) == 6.0


def test_next_interval_clamps_to_floor_and_ceiling():
    # never faster than the floor (respects a rate limit)...
    assert _next_interval(_p(0.2), floor=1.0, ceiling=15.0, adaptive=True) == 1.0
    # ...and never rests longer than the ceiling.
    assert _next_interval(_p(99.0), floor=1.0, ceiling=15.0, adaptive=True) == 15.0


def test_next_interval_falls_back_when_no_suggestion():
    # a stub/older model that emits no next_glance_s (0) or no percept -> floor.
    assert _next_interval(_p(0.0), floor=2.0, ceiling=15.0, adaptive=True) == 2.0
    assert _next_interval(None, floor=2.0, ceiling=15.0, adaptive=True) == 2.0


def test_tick_returns_percept_for_pacing(tmp_path):
    memory = _mem(tmp_path)
    memory.observe_frame(Frame(ts=0.0, meta={"scene": "person typing"}))
    percept = asyncio.run(
        looplib._tick(
            Glance(StubBackend("glance")), Focus(StubBackend("focus")), memory, 6, [].append
        )
    )
    assert percept is not None and percept.summary  # the tick hands back what it saw


# --- run(): parallel capture, clean shutdown, resilience ---


def test_run_terminates_on_finite_source(tmp_path):
    sensor = _ScriptedSensor(["a", "b", "the mug is empty"])
    glance = Glance(StubBackend("glance"))
    focus = Focus(StubBackend("focus"))
    actions = []
    # completes (doesn't hang) once the source is exhausted
    asyncio.run(
        looplib.run(sensor, glance, focus, _mem(tmp_path), on_action=actions.append, glance_fps=0)
    )


def test_run_survives_a_failing_backend(tmp_path):
    sensor = _ScriptedSensor(["the mug is empty"])
    glance = Glance(_BoomBackend())
    focus = Focus(StubBackend("focus"))
    actions = []
    asyncio.run(
        looplib.run(sensor, glance, focus, _mem(tmp_path), on_action=actions.append, glance_fps=0)
    )
    assert actions == []  # glance kept failing; no crash, no action


def test_intermittent_failure_is_reported_each_time_it_returns(tmp_path, capsys):
    """A tick that worked ends the streak. Without that, the second failure looks
    like a continuation of the first and stays hidden for _REPEAT_EVERY more hits,
    which is how a flaky camera quietly becomes a dead one. Paced rather than
    instant, because the suppression only shows up across several ticks."""

    class _PacedSensor:
        async def stream(self):
            for i in range(6):
                yield Frame(ts=float(i), image=None, meta={"scene": f"s{i}"})
                await asyncio.sleep(0.05)

    glance = Glance(_FlappingBackend())
    asyncio.run(
        looplib.run(
            _PacedSensor(),
            glance,
            Focus(StubBackend("focus")),
            _mem(tmp_path),
            on_action=[].append,
            glance_fps=50.0,
        )
    )
    reported = capsys.readouterr().out.count("skipped a tick")
    # Without the reset this is exactly 1, however many times it actually failed.
    assert reported >= 2, f"intermittent failures were swallowed, saw {reported}"


class _SlowFocusBackend:
    """A Focus that blocks until released. Stands in for a slow big model."""

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, prompt, frames, schema=None):
        self.entered.set()
        await self.release.wait()
        return '{"reasoning":"r","speak":true,"message":"hi"}'


def test_run_concurrent_focus_completes_backgrounded_action(tmp_path):
    """A Focus spawned in the background must still run its action to completion
    before run() returns: the shutdown drains the in-flight task."""
    slow = _SlowFocusBackend()

    class _Sensor:
        async def stream(self):
            yield Frame(ts=0.0, meta={"scene": "the mug is empty"})
            # give the spawned Focus a turn to start, then let it finish
            await slow.entered.wait()
            slow.release.set()

    actions = []
    asyncio.run(
        looplib.run(
            _Sensor(),
            Glance(StubBackend("glance")),
            Focus(slow),
            _mem(tmp_path),
            on_action=actions.append,
            glance_fps=0,
            concurrent_focus=True,
        )
    )
    assert actions == ["hi"]  # backgrounded Focus was drained and spoke


def test_concurrent_focus_is_single_slot(tmp_path):
    """While one Focus is in flight, further escalations don't stack a second;
    Glance keeps observing but only one Focus runs at a time."""
    slow = _SlowFocusBackend()
    starts = {"n": 0}
    orig = slow.complete

    async def counting(*a, **k):
        starts["n"] += 1
        return await orig(*a, **k)

    slow.complete = counting

    class _Sensor:
        async def stream(self):
            # three escalating frames back-to-back while Focus is blocked
            for _ in range(3):
                yield Frame(ts=0.0, meta={"scene": "the mug is empty"})
            await slow.entered.wait()
            slow.release.set()

    asyncio.run(
        looplib.run(
            _Sensor(),
            Glance(StubBackend("glance")),
            Focus(slow),
            _mem(tmp_path),
            on_action=[].append,
            glance_fps=0,
            concurrent_focus=True,
        )
    )
    assert starts["n"] == 1  # only one Focus ran despite three escalations


def _percept(summary: str) -> Percept:
    return Percept(ts=0.0, summary=summary, salience=0.1, escalate=False, next_glance_s=0.0)


def test_the_log_uses_the_terminal_it_is_actually_in(monkeypatch, capsys):
    """The summary column was hardcoded at 64, narrower than any terminal anyone
    uses, so every line was cut mid-word on a real run."""
    monkeypatch.setattr(
        looplib.shutil, "get_terminal_size", lambda fallback=None: os.terminal_size((200, 24))
    )
    looplib._log(_percept("x" * 150))
    assert "x" * 150 in capsys.readouterr().out


def test_a_narrow_terminal_still_gets_a_readable_column(monkeypatch, capsys):
    """Shrinking without a floor turns the log into one character per line."""
    monkeypatch.setattr(
        looplib.shutil, "get_terminal_size", lambda fallback=None: os.terminal_size((20, 24))
    )
    looplib._log(_percept("y" * 150))
    assert "y" * 39 in capsys.readouterr().out


def test_an_overlong_summary_is_marked_as_cut(monkeypatch, capsys):
    """A hard slice reads as the model having stopped mid-sentence. It didn't;
    we cut it, and the line should say so."""
    monkeypatch.setattr(
        looplib.shutil, "get_terminal_size", lambda fallback=None: os.terminal_size((80, 24))
    )
    looplib._log(_percept("z" * 300))
    assert "…" in capsys.readouterr().out


def test_the_escalate_marker_survives_a_long_summary(monkeypatch, capsys):
    """The reserved columns exist so the one flag that matters isn't what gets
    pushed off the end."""
    monkeypatch.setattr(
        looplib.shutil, "get_terminal_size", lambda fallback=None: os.terminal_size((80, 24))
    )
    looplib._log(Percept(ts=0.0, summary="w" * 300, salience=0.9, escalate=True, next_glance_s=2.0))
    out = capsys.readouterr().out
    assert "escalate" in out and "⟳2s" in out
