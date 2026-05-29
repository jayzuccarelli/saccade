"""The orchestrator. Two coroutines running in parallel, like the brain:

  capture     — continuously fills the sensory buffer at the sensor's rate
  glance loop — samples the buffer on its own (glance_fps) clock and reasons

Perception never pauses while the model thinks. If glance_fps >= capture_fps,
every captured frame gets glanced (lockstep — good for replay). If it's lower
(e.g. rate-limited live), the loop samples the latest while the buffer still
holds a dense clip for Focus.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from typing import Awaitable, Callable, Union

# on_action may be sync (print) or async (a Speaker.say) — the loop handles both.
Action = Callable[[str], Union[None, Awaitable[None]]]

from saccade.focus import Focus
from saccade.glance import Glance
from saccade.memory import Memory
from saccade.schema import Percept, Window


def _log(p: Percept) -> None:
    mark = "  ‼  escalate" if p.escalate else ""
    print(f"[glance] sal={p.salience:0.1f}  {p.summary[:64]:<64}{mark}")


async def _tick(
    glance: Glance,
    focus: Focus,
    memory: Memory,
    focus_clip_frames: int,
    on_action: Action,
) -> None:
    """One glance at the latest frame; if salient, one focused look at a clip,
    and maybe act. Pure function of the buffer's current state — easy to test."""
    latest = memory.sensory.recent(1)
    if not latest:
        return
    percept = await glance.perceive(Window(frames=latest), memory)
    memory.observe(percept)
    _log(percept)
    if percept.escalate:
        clip = memory.sensory.recent(focus_clip_frames)
        decision = await focus.reason(percept, Window(frames=clip), memory)
        # Log every verdict, not just spoken ones — otherwise deliberate silence
        # (Focus judging it not worth interrupting) looks identical to a dead path.
        print(f"[focus]  speak={str(decision.speak):5}  {decision.reasoning[:80]}")
        if decision.speak:
            memory.episodic.record(
                "action", {"message": decision.message, "trigger": percept.summary}
            )
            result = on_action(decision.message)
            if inspect.isawaitable(result):
                await result


async def run(
    sensor,
    glance: Glance,
    focus: Focus,
    memory: Memory,
    on_action: Action = print,
    glance_fps: float = 1.0,
    focus_clip_frames: int = 6,
) -> None:
    interval = 1.0 / glance_fps if glance_fps > 0 else 0.0
    stream_done = asyncio.Event()

    async def capture() -> None:
        # Never pauses while the model thinks — keeps the buffer current.
        try:
            async for frame in sensor.stream():
                memory.observe_frame(frame)
        finally:
            stream_done.set()

    capture_task = asyncio.create_task(capture())
    try:
        while True:
            # Resilience: one flaky call/response skips a tick, never kills the agent.
            try:
                await _tick(glance, focus, memory, focus_clip_frames, on_action)
            except Exception as e:  # noqa: BLE001 — resilience is the whole point here
                print(f"[loop] skipped a tick: {type(e).__name__}: {e}")
            if stream_done.is_set() and capture_task.done():
                break
            await asyncio.sleep(interval)
    finally:
        capture_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await capture_task  # re-raises a real sensor error (e.g. camera won't open)
