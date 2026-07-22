"""The orchestrator. Coroutines running in parallel, like the brain:

  capture:      continuously fills the sensory buffer at the sensor's rate
  glance loop:  samples the buffer on its own (glance_fps) clock and reasons
  focus:        with concurrent_focus, a salient frame spawns a background
                Focus so Glance never goes blind while the big model reasons

Perception never pauses while the model thinks. Capture never blocks on Glance;
with concurrent_focus, Glance never blocks on Focus (single-slot: one Focus at
a time, so slow reasoning doesn't stack interruptions). If glance_fps >=
capture_fps, every captured frame gets glanced (lockstep: good for replay). If
it's lower (e.g. rate-limited live), the loop samples the latest while the buffer
still holds a dense clip for Focus.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import shutil
import sys
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from saccade.focus import Focus
from saccade.glance import Glance
from saccade.memory import Memory
from saccade.schema import Percept, Window

if TYPE_CHECKING:
    from saccade.sensors.base import Sensor

# How often to re-print an error that keeps repeating identically.
_REPEAT_EVERY = 20

# Narrowest summary column we'll accept. Below this the log stops being skimmable,
# so a very narrow terminal wraps instead of shrinking further.
_MIN_SUMMARY = 40

# on_action may be sync (print) or async (a Speaker.say); the loop handles both.
Action = Callable[[str], None | Awaitable[None]]


def _fit(text: str, reserved: int, pad: bool = False) -> str:
    """Trim `text` to what's left of the terminal after `reserved` columns.

    One line per tick is the point: a fixed column you can skim for the moment
    something changed. The width used to be hardcoded at 64, which is narrower
    than any terminal anyone actually uses, so the summaries were cut mid-word
    and the Focus reasoning lost its ending, which is the half that says why."""
    width = max(_MIN_SUMMARY, shutil.get_terminal_size((100, 24)).columns - reserved)
    out = text if len(text) <= width else text[: width - 1] + "…"
    return f"{out:<{width}}" if pad else out


_live_line = False  # a quiet glance is sitting on the current row, unterminated


def _out(text: str, live: bool = False) -> None:
    """Print, clearing any live line first so a kept line never inherits its tail.

    A quiet tick overwrites the last quiet one instead of scrolling. An hour of an
    empty room was 3,600 lines of "a man is sitting at a desk", which buries the
    few lines that meant something, and when a screen is one of the sensors it
    feeds straight back in as input. Anything worth keeping (an escalation, what
    Focus decided, an error) scrolls normally."""
    global _live_line
    if _live_line:
        print("\r\033[K", end="")  # back to column 0, erase to end of line
    # Always flushed: at ~1 Hz the cost is nothing, and an agent that runs for
    # hours shouldn't lose its log to a block buffer when someone kills it.
    print(text, end="\r" if live else "\n", flush=True)
    _live_line = live


def _log(p: Percept) -> None:
    mark = "  ‼  escalate" if p.escalate else ""
    cadence = f"  ⟳{p.next_glance_s:0.0f}s" if p.next_glance_s > 0 else ""
    # "[glance] sal=0.1  " + the widest mark + the widest cadence.
    summary = _fit(p.summary, len("[glance] sal=0.1  ") + 13 + 7, pad=True)
    line = f"[glance] sal={p.salience:0.1f}  {summary}{mark}{cadence}"
    # Not a terminal: keep every tick, since that's a log someone will read later.
    _out(line, live=sys.stdout.isatty() and not p.escalate)


def _next_interval(percept: Percept | None, floor: float, ceiling: float, adaptive: bool) -> float:
    """How long to wait before the next glance. Without adaptive cadence (or a
    suggestion) it's the fixed floor. With it, the model's `next_glance_s` sets
    the pace, but clamped to [floor, ceiling]: floor is the fastest rate (the
    configured glance_fps, which may exist to respect a rate limit), ceiling caps
    how long we'll rest on a calm scene. Adaptive only ever slows us down."""
    if not adaptive or percept is None or percept.next_glance_s <= 0:
        return floor
    return max(floor, min(percept.next_glance_s, ceiling))


async def _glance(glance: Glance, memory: Memory) -> Percept | None:
    """Sample the latest frame and observe a Percept. Returns None if the buffer's
    empty. The cheap, serial half of a tick (always runs on the glance clock)."""
    # One frame per input, not just the newest frame: with a camera and a mic both
    # running, the camera wins nearly every tick and the room is never heard.
    latest = memory.sensory.latest_per_source()
    if not latest:
        return None
    percept = await glance.perceive(Window(frames=latest), memory)
    memory.observe(percept)
    _log(percept)
    return percept


async def _focus_act(
    focus: Focus,
    memory: Memory,
    percept: Percept,
    focus_clip_frames: int,
    on_action: Action,
) -> None:
    """The expensive half: reason over a clip and maybe speak. Can be slow (the
    big model), so run() may run this concurrently while Glance keeps watching.
    It owns its own resilience: a failure here must never take down the loop or
    surface as an unretrieved task exception."""
    try:
        clip = memory.sensory.recent(focus_clip_frames)
        decision = await focus.reason(percept, Window(frames=clip), memory)
        # Log every verdict, not just spoken ones; otherwise deliberate silence
        # (Focus judging it not worth interrupting) looks identical to a dead path.
        _out(f"[focus]  speak={str(decision.speak):5}  {_fit(decision.reasoning, 22)}")
        if decision.speak:
            memory.episodic.record(
                "action", {"message": decision.message, "trigger": percept.summary}
            )
            result = on_action(decision.message)
            if inspect.isawaitable(result):
                await result
    except Exception as e:  # noqa: BLE001 (a bad Focus must not kill the agent)
        _out(f"[focus]  skipped: {type(e).__name__}: {e}")


async def _tick(
    glance: Glance,
    focus: Focus,
    memory: Memory,
    focus_clip_frames: int,
    on_action: Action,
) -> Percept | None:
    """One glance, and, if salient, a focused look, inline (serial). Returns the
    Percept (for pacing) or None if the buffer's empty. run() uses this for the
    non-concurrent path; it's also the easy-to-test unit of loop behavior."""
    percept = await _glance(glance, memory)
    if percept and percept.escalate:
        await _focus_act(focus, memory, percept, focus_clip_frames, on_action)
    return percept


async def run(
    sensor: Sensor,
    glance: Glance,
    focus: Focus,
    memory: Memory,
    on_action: Action = print,
    glance_fps: float = 1.0,
    focus_clip_frames: int = 6,
    adaptive_cadence: bool = False,
    glance_max_interval: float = 15.0,
    concurrent_focus: bool = False,
) -> None:
    floor = 1.0 / glance_fps if glance_fps > 0 else 0.0
    stream_done = asyncio.Event()
    focus_task: asyncio.Task[None] | None = None  # single in-flight Focus (concurrent mode)
    last_err, repeats = "", 0

    async def capture() -> None:
        # Never pauses while the model thinks: keeps the buffer current.
        try:
            async for frame in sensor.stream():
                memory.observe_frame(frame)
        finally:
            stream_done.set()

    capture_task = asyncio.create_task(capture())
    try:
        while True:
            # Resilience: one flaky call/response skips a tick, never kills the agent.
            percept = None
            try:
                if concurrent_focus:
                    # Glance stays on its clock; a salient frame kicks off Focus in
                    # the background so perception never goes blind while it reasons.
                    # Single-slot: if a Focus is still running, keep watching rather
                    # than stacking interruptions; Glance still observes the frames.
                    percept = await _glance(glance, memory)
                    if percept and percept.escalate and (focus_task is None or focus_task.done()):
                        focus_task = asyncio.create_task(
                            _focus_act(focus, memory, percept, focus_clip_frames, on_action)
                        )
                else:
                    percept = await _tick(glance, focus, memory, focus_clip_frames, on_action)
                # A tick that worked ends the streak. Without this an intermittent
                # fault reports once and then hides for _REPEAT_EVERY more hits,
                # and the "still failing" count describes a streak that already
                # recovered.
                last_err, repeats = "", 0
            except ModuleNotFoundError:
                # Not a bad tick, a bad install: the SDK for the configured
                # backend isn't here and never will be at this rate. Retrying it
                # every second forever looks like the agent is working when it
                # has not made a single model call. Fail out and let the CLI turn
                # it into the install command.
                raise
            except Exception as e:  # noqa: BLE001 (resilience is the whole point here)
                # A broken backend fails identically every tick. Say it once and
                # then stay quiet, or the one line that tells you how to fix it
                # scrolls away under a thousand copies of itself.
                msg = f"{type(e).__name__}: {e}"
                if msg == last_err:
                    repeats += 1
                    if repeats % _REPEAT_EVERY == 0:
                        _out(f"[loop] still failing ({repeats + 1}x): {msg}")
                else:
                    _out(f"[loop] skipped a tick: {msg}")
                    last_err, repeats = msg, 0
            if stream_done.is_set() and capture_task.done():
                break
            await asyncio.sleep(
                _next_interval(percept, floor, glance_max_interval, adaptive_cadence)
            )
    finally:
        capture_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await capture_task  # re-raises a real sensor error (e.g. camera won't open)
        # Let an in-flight Focus finish speaking before we exit (it's one model
        # call + one action, not an unbounded loop). _focus_act never raises.
        if focus_task is not None:
            await focus_task
