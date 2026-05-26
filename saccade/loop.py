"""The orchestrator. The whole harness in one loop.

    glance every tick -> form a percept -> if salient, focus -> maybe act

If this ever grows past a dozen lines, something leaked into the wrong layer.
"""

from __future__ import annotations

import time
from typing import Callable

from saccade.focus import Focus
from saccade.glance import Glance
from saccade.memory import Memory
from saccade.schema import Percept, Window


def _log(p: Percept) -> None:
    mark = "  ‼  escalate" if p.escalate else ""
    print(f"[glance] sal={p.salience:0.1f}  {p.summary[:64]:<64}{mark}")


async def run(
    sensor,
    glance: Glance,
    focus: Focus,
    memory: Memory,
    on_action: Callable[[str], None] = print,
    glance_fps: float = 1.0,
    focus_clip_frames: int = 6,
) -> None:
    # Capture rate (how fast frames arrive) is the sensor's; glance_fps is how
    # often we actually call the model. Every frame is buffered; we only glance
    # on the slower clock. glance_fps <= 0 means "glance every captured frame".
    interval = 1.0 / glance_fps if glance_fps > 0 else 0.0
    last_glance = float("-inf")

    async for frame in sensor.stream():
        # An always-on agent must survive a flaky API call or a bad response.
        # Log the tick's error and keep watching — never let one frame kill it.
        try:
            memory.observe_frame(frame)  # buffer every frame, at capture rate
            now = time.monotonic()
            if now - last_glance < interval:
                continue  # captured, but not yet time to glance
            last_glance = now
            # Glance sees only the latest frame (peripheral, low-acuity).
            percept = await glance.perceive(Window(frames=[frame]), memory)
            memory.observe(percept)
            _log(percept)
            if percept.escalate:
                # Focus sees a short clip (the last few seconds) so it reads motion.
                clip = memory.sensory.recent(focus_clip_frames)
                decision = await focus.reason(percept, Window(frames=clip), memory)
                if decision.speak:
                    memory.episodic.record(
                        "action", {"message": decision.message, "trigger": percept.summary}
                    )
                    on_action(decision.message)
        except Exception as e:  # noqa: BLE001 — resilience is the whole point here
            print(f"[loop] skipped a tick: {type(e).__name__}: {e}")
