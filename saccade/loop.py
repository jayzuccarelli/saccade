"""The orchestrator. The whole harness in one loop.

    glance every tick -> form a percept -> if salient, focus -> maybe act

If this ever grows past a dozen lines, something leaked into the wrong layer.
"""

from __future__ import annotations

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
    focus_clip_frames: int = 6,
) -> None:
    async for frame in sensor.stream():
        # An always-on agent must survive a flaky API call or a bad response.
        # Log the tick's error and keep watching — never let one frame kill it.
        try:
            memory.observe_frame(frame)
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
