"""Focus — sharp attention. The expensive model, only on escalation.

It reasons over the trigger + recent context + your preferences and decides
whether to speak and what to say. There is no rule gating this: "should I
interrupt?" is the model's single judgment, informed by context we hand it.
"""

from __future__ import annotations

import time

from saccade.backends.base import Backend
from saccade.memory import Memory
from saccade.schema import DECISION_SCHEMA, Decision, Percept, Window, decision_from

PROMPT = """You are the focused attention of an ambient assistant. Something caught \
the periphery's eye and you are now looking closely — at a short clip of the last \
few seconds (the images are in time order) — to decide whether to speak.

What just stood out: {summary}
Recently observed: {recent}
What you recently said to the user:
{recent_said}
What the user has told you about when to help (their preferences):
{prefs}

Decide whether saying something now is genuinely worth interrupting the user. \
Don't repeat something you just said — but DO speak up if something new or urgent \
warrants it, even right after speaking. Respect their preferences. If it isn't \
worth it, stay quiet. Respond with ONLY a JSON object (think in `reasoning` first, \
then decide):
{{
  "reasoning": "weigh it: is this worth interrupting, given their preferences?",
  "speak": true,
  "message": "what you'd say to the user, short and natural"
}}"""


class Focus:
    def __init__(self, backend: Backend, recent_said_window_s: float = 180.0):
        self.backend = backend
        self.window_s = recent_said_window_s

    def _recent_said(self, memory: Memory) -> str:
        now = time.time()
        actions = [
            a for a in memory.episodic.recent(5, kind="action") if now - a["ts"] <= self.window_s
        ]
        if not actions:
            return "(you haven't said anything recently)"
        return "\n".join(f'- {int(now - a["ts"])}s ago: "{a.get("message", "")}"' for a in actions)

    async def reason(self, percept: Percept, window: Window, memory: Memory) -> Decision:
        prompt = PROMPT.format(
            summary=percept.summary,
            recent=memory.working.summary(),
            recent_said=self._recent_said(memory),
            prefs=memory.semantic.text(),
        )
        raw = await self.backend.complete(prompt, window.frames, schema=DECISION_SCHEMA)
        return decision_from(raw, percept.ts)
