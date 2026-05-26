"""Focus — sharp attention. The expensive model, only on escalation.

It reasons over the trigger + recent context + your preferences and decides
whether to speak and what to say. There is no rule gating this: "should I
interrupt?" is the model's single judgment, informed by context we hand it.
"""

from __future__ import annotations

from saccade.backends.base import Backend
from saccade.memory import Memory
from saccade.schema import DECISION_SCHEMA, Decision, Percept, Window, decision_from

PROMPT = """You are the focused attention of an ambient assistant. Something caught \
the periphery's eye and you are now looking closely — at a short clip of the last \
few seconds (the images are in time order) — to decide whether to speak.

What just stood out: {summary}
Recently observed: {recent}
What the user has told you about when to help (their preferences):
{prefs}

Decide whether saying something now is genuinely worth interrupting the user. \
Respect their preferences. If it isn't worth it, stay quiet. Respond with ONLY \
a JSON object (think in `reasoning` first, then decide):
{{
  "reasoning": "weigh it: is this worth interrupting, given their preferences?",
  "speak": true,
  "message": "what you'd say to the user, short and natural"
}}"""


class Focus:
    def __init__(self, backend: Backend):
        self.backend = backend

    async def reason(self, percept: Percept, window: Window, memory: Memory) -> Decision:
        prompt = PROMPT.format(
            summary=percept.summary,
            recent=memory.working.summary(),
            prefs=memory.semantic.text(),
        )
        raw = await self.backend.complete(prompt, window.frames, schema=DECISION_SCHEMA)
        return decision_from(raw, percept.ts)
