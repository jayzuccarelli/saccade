"""Glance — peripheral awareness. The cheap model's ~1Hz pass.

It answers one question: "is anything here worth a closer look?" Recency lives
in the prompt as CONTEXT (recent percepts), not as an if-branch — the model
decides `escalate` itself, so it won't keep re-flagging the same ongoing thing.
"""

from __future__ import annotations

from saccade.backends.base import Backend
from saccade.memory import Memory
from saccade.schema import PERCEPT_SCHEMA, Percept, Window, percept_from

PROMPT = """You are the peripheral awareness of an ambient assistant. You take a quick \
glance about once a second and decide only whether something is worth a closer look.

Recently you saw: {recent}

Look at the current input and respond with ONLY a JSON object:
{{
  "summary": "one short line: what you see now",
  "tags": ["a", "few", "labels"],
  "salience": 0.0,            // 0-1, how much this stands out as worth attention
  "escalate": false,         // true ONLY if something newly worth a closer look
  "state_delta": "what changed vs what you recently saw"
}}

Do not escalate for things that are simply ongoing and already noted. Escalate \
when something new, useful, or important appears."""


class Glance:
    def __init__(self, backend: Backend):
        self.backend = backend

    async def perceive(self, window: Window, memory: Memory) -> Percept:
        prompt = PROMPT.format(recent=memory.working.summary())
        raw = await self.backend.complete(prompt, window.frames, schema=PERCEPT_SCHEMA)
        return percept_from(raw, window.ts)
