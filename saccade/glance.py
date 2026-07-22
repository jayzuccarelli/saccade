"""Glance: peripheral awareness. The cheap model's ~1Hz pass.

It answers one question: "is anything here worth a closer look?" Recency lives
in the prompt as CONTEXT (recent percepts), not as an if-branch; the model
decides `escalate` itself, so it won't keep re-flagging the same ongoing thing.
"""

from __future__ import annotations

from dataclasses import replace

from saccade.backends.base import Backend
from saccade.imageutil import downscale
from saccade.memory import Memory
from saccade.schema import PERCEPT_SCHEMA, Frame, Percept, Window, percept_from

PROMPT = """You are the peripheral awareness of an ambient assistant. You take a quick \
glance about once a second and decide only whether something is worth a closer look.

Recently you saw (newest last; a line marked [escalated] is one you ALREADY flagged \
for a closer look):
{recent}

Take in the current input (an image, a short audio clip, or both) and respond \
with ONLY a JSON object:
{{
  "summary": "who or what is there and what they are doing, in a sentence",
  "tags": ["a", "few", "labels"],
  "salience": 0.0,            // 0-1, how much this stands out as worth attention
  "escalate": false,         // true ONLY if something newly worth a closer look
  "state_delta": "what changed vs what you recently saw, or 'nothing changed'",
  "next_glance_s": 1.0       // how soon the next glance is worth taking (seconds)
}}

`summary` is the only record of this moment: it is what you will be shown as \
CONTEXT on later glances, and all the closer look gets to work from. So write \
what someone who cannot see the input would need. "a man at a desk, typing, \
nobody else in the room" is a summary. "man" is a label, and a run of them is \
indistinguishable from nothing happening. Name the objects and the action, and \
say when you are unsure rather than guessing.

Describe THIS input. The CONTEXT above is there so you can tell what changed; \
repeating its wording back is how a run of identical summaries starts, and once \
started it hides everything that happens next.

A screen may be showing this assistant's own output: lines like `[glance] sal=0.6` \
are you, a moment ago. Reading your own log back is not an event, and neither is \
someone typing about it. Report what the person is doing, not the text you \
recognise on their screen.

Judge change, not the static scene: someone who has simply been sitting or standing \
there is ONE ongoing event, not a new one every second. If you already escalated an \
ongoing situation (see the [escalated] lines above), do NOT escalate it again. \
Escalate only when something genuinely new appears, or an ongoing thing meaningfully \
changes: they get up, a new person enters, they start searching for something.

Set `next_glance_s` by how much is happening: a small value (~1s) when the scene is \
active or changing and you want to watch closely, a larger one (up to ~15s) when it is \
calm and static and nothing needs a quick recheck. This paces your own attention: \
watch hard when it matters, rest when it doesn't."""


class Glance:
    def __init__(self, backend: Backend, max_dim: int = 0):
        self.backend = backend
        self.max_dim = max_dim  # peripheral vision is low-acuity: shrink to save tokens

    def _recent(self, memory: Memory, n: int = 8) -> str:
        """Recent percepts, marking the ones already escalated, so 'newly worth a
        closer look' has an anchor and an ongoing event isn't re-flagged each tick."""
        percepts = memory.working.recent(n)
        if not percepts:
            return "(nothing yet)"
        return " | ".join(f"[escalated] {p.summary}" if p.escalate else p.summary for p in percepts)

    def _downscaled(self, window: Window) -> list[Frame]:
        if not self.max_dim:
            return window.frames
        out: list[Frame] = []
        for f in window.frames:
            shrunk = downscale(f.image, self.max_dim) if f.image else None
            out.append(replace(f, image=shrunk[0], mime=shrunk[1]) if shrunk else f)
        return out

    async def perceive(self, window: Window, memory: Memory) -> Percept:
        prompt = PROMPT.format(recent=self._recent(memory))
        raw = await self.backend.complete(prompt, self._downscaled(window), schema=PERCEPT_SCHEMA)
        return percept_from(raw, window.ts)
