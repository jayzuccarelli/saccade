"""A placeholder backend that MIMICS a model so the loop runs with no API key.

Important: this fakes the *model's* judgment, it is not system logic. The real
model replaces it entirely. The keyword check below stands in for "the model
decided something was worth a closer look" — in production that judgment is the
model's, never a rule in our code.
"""

from __future__ import annotations

import json

from saccade.schema import Frame

_INTERESTING = ("empty", "searching", "stretches", "looking around")


class StubBackend:
    def __init__(self, role: str = "glance"):
        self.role = role

    async def complete(self, prompt: str, frames: list[Frame], schema: dict | None = None) -> str:
        scene = frames[-1].meta.get("scene", "") if frames else ""
        if self.role == "glance":
            interesting = any(k in scene for k in _INTERESTING)
            return json.dumps(
                {
                    "summary": scene or "nothing notable",
                    "tags": scene.split()[:4],
                    "salience": 0.8 if interesting else 0.1,
                    "escalate": interesting,
                    "state_delta": scene,
                    # watch closely when something's up, rest when it's calm
                    "next_glance_s": 1.0 if interesting else 8.0,
                }
            )
        return json.dumps(
            {
                "speak": True,
                "message": f"Looks like: {scene}. Want a hand with that?",
                "reasoning": "stub focus response",
            }
        )
