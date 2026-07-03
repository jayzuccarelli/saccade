"""The contracts every part speaks. Single source of truth.

Stdlib dataclasses (no install) so the harness runs anywhere. Percept is the
note Glance produces each tick; Decision is what Focus returns. Models emit
these as JSON, parsed leniently below.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field


@dataclass
class Frame:
    """One captured sample from a Sensor. A frame can carry an image, an audio
    clip, or both — a webcam+mic sensor could fill both for the same instant."""

    ts: float
    image: bytes | None = None  # JPEG/PNG bytes for vision; None for audio/text/stub
    mime: str = "image/jpeg"
    audio: bytes | None = None  # WAV bytes for hearing; None for vision/text/stub
    audio_mime: str = "audio/wav"
    meta: dict = field(default_factory=dict)


@dataclass
class Window:
    """The sensory slice handed to a model on one tick."""

    frames: list[Frame] = field(default_factory=list)

    @property
    def ts(self) -> float:
        return self.frames[-1].ts if self.frames else time.time()


@dataclass
class Percept:
    """Glance's structured observation. `escalate` is the model's OWN call —
    there is no threshold or rule outside the model."""

    ts: float
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    salience: float = 0.0
    escalate: bool = False
    state_delta: str = ""


@dataclass
class Decision:
    """Focus's verdict: whether to speak, and what."""

    ts: float
    speak: bool = False
    message: str = ""
    reasoning: str = ""


def _loads_lenient(raw: str) -> dict:
    """Pull the first {...} block out of a model response and parse it."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


# Neutral JSON Schemas — provider-agnostic. Each Backend translates these into
# its own native structured-output mechanism. Roles own their schema; backends
# never hardcode shape. (All keys required + additionalProperties:false so
# OpenAI strict mode accepts them.)
PERCEPT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "salience": {"type": "number"},
        "escalate": {"type": "boolean"},
        "state_delta": {"type": "string"},
    },
    "required": ["summary", "tags", "salience", "escalate", "state_delta"],
    "additionalProperties": False,
}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},  # first: model thinks before deciding
        "speak": {"type": "boolean"},
        "message": {"type": "string"},
    },
    "required": ["reasoning", "speak", "message"],
    "additionalProperties": False,
}


def percept_from(raw: str, ts: float) -> Percept:
    d = _loads_lenient(raw)
    return Percept(
        ts=ts,
        summary=d.get("summary", ""),
        tags=list(d.get("tags") or []),
        salience=float(d.get("salience", 0.0) or 0.0),
        escalate=bool(d.get("escalate", False)),
        state_delta=d.get("state_delta", ""),
    )


def decision_from(raw: str, ts: float) -> Decision:
    d = _loads_lenient(raw)
    return Decision(
        ts=ts,
        speak=bool(d.get("speak", False)),
        message=d.get("message", ""),
        reasoning=d.get("reasoning", ""),
    )
