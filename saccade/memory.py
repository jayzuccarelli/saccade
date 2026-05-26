"""Memory stores, mirroring the canonical hierarchy (sensory -> working ->
long-term), defined now so there's no migration later.

- sensory   : volatile ring buffer of recent raw FRAMES. "what did the eye just see?"
- working   : volatile ring buffer of recent percepts.    "what's happening now?"
- episodic  : durable append-only event log.              "what happened?"
- semantic  : durable, human-readable user model.          "what's true about you?"

Sensory exists so Focus can be handed a short *clip* (motion), not one freeze
frame. Consolidation (learning episodic -> semantic) is still deferred behavior.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque

from saccade.schema import Frame, Percept


class SensoryMemory:
    """Raw recent frames, FIFO. Held just long enough to give Focus a clip of the
    last few seconds so it perceives motion, not a still."""

    def __init__(self, maxlen: int = 16):
        self.buf: deque[Frame] = deque(maxlen=maxlen)

    def observe(self, frame: Frame) -> None:
        self.buf.append(frame)

    def recent(self, n: int) -> list[Frame]:
        return list(self.buf)[-n:]


class WorkingMemory:
    def __init__(self, maxlen: int = 30):
        self.buf: deque[Percept] = deque(maxlen=maxlen)

    def observe(self, percept: Percept) -> None:
        self.buf.append(percept)

    def recent(self, n: int = 8) -> list[Percept]:
        return list(self.buf)[-n:]

    def summary(self, n: int = 8) -> str:
        return " | ".join(p.summary for p in self.recent(n)) or "(nothing yet)"


class EpisodicMemory:
    """Append-only JSONL log + an in-RAM tail for fast recall (e.g. "what did I
    just say?"). The file is the durable record; the tail is for live decisions."""

    def __init__(self, path: str = "episodic.jsonl", tail: int = 50):
        self.path = path
        self.tail: deque[dict] = deque(maxlen=tail)

    def record(self, kind: str, payload: dict) -> None:
        entry = {"ts": time.time(), "kind": kind, **payload}
        self.tail.append(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def recent(self, n: int = 5, kind: str | None = None) -> list[dict]:
        items = [e for e in self.tail if kind is None or e["kind"] == kind]
        return items[-n:]


class SemanticMemory:
    """Durable user model / preferences as plain markdown the model reads.
    Auto-learning is deferred; in v0 you hand-edit this file."""

    def __init__(self, path: str = "preferences.md"):
        self.path = path

    def text(self) -> str:
        if os.path.exists(self.path):
            return open(self.path).read().strip()
        return "(no preferences set yet)"


class Memory:
    def __init__(
        self,
        episodic_path: str,
        preferences_path: str,
        sensory_n: int = 16,
        working_n: int = 30,
    ):
        self.sensory = SensoryMemory(sensory_n)
        self.working = WorkingMemory(working_n)
        self.episodic = EpisodicMemory(episodic_path)
        self.semantic = SemanticMemory(preferences_path)

    def observe(self, percept: Percept) -> None:
        self.working.observe(percept)

    def observe_frame(self, frame: Frame) -> None:
        self.sensory.observe(frame)
