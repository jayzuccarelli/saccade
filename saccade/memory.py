"""Three memory stores, defined now so there's no migration later.

- working   : volatile ring buffer of recent percepts. "what's happening now?"
- episodic  : durable append-only event log.            "what happened?"
- semantic  : durable, human-readable user model.        "what's true about you?"

All three exist from day one. Only working is populated automatically in v0;
episodic logs actions; semantic is a hand-edited file. The *consolidation*
(learning episodic -> semantic) is deferred — that's behavior, not structure.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque

from saccade.schema import Percept


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
    """Append-only JSONL log. Queryable by time later; for now, a faithful record."""

    def __init__(self, path: str = "episodic.jsonl"):
        self.path = path

    def record(self, kind: str, payload: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps({"ts": time.time(), "kind": kind, **payload}) + "\n")


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
    def __init__(self, episodic_path: str, preferences_path: str):
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(episodic_path)
        self.semantic = SemanticMemory(preferences_path)

    def observe(self, percept: Percept) -> None:
        self.working.observe(percept)
