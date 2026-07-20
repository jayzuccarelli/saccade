"""Measure Glance's salience judgment, the hard part, instead of tuning by vibe.

An eval is a set of cases (a scene + whether it *should* have been flagged worth
a closer look). The runner asks Glance, compares to the label, and scores it:
precision (when it flags, is it right?) and recall (does it catch what matters?).

    python -m saccade.evals                 # run evals/scenes.json with the configured backend
    python -m saccade.evals my_cases.json

Cases use either `scene` (text, for the stub) or `image` (a file path, for real
vision models). Tune a prompt, re-run, watch the numbers move.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from saccade.config import Config
from saccade.memory import Memory
from saccade.schema import Frame, Window

if TYPE_CHECKING:
    from saccade.glance import Glance


@dataclass
class Case:
    name: str
    expect_escalate: bool
    scene: str | None = None  # text, for the stub backend
    image: str | None = None  # file path, for real vision models


def _frame(case: Case) -> Frame:
    if case.image:
        with open(case.image, "rb") as f:
            data = f.read()
        mime = "image/png" if case.image.lower().endswith(".png") else "image/jpeg"
        return Frame(ts=0.0, image=data, mime=mime)
    return Frame(ts=0.0, meta={"scene": case.scene or ""})


def score(results: list[tuple[bool, bool]]) -> dict[str, float]:
    """results: list of (expected, got) for the escalate decision."""
    tp = sum(1 for e, g in results if e and g)
    fp = sum(1 for e, g in results if not e and g)
    fn = sum(1 for e, g in results if e and not g)
    tn = sum(1 for e, g in results if not e and not g)
    n = len(results)
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 1.0,
        "recall": tp / (tp + fn) if (tp + fn) else 1.0,
    }


async def evaluate(
    cases: list[Case], glance: Glance, memory_factory: Callable[[], Memory]
) -> tuple[dict[str, float], list[tuple[str, bool, bool, float]]]:
    """Run Glance over every case (fresh memory each, so cases don't bleed).
    Returns (metrics, rows) where rows = [(name, expected, got, salience)]."""
    rows: list[tuple[str, bool, bool, float]] = []
    for c in cases:
        frame = _frame(c)
        memory = memory_factory()
        memory.observe_frame(frame)
        percept = await glance.perceive(Window(frames=[frame]), memory)
        rows.append((c.name, c.expect_escalate, percept.escalate, percept.salience))
    metrics = score([(e, g) for _, e, g, _ in rows])
    return metrics, rows


def _load(path: str) -> list[Case]:
    with open(path) as f:
        return [Case(**c) for c in json.load(f)]


async def _main(path: str) -> None:
    from saccade.__main__ import make_backend  # reuse the provider factory
    from saccade.glance import Glance

    c = Config()
    cases = _load(path)
    glance = Glance(make_backend(c.glance_backend, "glance", c), max_dim=c.glance_max_dim)
    tmp = tempfile.mkdtemp()

    def fresh_memory() -> Memory:
        return Memory(f"{tmp}/ep.jsonl", f"{tmp}/prefs.md")

    metrics, rows = await evaluate(cases, glance, fresh_memory)
    print(f"\nsaccade evals: backend={c.glance_backend}  ({metrics['n']} cases)\n")
    for name, exp, got, sal in rows:
        mark = "ok " if exp == got else "MISS"
        print(f"  [{mark}] expect={int(exp)} got={int(got)} sal={sal:0.2f}  {name}")
    print(
        f"\n  accuracy={metrics['accuracy']:.2f}  precision={metrics['precision']:.2f}  "
        f"recall={metrics['recall']:.2f}  (tp={metrics['tp']} fp={metrics['fp']} "
        f"fn={metrics['fn']} tn={metrics['tn']})\n"
    )


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1] if len(sys.argv) > 1 else "evals/scenes.json"))
