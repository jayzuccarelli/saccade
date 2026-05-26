"""The eval scorer's math is exact, and the runner works end-to-end on the stub
(whose escalate logic is known, so the metrics are deterministic)."""

import asyncio

from saccade.evals import Case, evaluate, score
from saccade.glance import Glance
from saccade.memory import Memory
from saccade.backends.stub import StubBackend


def test_score_math():
    # expected, got
    m = score([(True, True), (True, False), (False, True), (False, False)])
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5


def test_score_no_positives_is_perfect_precision_recall():
    m = score([(False, False), (False, False)])
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["accuracy"] == 1.0


def test_evaluate_against_stub_is_deterministic(tmp_path):
    # stub escalates iff scene contains empty/searching/stretches/looking around
    cases = [
        Case("empty mug", expect_escalate=True, scene="the mug is empty"),  # stub: True  -> tp
        Case("typing", expect_escalate=False, scene="person typing"),  # stub: False -> tn
        Case("walking", expect_escalate=True, scene="person walking by"),  # stub: False -> fn
    ]

    def mem():
        return Memory(str(tmp_path / "e.jsonl"), str(tmp_path / "p.md"))

    metrics, rows = asyncio.run(evaluate(cases, Glance(StubBackend("glance")), mem))
    assert (metrics["tp"], metrics["fn"], metrics["fp"], metrics["tn"]) == (1, 1, 0, 1)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert round(metrics["accuracy"], 3) == round(2 / 3, 3)
