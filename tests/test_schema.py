"""Parsing model output must be forgiving — models wrap JSON in prose/fences."""

from saccade.schema import decision_from, percept_from


def test_percept_parses_clean_json():
    p = percept_from(
        '{"summary":"x","salience":0.7,"escalate":true,"tags":["a"],"state_delta":"d"}', 1.0
    )
    assert p.summary == "x"
    assert p.salience == 0.7
    assert p.escalate is True
    assert p.tags == ["a"]


def test_percept_parses_next_glance_cadence():
    p = percept_from('{"summary":"busy","next_glance_s":2.5}', 1.0)
    assert p.next_glance_s == 2.5


def test_percept_next_glance_defaults_to_zero_when_absent():
    # an older/stub model that omits it -> 0, and the loop uses its fixed cadence.
    p = percept_from('{"summary":"x","escalate":false}', 1.0)
    assert p.next_glance_s == 0.0


def test_percept_tolerates_prose_and_fences():
    raw = 'Sure!\n```json\n{"summary":"hi","escalate":false}\n```\n'
    p = percept_from(raw, 1.0)
    assert p.summary == "hi"
    assert p.escalate is False


def test_percept_defaults_on_garbage():
    p = percept_from("not json at all", 2.0)
    assert p.summary == ""
    assert p.escalate is False
    assert p.salience == 0.0
    assert p.ts == 2.0  # ts always comes from the harness, never the model


def test_decision_parses():
    d = decision_from('{"reasoning":"r","speak":true,"message":"hey"}', 3.0)
    assert d.speak is True
    assert d.message == "hey"
