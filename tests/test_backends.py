"""Provider dispatch + schema sanity. SDK-free (cloud SDKs import lazily)."""

import asyncio
import json

from saccade.__main__ import DEFAULT_MODELS, make_backend
from saccade.backends.gemini import GeminiBackend
from saccade.backends.ollama import OllamaBackend
from saccade.backends.stub import StubBackend
from saccade.config import Config
from saccade.schema import DECISION_SCHEMA, PERCEPT_SCHEMA, Frame


def test_make_backend_dispatch_and_default_models():
    c = Config()
    assert isinstance(make_backend("stub", "glance", c), StubBackend)
    g = make_backend("gemini", "glance", c)
    assert isinstance(g, GeminiBackend)
    assert g.model == "gemini-3.1-flash-lite"
    assert make_backend("gemini", "focus", c).model == "gemini-3.5-flash"
    o = make_backend("ollama", "glance", c)
    assert isinstance(o, OllamaBackend)
    assert o.model == "gemma3:4b"
    assert make_backend("ollama", "focus", c).model == "gemma3:12b"


def test_default_models_cover_every_provider_and_tier():
    for provider in ("gemini", "openai", "anthropic", "ollama"):
        for role in ("glance", "focus"):
            assert (provider, role) in DEFAULT_MODELS


def test_schemas_are_strict_and_wellformed():
    for s in (PERCEPT_SCHEMA, DECISION_SCHEMA):
        assert s["type"] == "object"
        assert s["additionalProperties"] is False
        # strict structured output requires every property to be required
        assert set(s["required"]) == set(s["properties"])


def test_stub_output_conforms_to_percept_schema():
    raw = asyncio.run(
        StubBackend("glance").complete(
            "", [Frame(ts=0.0, meta={"scene": "the mug is empty"})], schema=PERCEPT_SCHEMA
        )
    )
    d = json.loads(raw)
    assert set(d) >= set(PERCEPT_SCHEMA["properties"])
    assert d["escalate"] is True  # "empty" is a salient cue in the stub
