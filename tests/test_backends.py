"""Provider dispatch + schema sanity. SDK-free (cloud SDKs import lazily)."""

import asyncio
import json
from urllib import error as urlerror

import pytest

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


class _Answering:
    """Minimal stand-in for a urlopen context manager that answers."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return b'{"models": [{"name": "gemma3:4b"}]}'


def _refuse_urlopen(*a, **k):
    raise urlerror.URLError("connection refused")


def test_a_stopped_ollama_is_started_not_reported(monkeypatch, capsys):
    """The loop used to print `start it: ollama serve` on every tick. We know the
    command and can run it; printing it at someone unattended is not a fix."""
    from saccade.backends import ollama as mod

    monkeypatch.setattr(mod, "_start_attempted", False)
    monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/local/bin/ollama")
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: None)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(mod.request, "urlopen", lambda *a, **k: _Answering())
    assert mod._start_daemon("http://localhost:11434")
    assert "starting it" in capsys.readouterr().out


def test_a_remote_ollama_is_never_started(monkeypatch):
    """Someone else's machine is down, so we start a daemon on this one: now two
    servers exist and the frames go to the wrong one."""
    from saccade.backends import ollama as mod

    monkeypatch.setattr(mod, "_start_attempted", False)
    monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/local/bin/ollama")
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: pytest.fail("started remotely"))
    assert not mod._start_daemon("http://nas.local:11434")


def test_it_is_tried_once_not_every_tick(monkeypatch):
    """A daemon that dies on startup would otherwise get a fresh one every glance."""
    from saccade.backends import ollama as mod

    spawns = []
    monkeypatch.setattr(mod, "_start_attempted", False)
    monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/local/bin/ollama")
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: spawns.append(1))
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(mod.request, "urlopen", _refuse_urlopen)
    assert not mod._start_daemon("http://localhost:11434")
    assert not mod._start_daemon("http://localhost:11434")
    assert spawns == [1]


def test_the_old_message_still_lands_when_starting_fails(monkeypatch):
    """It has to end somewhere. If we can't start it, say the thing we always said."""
    from saccade.backends import ollama as mod

    monkeypatch.setattr(mod, "_start_attempted", True)  # already tried, didn't take
    b = OllamaBackend("gemma3:4b", host="http://localhost:11434")
    monkeypatch.setattr(mod.request, "urlopen", _refuse_urlopen)
    with pytest.raises(mod.OllamaError, match="isn't reachable"):
        b._post({"model": "gemma3:4b"})
