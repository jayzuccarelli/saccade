"""Verify each backend translates one neutral schema into its provider's native
mechanism. SDKs are mocked, so these check request construction, not the network.
Skipped automatically where a provider SDK isn't installed (e.g. lean CI)."""

import asyncio
import base64
import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saccade.schema import PERCEPT_SCHEMA, Frame

_IMG = Frame(ts=0.0, image=b"\xff\xd8jpegbytes", mime="image/jpeg")
_AUD = Frame(ts=0.0, audio=b"RIFFwavbytes", audio_mime="audio/wav")


def test_gemini_passes_json_schema_and_image():
    pytest.importorskip("google.genai")
    from saccade.backends.gemini import GeminiBackend

    resp = MagicMock(
        text='{"summary":"x","tags":[],"salience":0.1,"escalate":false,"state_delta":""}'
    )
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=resp)
    with patch("google.genai.Client", return_value=client):
        out = asyncio.run(
            GeminiBackend("m", api_key="k").complete("hi", [_IMG], schema=PERCEPT_SCHEMA)
        )
    assert out == resp.text
    kwargs = client.aio.models.generate_content.call_args.kwargs
    assert kwargs["config"].response_json_schema == PERCEPT_SCHEMA
    assert len(kwargs["contents"]) == 2  # prompt + one image part


def test_gemini_sends_audio_part():
    pytest.importorskip("google.genai")
    from saccade.backends.gemini import GeminiBackend

    resp = MagicMock(text="{}")
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=resp)
    with patch("google.genai.Client", return_value=client):
        asyncio.run(GeminiBackend("m", api_key="k").complete("hi", [_AUD]))
    contents = client.aio.models.generate_content.call_args.kwargs["contents"]
    assert len(contents) == 2  # prompt + one audio part (mic-only frame)


def test_openai_uses_response_format_json_schema():
    pytest.importorskip("openai")
    from saccade.backends.openai import OpenAIBackend

    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content='{"ok":true}'))]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=resp)
    with patch("openai.AsyncOpenAI", return_value=client):
        out = asyncio.run(
            OpenAIBackend("m", api_key="k").complete("hi", [_IMG], schema=PERCEPT_SCHEMA)
        )
    assert out == '{"ok":true}'
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["schema"] == PERCEPT_SCHEMA
    assert kwargs["response_format"]["json_schema"]["strict"] is True


def test_ollama_passes_json_schema_and_base64_image():
    from saccade.backends.ollama import OllamaBackend

    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data)
        return io.BytesIO(json.dumps({"message": {"content": '{"ok":true}'}}).encode())

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        out = asyncio.run(
            OllamaBackend("gemma3:4b", host="http://localhost:11434").complete(
                "hi", [_IMG], schema=PERCEPT_SCHEMA
            )
        )
    assert out == '{"ok":true}'
    assert captured["url"] == "http://localhost:11434/api/chat"
    body = captured["body"]
    assert body["model"] == "gemma3:4b"
    assert body["stream"] is False
    assert body["format"] == PERCEPT_SCHEMA
    msg = body["messages"][0]
    assert msg["content"] == "hi"
    assert msg["images"] == [base64.b64encode(_IMG.image).decode()]


def test_anthropic_forces_tool_use_and_returns_input():
    pytest.importorskip("anthropic")
    from saccade.backends.anthropic import AnthropicBackend

    tool_block = MagicMock(type="tool_use", input={"summary": "x", "escalate": True})
    resp = MagicMock(content=[tool_block])
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=resp)
    with patch("anthropic.AsyncAnthropic", return_value=client):
        out = asyncio.run(
            AnthropicBackend("m", api_key="k").complete("hi", [_IMG], schema=PERCEPT_SCHEMA)
        )
    assert json.loads(out) == {"summary": "x", "escalate": True}
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["tools"][0]["input_schema"] == PERCEPT_SCHEMA
    assert kwargs["tool_choice"] == {"type": "tool", "name": "emit"}
