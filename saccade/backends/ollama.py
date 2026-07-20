"""Ollama backend: local, private, free. Stdlib only.

Talks to a local Ollama instance (http://localhost:11434 by default; override
with OLLAMA_HOST or SACCADE_OLLAMA_HOST). No new dependencies: urllib + json.

    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull gemma3:4b           # ~3GB, multimodal, fast (Glance default)
    ollama pull gemma3:12b          # smarter (Focus default)

    SACCADE_GLANCE_BACKEND=ollama SACCADE_FOCUS_BACKEND=ollama python -m saccade

Structured output uses Ollama's native `format` field (a JSON Schema), enforced
by the runtime, not prompt-engineered. Vision frames go as base64 in `images`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any
from urllib import error, request

from saccade.schema import Frame, JsonSchema, heard_text

_DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    """Ollama refused, with the command that fixes it. The loop prints the
    message verbatim, so it has to read like an instruction: a bare
    `URLError: [Errno 61] Connection refused` on every tick tells you nothing."""


class OllamaBackend:
    def __init__(self, model: str, host: str | None = None, timeout: float = 120.0):
        self.model = model
        self.host = (
            host
            or os.environ.get("SACCADE_OLLAMA_HOST")
            or os.environ.get("OLLAMA_HOST")
            or _DEFAULT_HOST
        ).rstrip("/")
        self.timeout = timeout

    async def complete(
        self, prompt: str, frames: list[Frame], schema: JsonSchema | None = None
    ) -> str:
        message: dict[str, Any] = {"role": "user", "content": prompt + heard_text(frames)}
        images = [base64.b64encode(f.image).decode() for f in frames if f.image]
        if images:
            message["images"] = images

        body: dict[str, Any] = {"model": self.model, "messages": [message], "stream": False}
        if schema:
            body["format"] = schema

        return await asyncio.to_thread(self._post, body)

    def _post(self, body: dict[str, Any]) -> str:
        req = request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except error.HTTPError as e:  # subclasses URLError; catch it first
            if e.code == 404:
                raise OllamaError(
                    f"Ollama has no model {self.model!r}; pull it: ollama pull {self.model}"
                ) from e
            raise OllamaError(f"Ollama returned HTTP {e.code}") from e
        except error.URLError as e:
            raise OllamaError(
                f"Ollama isn't reachable at {self.host}; start it: ollama serve"
            ) from e
        content: str = data.get("message", {}).get("content", "")
        return content
