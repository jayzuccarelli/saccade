"""Ollama backend — local, private, free. Stdlib only.

Talks to a local Ollama instance (http://localhost:11434 by default; override
with OLLAMA_HOST or SACCADE_OLLAMA_HOST). No new dependencies — urllib + json.

    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull gemma3:4b           # ~3GB, multimodal, fast (Glance default)
    ollama pull gemma3:12b          # smarter (Focus default)

    SACCADE_GLANCE_BACKEND=ollama SACCADE_FOCUS_BACKEND=ollama python -m saccade

Structured output uses Ollama's native `format` field (a JSON Schema), enforced
by the runtime — not prompt-engineered. Vision frames go as base64 in `images`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from urllib import request

from saccade.schema import Frame

_DEFAULT_HOST = "http://localhost:11434"


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

    async def complete(self, prompt: str, frames: list[Frame], schema: dict | None = None) -> str:
        message: dict = {"role": "user", "content": prompt}
        images = [base64.b64encode(f.image).decode() for f in frames if f.image]
        if images:
            message["images"] = images

        body: dict = {"model": self.model, "messages": [message], "stream": False}
        if schema:
            body["format"] = schema

        return await asyncio.to_thread(self._post, body)

    def _post(self, body: dict) -> str:
        req = request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return data.get("message", {}).get("content", "")
