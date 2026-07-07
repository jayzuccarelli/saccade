"""Gemini backend. Defaults per tier live in DEFAULT_MODELS in __main__.py.

Lazy import of google-genai so the harness runs without it. Validated live end
to end — RTSP camera → Glance @ 1Hz → Focus → TTS.

    pip install google-genai   # then set GEMINI_API_KEY
"""

from __future__ import annotations

import os
from typing import Any

from saccade.schema import Frame, JsonSchema


class GeminiBackend:
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client: Any = None

    def _client_lazy(self) -> Any:
        # Built once and reused — Glance calls this ~1/sec, and a fresh Client per
        # call rebuilds the connection pool every tick.
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def complete(
        self, prompt: str, frames: list[Frame], schema: JsonSchema | None = None
    ) -> str:
        from google.genai import types

        client = self._client_lazy()
        contents: list[Any] = [prompt]
        for f in frames:
            if f.image:
                contents.append(types.Part.from_bytes(data=f.image, mime_type=f.mime))
            if f.audio:
                contents.append(types.Part.from_bytes(data=f.audio, mime_type=f.audio_mime))
        config = None
        if schema:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=schema,
            )
        resp = await client.aio.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        return resp.text or ""
