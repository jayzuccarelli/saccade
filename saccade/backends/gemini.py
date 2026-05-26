"""Gemini backend. Default models: Glance = 2.5 Flash-Lite, Focus = 3.5 Flash.

Lazy import of google-genai so the harness runs without it. UNTESTED until a key
is wired — verify the async call shape against the installed SDK version.

    pip install google-genai   # then set GEMINI_API_KEY
"""

from __future__ import annotations

import os

from saccade.schema import Frame


class GeminiBackend:
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

    async def complete(
        self, prompt: str, frames: list[Frame], schema: dict | None = None
    ) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        contents: list = [prompt]
        for f in frames:
            if f.image:
                contents.append(types.Part.from_bytes(data=f.image, mime_type=f.mime))
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
