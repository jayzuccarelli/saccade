"""OpenAI backend. Structured output via response_format json_schema (strict).

Lazy import so the harness runs without the SDK. UNTESTED until a key is wired.

    pip install openai   # then set OPENAI_API_KEY
"""

from __future__ import annotations

import base64
import os

from saccade.schema import Frame


class OpenAIBackend:
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    async def complete(
        self, prompt: str, frames: list[Frame], schema: dict | None = None
    ) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)
        content: list = [{"type": "text", "text": prompt}]
        for f in frames:
            if f.image:
                b64 = base64.b64encode(f.image).decode()
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{f.mime};base64,{b64}"}}
                )
        kwargs: dict = {"model": self.model, "messages": [{"role": "user", "content": content}]}
        if schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "strict": True, "schema": schema},
            }
        resp = await client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
