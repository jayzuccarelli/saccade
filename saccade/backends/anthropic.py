"""Anthropic (Claude) backend. Structured output via FORCED TOOL USE.

Claude has no json-schema response param, so the canonical way to guarantee a
shape is to expose a single tool whose input_schema *is* the schema and force
it with tool_choice. The harness never sees this — it just gets JSON back.

Lazy import so the harness runs without the SDK. UNTESTED until a key is wired.

    pip install anthropic   # then set ANTHROPIC_API_KEY
"""

from __future__ import annotations

import base64
import json
import os

from saccade.schema import Frame


class AnthropicBackend:
    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 1024):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self._client = None

    def _client_lazy(self):
        # Reuse one client — it owns a connection pool meant to live across calls.
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def complete(
        self, prompt: str, frames: list[Frame], schema: dict | None = None
    ) -> str:
        client = self._client_lazy()
        blocks: list = [{"type": "text", "text": prompt}]
        for f in frames:
            if f.image:
                b64 = base64.b64encode(f.image).decode()
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": f.mime, "data": b64},
                    }
                )
        messages = [{"role": "user", "content": blocks}]

        if schema:
            tool = {
                "name": "emit",
                "description": "Return the structured result.",
                "input_schema": schema,
            }
            resp = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages,
                tools=[tool],
                tool_choice={"type": "tool", "name": "emit"},
            )
            for block in resp.content:
                if block.type == "tool_use":
                    return json.dumps(block.input)
            return ""

        resp = await client.messages.create(
            model=self.model, max_tokens=self.max_tokens, messages=messages
        )
        return "".join(b.text for b in resp.content if b.type == "text")
