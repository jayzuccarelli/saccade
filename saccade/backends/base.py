"""A Backend is a swappable model. Both Glance and Focus speak only this.

The whole point: vendor SDKs live ONLY in backend implementations. Nothing else
in the harness imports openai/google/etc. Swap a model = swap a Backend.
"""

from __future__ import annotations

from typing import Protocol

from saccade.schema import Frame


class Backend(Protocol):
    async def complete(self, prompt: str, frames: list[Frame], schema: dict | None = None) -> str:
        """Run multimodal inference. Return the model's raw text.

        If `schema` (a JSON Schema dict) is given, the backend enforces structured
        output using its provider's native mechanism and returns a JSON string.
        The caller (Glance/Focus) owns the schema; the backend owns the translation.
        """
        ...
