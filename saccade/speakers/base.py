"""A Speaker turns a decision into output. Swappable, like Backend.

The loop only calls `say(text)`. *How* it comes out — printed, synthesized to a
file, played on a speaker, eventually spoken through the camera — is the
Speaker's concern, never the loop's. Swap the output = swap the Speaker.
"""

from __future__ import annotations

from typing import Protocol


class Speaker(Protocol):
    async def say(self, text: str) -> None:
        """Emit one utterance. May do I/O (synthesis, playback)."""
        ...
