"""A Sensor is any input stream. Camera, mic, screen — all the same shape.

This Protocol is the whole reason the harness is input-agnostic: a new modality
is a new file implementing `stream()`, and nothing in the core changes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from saccade.schema import Frame


class Sensor(Protocol):
    def stream(self) -> AsyncIterator[Frame]:
        """Yield Frames as they arrive, at the sensor's own pace."""
        ...
