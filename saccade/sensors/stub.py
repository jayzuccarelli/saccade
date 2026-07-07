"""A scripted sensor so the whole loop runs with no camera and no API key.

Each Frame carries a text `scene` in meta describing what's "visible" that
second. The stub backend turns that into a Percept. Together they let you watch
glance -> focus -> speak end-to-end before any hardware or key exists.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import AsyncIterator

from saccade.schema import Frame

SCRIPT = [
    "person sitting at a desk, typing on a laptop",
    "person still typing, focused on the screen",
    "person stands up and stretches",
    "person looking around the room, seems to be searching for something",
    "person holding a coffee mug, the mug is empty",
    "person sits back down and resumes typing",
]


class StubSensor:
    def __init__(self, fps: float = 1.0, loop: bool = False):
        self.interval = 1.0 / fps
        self.loop = loop

    async def stream(self) -> AsyncIterator[Frame]:
        seq = itertools.cycle(SCRIPT) if self.loop else iter(SCRIPT)
        for scene in seq:
            yield Frame(ts=time.time(), image=None, meta={"scene": scene})
            await asyncio.sleep(self.interval)
