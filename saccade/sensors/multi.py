"""Several sensors at once, merged into one stream.

    SACCADE_SENSOR=screen,mic python -m saccade

Each sub-sensor runs at its own pace and its frames are yielded as they arrive,
interleaved. That's the difference from `av`, which *fuses* a camera grab and a
mic clip into one Frame carrying both: use `av` when the image and the sound
have to describe the same instant, and this when the inputs are independent
(watch the screen, listen to the room).

A sub-sensor that fails takes the whole stream down rather than quietly leaving
you with half the inputs you asked for. Half-blind while reporting healthy is
the worse failure for something you stop watching.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from saccade.schema import Frame

if TYPE_CHECKING:
    from saccade.sensors.base import Sensor

# Sentinel: one sub-sensor's stream ended.
_DONE = object()


class MultiSensor:
    def __init__(self, sensors: list[Sensor]) -> None:
        if not sensors:
            raise ValueError("MultiSensor needs at least one sensor")
        self.sensors = sensors

    async def stream(self) -> AsyncIterator[Frame]:
        queue: asyncio.Queue[object] = asyncio.Queue()

        async def pump(sensor: Sensor) -> None:
            try:
                async for frame in sensor.stream():
                    await queue.put(frame)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — forwarded to the consumer below
                await queue.put(e)
            finally:
                await queue.put(_DONE)

        tasks = [asyncio.create_task(pump(s)) for s in self.sensors]
        live = len(tasks)
        try:
            while live:
                item = await queue.get()
                if item is _DONE:
                    live -= 1
                elif isinstance(item, Exception):
                    raise item
                else:
                    assert isinstance(item, Frame)
                    yield item
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
