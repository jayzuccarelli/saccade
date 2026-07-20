"""Several sensors at once. The merge is the contract: frames from every input
arrive interleaved, and a failing input is loud rather than silently absent."""

import asyncio

import pytest

from saccade.schema import Frame
from saccade.sensors.multi import MultiSensor


class Fake:
    """Yields n frames tagged with its own name, pausing so the two interleave."""

    def __init__(self, name, n=3, delay=0.001):
        self.name, self.n, self.delay = name, n, delay

    async def stream(self):
        for i in range(self.n):
            await asyncio.sleep(self.delay)
            yield Frame(ts=float(i), meta={"from": self.name})


class Boom:
    async def stream(self):
        yield Frame(ts=0.0, meta={"from": "boom"})
        raise RuntimeError("camera unplugged")


async def _drain(sensor):
    return [f async for f in sensor.stream()]


def test_frames_from_every_sensor_come_through():
    frames = asyncio.run(_drain(MultiSensor([Fake("screen"), Fake("mic")])))
    assert len(frames) == 6
    assert {f.meta["from"] for f in frames} == {"screen", "mic"}


def test_streams_interleave_rather_than_running_in_series():
    """The point of merging: the mic isn't waiting for the screen to finish. If
    these ran one after the other the first three would all be 'screen'."""
    frames = asyncio.run(_drain(MultiSensor([Fake("screen", delay=0.001), Fake("mic", delay=0.001)])))
    assert len({f.meta["from"] for f in frames[:3]}) == 2


def test_a_failing_sensor_takes_the_stream_down():
    """Half-blind while reporting healthy is the worse failure for something you
    stop paying attention to. The loop re-raises this and says which input died."""
    with pytest.raises(RuntimeError, match="camera unplugged"):
        asyncio.run(_drain(MultiSensor([Boom(), Fake("mic", n=50, delay=0.01)])))


def test_one_sensor_is_still_valid():
    assert len(asyncio.run(_drain(MultiSensor([Fake("solo", n=2)])))) == 2


def test_no_sensors_is_a_config_error_not_a_silent_no_op():
    with pytest.raises(ValueError):
        MultiSensor([])
