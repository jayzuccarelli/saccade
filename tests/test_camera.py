"""Reading the frame that is true *now*.

OpenCV returns the oldest queued frame and CAP_PROP_BUFFERSIZE is honored only
by V4L2, so anything reading slower than the camera produces falls further
behind on every read. An agent describing a frame from a minute ago is
indistinguishable from one making things up.
"""

import time

from saccade.sensors._camera import _MAX_DRAIN, drain_to_latest


class _Cap:
    """A driver holding `queued` frames, then live ones at `live_delay`."""

    def __init__(self, queued, live_delay=0.02):
        self.queued, self.live_delay, self.grabs = queued, live_delay, 0

    def grab(self):
        self.grabs += 1
        if self.queued > 0:
            self.queued -= 1  # already in memory: returns immediately
            return True
        time.sleep(self.live_delay)  # waiting on the sensor
        return True


def test_a_backlog_is_thrown_away():
    """The bug: cover the lens and the model keeps describing a minute-old frame."""
    cap = _Cap(queued=40)
    drain_to_latest(cap)
    assert cap.queued == 0
    assert cap.grabs == 41  # the 40 stale ones, then one that had to wait


def test_an_empty_queue_costs_one_grab():
    """Nothing buffered is the normal case; it mustn't turn into a stall."""
    cap = _Cap(queued=0)
    drain_to_latest(cap)
    assert cap.grabs == 1


def test_a_camera_that_never_blocks_does_not_spin_forever():
    """A file or a dead device can answer instantly forever."""
    cap = _Cap(queued=10**6, live_delay=0.0)
    drain_to_latest(cap)
    assert cap.grabs == _MAX_DRAIN


def test_a_failing_grab_stops_the_drain():
    class _Dead:
        grabs = 0

        def grab(self):
            self.grabs += 1
            return False

    cap = _Dead()
    drain_to_latest(cap)
    assert cap.grabs == 1
