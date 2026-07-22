"""Shared camera helpers: reading the frame that's true *now*.

OpenCV hands back the oldest frame the driver has queued, not the newest, and
`CAP_PROP_BUFFERSIZE` is honored only by V4L2 (AVFoundation and MSMF ignore it
silently). Anything that reads slower than the camera produces therefore falls
further behind on every read, and an ambient agent describing a frame from a
minute ago is indistinguishable from one making things up.
"""

from __future__ import annotations

import time
from typing import Any

# A queued frame is already in memory and comes back immediately; a live one has
# to wait for the sensor. Anything slower than this was worth waiting for, so it's
# the newest and draining stops.
_QUEUED_MAX_S = 0.005

# Don't drain forever if every grab looks instant (a file, a dead device).
_MAX_DRAIN = 300


def drain_to_latest(cap: Any) -> None:
    """Discard frames the driver queued while the caller was busy."""
    for _ in range(_MAX_DRAIN):
        start = time.perf_counter()
        if not cap.grab():
            return
        if time.perf_counter() - start > _QUEUED_MAX_S:
            return


def latest_jpeg(cap: Any) -> bytes | None:
    """The current view, JPEG-encoded, or None if the camera gave nothing."""
    import cv2

    drain_to_latest(cap)
    ok, frame = cap.read()
    if not ok:
        return None
    ok2, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok2 else None
