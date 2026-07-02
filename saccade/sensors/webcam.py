"""Local webcam as a Sensor — Mac (AVFoundation), Linux (V4L2), Windows (MSMF).

    SACCADE_SENSOR=webcam SACCADE_WEBCAM_INDEX=0 python -m saccade

cv2 is imported lazily so the rest of the harness runs without opencv installed.
"""

from __future__ import annotations

import asyncio
import sys
import time

from saccade.schema import Frame


class WebcamSensor:
    def __init__(self, index: int = 0, fps: float = 1.0):
        self.index = index
        self.interval = 1.0 / fps

    async def stream(self):
        import cv2  # lazy: pip install opencv-python-headless

        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            hint = ""
            if sys.platform == "darwin":
                hint = (
                    " — on macOS, grant Camera access to your terminal app"
                    " (System Settings > Privacy & Security > Camera), then rerun"
                )
            raise RuntimeError(f"could not open webcam index {self.index}{hint}")
        # Shallow driver buffer so cap.read() leans toward the newest frame. Only
        # V4L2 honors this (AVFoundation/MSMF silently ignore it) — at 1fps the
        # residual staleness is one capture interval, fine for ambient watching.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fails = 0
        try:
            while True:
                await asyncio.sleep(self.interval)
                ok, frame = cap.read()
                if not ok:
                    # Unplugged USB cam or macOS sleep/wake killing the session:
                    # read() then fails forever. After a few ticks, reopen —
                    # otherwise the loop re-analyzes nothing, silently, at 1Hz.
                    fails += 1
                    if fails >= 3:
                        cap.release()
                        cap.open(self.index)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        fails = 0
                    continue
                fails = 0
                ok2, buf = cv2.imencode(".jpg", frame)
                if ok2:
                    yield Frame(ts=time.time(), image=buf.tobytes(), mime="image/jpeg")
        finally:
            cap.release()
