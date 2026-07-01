"""Local webcam as a Sensor — Mac (AVFoundation), Linux (V4L2), Windows (MSMF).

    SACCADE_SENSOR=webcam SACCADE_WEBCAM_INDEX=0 python -m saccade

cv2 is imported lazily so the rest of the harness runs without opencv installed.
"""

from __future__ import annotations

import asyncio
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
            raise RuntimeError(f"could not open webcam index {self.index}")
        # Keep the driver buffer shallow so cap.read() returns the newest frame,
        # not one queued 500ms ago while the model was thinking.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            while True:
                await asyncio.sleep(self.interval)
                ok, frame = cap.read()
                if not ok:
                    continue
                ok2, buf = cv2.imencode(".jpg", frame)
                if ok2:
                    yield Frame(ts=time.time(), image=buf.tobytes(), mime="image/jpeg")
        finally:
            cap.release()
