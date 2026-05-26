"""Reolink (or any RTSP/ONVIF camera) as a Sensor.

Pulls frames over RTSP and emits them as JPEG-encoded Frames at `fps`. cv2 is
imported lazily so the rest of the harness runs without opencv installed.

    rtsp://<user>:<pass>@<camera-ip>:554/h264Preview_01_main
"""

from __future__ import annotations

import asyncio
import time

from saccade.schema import Frame


class ReolinkSensor:
    def __init__(self, rtsp_url: str, fps: float = 1.0, max_dim: int = 0):
        self.rtsp_url = rtsp_url
        self.interval = 1.0 / fps
        self.max_dim = max_dim

    async def stream(self):
        import cv2  # lazy: pip install opencv-python-headless

        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            raise RuntimeError(f"could not open RTSP stream: {self.rtsp_url}")
        try:
            while True:
                # cap.read() blocks; fine at 1Hz. Move to an executor if fps climbs.
                ok, frame = cap.read()
                if not ok:
                    await asyncio.sleep(self.interval)
                    continue
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    data = buf.tobytes()
                    if self.max_dim:
                        from saccade.imageutil import downscale_jpeg

                        data = downscale_jpeg(data, self.max_dim)
                    yield Frame(ts=time.time(), image=data, mime="image/jpeg")
                await asyncio.sleep(self.interval)
        finally:
            cap.release()
