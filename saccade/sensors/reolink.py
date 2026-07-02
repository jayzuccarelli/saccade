"""Reolink (or any RTSP/ONVIF camera) as a Sensor.

Pulls frames over RTSP and emits them as JPEG-encoded Frames at `fps`. cv2 is
imported lazily so the rest of the harness runs without opencv installed.

    rtsp://<user>:<pass>@<camera-ip>:554/h264Preview_01_main
"""

from __future__ import annotations

import asyncio
import re
import threading
import time

from saccade.schema import Frame


class ReolinkSensor:
    def __init__(self, rtsp_url: str, fps: float = 1.0):
        self.rtsp_url = rtsp_url
        self.interval = 1.0 / fps

    async def stream(self):
        import cv2  # lazy: pip install opencv-python-headless

        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            # Redact userinfo — this message lands in terminals and bug reports.
            redacted = re.sub(r"//[^@/]+@", "//***@", self.rtsp_url)
            raise RuntimeError(f"could not open RTSP stream: {redacted}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # A reader thread consumes frames as fast as the camera sends them (~15-20fps)
        # and keeps only the newest. Without it, reading one frame per `interval` from
        # an undrained buffer returns ever-older frames — the feed falls minutes behind
        # real-time, so the agent reacts to the past. Draining keeps us on the present.
        latest: dict = {"frame": None}
        stop = threading.Event()

        def _reader() -> None:
            fails = 0
            while not stop.is_set():
                ok, f = cap.read()
                if ok:
                    latest["frame"] = f
                    fails = 0
                    continue
                # A 24/7 camera drops occasionally. After ~2s of failed reads assume
                # the stream died and reopen it — otherwise we'd silently serve the
                # last frame forever (the stale-feed bug, one connection drop later).
                fails += 1
                if fails >= 40:
                    cap.release()
                    cap.open(self.rtsp_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    fails = 0
                time.sleep(0.05)

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()
        try:
            while True:
                await asyncio.sleep(self.interval)
                frame = latest["frame"]
                if frame is None:
                    continue
                # full resolution — Glance downscales its own input; Focus wants detail
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    yield Frame(ts=time.time(), image=buf.tobytes(), mime="image/jpeg")
        finally:
            stop.set()
            # Let the reader leave cap.read() before releasing — releasing under
            # a blocked read is a use-after-free in OpenCV's FFmpeg backend.
            reader.join(timeout=2)
            cap.release()
