"""Webcam + microphone fused into one Sensor — a glance that both sees and hears.

    SACCADE_SENSOR=av python -m saccade

Each tick grabs one camera frame and records one glance-interval audio clip for
the same instant, yielding a single Frame carrying both. The recording (which
blocks for the interval) paces the stream and is run off-thread; the camera
grab is taken at the start of that window.

Needs both the camera and audio extras, plus an audio-capable backend (Gemini)
to actually use the sound. cv2 + sounddevice are imported lazily.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator
from typing import Any

from saccade.schema import Frame
from saccade.sensors.mic import SAMPLE_RATE, record_pcm, require_audio, wav_bytes


class AVSensor:
    def __init__(
        self,
        webcam_index: int = 0,
        mic_index: int | None = None,
        fps: float = 1.0,
        sample_rate: int = SAMPLE_RATE,
        transcriber: Any | None = None,
    ):
        self.webcam_index = webcam_index
        self.mic_index = mic_index
        self.seconds = 1.0 / fps  # audio clip length = one glance interval
        self.sample_rate = sample_rate
        self.transcriber = transcriber  # anything with `async transcribe(wav) -> str`

    def _grab_jpeg(self, cap: Any) -> bytes | None:
        import cv2

        ok, frame = cap.read()
        if not ok:
            return None
        ok2, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes() if ok2 else None

    async def stream(self) -> AsyncIterator[Frame]:
        import cv2  # lazy: pip install opencv-python-headless

        require_audio()  # fail early if the mic stack is missing
        cap = cv2.VideoCapture(self.webcam_index)
        if not cap.isOpened():
            hint = ""
            if sys.platform == "darwin":
                hint = (
                    " — on macOS, grant Camera access to your terminal app"
                    " (System Settings > Privacy & Security > Camera), then rerun"
                )
            raise RuntimeError(f"could not open webcam index {self.webcam_index}{hint}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # V4L2-only; harmless elsewhere
        try:
            while True:
                # Frame first (start of the window), then record audio across it.
                image = self._grab_jpeg(cap)
                pcm = await asyncio.to_thread(
                    record_pcm, self.seconds, self.sample_rate, self.mic_index
                )
                wav = wav_bytes(pcm, self.sample_rate)
                if self.transcriber is None:
                    yield Frame(
                        ts=time.time(),
                        image=image,
                        mime="image/jpeg",
                        audio=wav,
                        audio_mime="audio/wav",
                    )
                else:
                    # Transcribed here, so the audio stays here — the picture still
                    # goes to whichever backend, but the room is never uploaded.
                    yield Frame(
                        ts=time.time(),
                        image=image,
                        mime="image/jpeg",
                        text=await self.transcriber.transcribe(wav),
                    )
        finally:
            cap.release()
