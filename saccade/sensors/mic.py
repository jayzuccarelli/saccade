"""Local microphone as a Sensor: captures a short audio clip each tick.

    SACCADE_SENSOR=mic SACCADE_MIC_INDEX=0 python -m saccade

Each tick records one glance interval of mono 16kHz audio and yields it as a WAV
Frame, the audio twin of WebcamSensor's JPEG frames. Recording itself paces the
stream (a 1s clip takes ~1s), so there's no separate sleep.

Needs the audio extra (sounddevice + PortAudio). Raw audio only reaches a backend
that accepts it: Gemini today; Anthropic/Ollama are vision-only.

With a `transcriber` (SACCADE_STT=whisper) the clip is turned into text here
instead, and the audio is **not** attached to the Frame. That's the point: the
room is never uploaded, and the transcript reaches any backend, local included.
Sending both would hand the audio to the vendor anyway and give up the reason to
transcribe locally at all.

sounddevice is imported lazily so the rest of the harness runs without it.
"""

from __future__ import annotations

import asyncio
import io
import time
import wave
from collections.abc import AsyncIterator
from typing import Any

from saccade.schema import Frame

SAMPLE_RATE = 16000  # 16kHz mono: plenty for speech, keeps the payload small


def require_audio() -> None:
    """Fail early with a clear message if the audio stack isn't installed;
    beats a mid-loop crash. Shared by every sensor that records."""
    try:
        import sounddevice  # noqa: F401
    except ImportError as e:
        raise RuntimeError("microphone needs the audio extra: uv pip install -e '.[audio]'") from e
    except OSError as e:
        # sounddevice imports but PortAudio (the C lib) is missing: bundled in
        # the Mac/Windows wheels, apt/brew territory on Linux.
        raise RuntimeError(
            "microphone needs PortAudio: `sudo apt install libportaudio2` (Linux)"
        ) from e


def record_pcm(seconds: float, sample_rate: int, device: int | None) -> bytes:
    """Blocking capture of one mono clip; run off-thread. Returns raw int16 PCM."""
    import sounddevice as sd  # lazy: pip install sounddevice

    n = int(seconds * sample_rate)
    audio = sd.rec(n, samplerate=sample_rate, channels=1, dtype="int16", device=device)
    sd.wait()
    pcm: bytes = audio.tobytes()
    return pcm


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw int16 mono PCM in a WAV container so the model gets a
    self-describing clip. Reused by the mic and combined AV sensors."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # int16 = 2 bytes/sample
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class MicSensor:
    def __init__(
        self,
        index: int | None = None,
        fps: float = 1.0,
        sample_rate: int = SAMPLE_RATE,
        transcriber: Any | None = None,
        seconds: float = 0.0,
    ):
        self.index = index  # None = system default input device
        # Not 1/fps: that made a one-second clip at the default rate, which is
        # too short for a sentence and usually transcribes to nothing.
        self.seconds = seconds if seconds > 0 else 1.0 / fps
        self.sample_rate = sample_rate
        self.transcriber = transcriber  # anything with `async transcribe(wav) -> str`

    def _record(self) -> bytes:
        return record_pcm(self.seconds, self.sample_rate, self.index)

    def _wav(self, pcm: bytes) -> bytes:
        return wav_bytes(pcm, self.sample_rate)

    async def stream(self) -> AsyncIterator[Frame]:
        require_audio()
        while True:
            pcm = await asyncio.to_thread(self._record)
            wav = self._wav(pcm)
            if self.transcriber is None:
                yield Frame(ts=time.time(), audio=wav, audio_mime="audio/wav")
                continue
            # Transcribed here, so the audio stays here. Most clips are silence,
            # which comes back as "", still a Frame, so the loop keeps its
            # cadence and Glance sees "nothing was said" rather than nothing.
            yield Frame(ts=time.time(), text=await self.transcriber.transcribe(wav))
