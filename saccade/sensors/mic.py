"""Local microphone as a Sensor — captures a short audio clip each tick.

    SACCADE_SENSOR=mic SACCADE_MIC_INDEX=0 python -m saccade

Each tick records one glance interval of mono 16kHz audio and yields it as a WAV
Frame, the audio twin of WebcamSensor's JPEG frames. Recording itself paces the
stream (a 1s clip takes ~1s), so there's no separate sleep.

Needs the audio extra (sounddevice + PortAudio) AND a backend that accepts audio
— Gemini today; Anthropic/Ollama are vision-only, OpenAI needs an audio model.
sounddevice is imported lazily so the rest of the harness runs without it.
"""

from __future__ import annotations

import asyncio
import io
import time
import wave

from saccade.schema import Frame

SAMPLE_RATE = 16000  # 16kHz mono — plenty for speech, keeps the payload small


class MicSensor:
    def __init__(self, index: int | None = None, fps: float = 1.0, sample_rate: int = SAMPLE_RATE):
        self.index = index  # None = system default input device
        self.seconds = 1.0 / fps  # each clip spans one glance interval
        self.sample_rate = sample_rate

    def _record(self) -> bytes:
        """Blocking capture of one clip — run off-thread so the glance loop keeps
        ticking. Returns raw int16 mono PCM."""
        import sounddevice as sd  # lazy: pip install sounddevice

        n = int(self.seconds * self.sample_rate)
        audio = sd.rec(n, samplerate=self.sample_rate, channels=1, dtype="int16", device=self.index)
        sd.wait()
        return audio.tobytes()

    def _wav(self, pcm: bytes) -> bytes:
        """Wrap raw PCM in a WAV container so the model gets a self-describing clip."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # int16 = 2 bytes/sample
            w.setframerate(self.sample_rate)
            w.writeframes(pcm)
        return buf.getvalue()

    async def stream(self):
        try:
            import sounddevice  # noqa: F401  — surface a clear error, not a mid-loop crash
        except ImportError as e:
            raise RuntimeError("microphone needs the audio extra: uv pip install -e '.[audio]'") from e
        except OSError as e:
            # sounddevice imports but PortAudio (the C lib) is missing — bundled in
            # the Mac/Windows wheels, apt/brew territory on Linux.
            raise RuntimeError(
                "microphone needs PortAudio — `sudo apt install libportaudio2` (Linux)"
            ) from e
        while True:
            pcm = await asyncio.to_thread(self._record)
            yield Frame(ts=time.time(), audio=self._wav(pcm), audio_mime="audio/wav")
