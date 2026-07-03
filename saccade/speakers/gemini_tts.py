"""Synthesize speech with Gemini TTS and write it to a wav.

The box saccade watches from may have no audio out, so by default we just write
the clip and print where it landed. Two ways to actually play it:

  - `play_cmd` (SACCADE_PLAY_CMD): a command taking the file path — `aplay`,
    `afplay`, or a wrapper that pushes it to a speaker / the camera. Uses the OS
    default output device.
  - `out_index` (SACCADE_AUDIO_OUT_INDEX): play to a specific device by index
    (the numbers `saccade devices` lists) via sounddevice — the symmetric twin
    of picking a mic. Wins over play_cmd when set.

The SDK/sounddevice imports are lazy so the harness has no hard dependency on
google-genai or PortAudio.
"""

from __future__ import annotations

import asyncio
import time
import wave
from pathlib import Path

# Gemini TTS returns 16-bit signed PCM, mono, 24 kHz (mime audio/L16;rate=24000).
_RATE = 24000
_SAMPLE_WIDTH = 2
_CHANNELS = 1


class GeminiTTSSpeaker:
    def __init__(
        self,
        model: str,
        voice: str,
        out_dir: str,
        play_cmd: str = "",
        out_index: int = -1,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.voice = voice
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.play_cmd = play_cmd
        self.out_index = out_index  # -1 = OS default (via play_cmd); >=0 = that device
        self._api_key = api_key
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key) if self._api_key else genai.Client()
        return self._client

    async def synthesize(self, text: str) -> Path:
        """Synthesize `text` to a wav and return its path. Reused by speakers that
        play the audio elsewhere (e.g. HomeAssistantSpeaker)."""
        from google.genai import types

        client = self._client_lazy()
        resp = await client.aio.models.generate_content(
            model=self.model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice)
                    )
                ),
            ),
        )
        pcm = resp.candidates[0].content.parts[0].inline_data.data
        path = self.out_dir / f"utt_{int(time.time() * 1000)}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(_CHANNELS)
            w.setsampwidth(_SAMPLE_WIDTH)
            w.setframerate(_RATE)
            w.writeframes(pcm)
        return path

    def _play_to_device(self, path: Path) -> None:
        """Blocking playback of a wav to a specific output device — run off-thread."""
        import numpy as np
        import sounddevice as sd

        with wave.open(str(path), "rb") as w:
            rate, channels = w.getframerate(), w.getnchannels()
            pcm = w.readframes(w.getnframes())
        data = np.frombuffer(pcm, dtype=np.int16)
        if channels > 1:
            data = data.reshape(-1, channels)
        sd.play(data, samplerate=rate, device=self.out_index)
        sd.wait()

    async def say(self, text: str) -> None:
        path = await self.synthesize(text)
        print(f"\n    \033[1m\033[96m💬  {text}\033[0m   🔊 {path}\n")
        if self.out_index is not None and self.out_index >= 0:
            await asyncio.to_thread(self._play_to_device, path)  # pick the speaker
        elif self.play_cmd:
            proc = await asyncio.create_subprocess_exec(*self.play_cmd.split(), str(path))
            await proc.wait()
