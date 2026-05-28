"""Synthesize speech with Gemini TTS and write it to a wav.

The box saccade watches from may have no audio out, so by default we just write
the clip and print where it landed. Set `play_cmd` (SACCADE_PLAY_CMD) to a
command that takes the file path to actually play it — `aplay`, `afplay`, or a
wrapper that pushes it to a speaker / the camera. The SDK import is lazy so the
harness has no hard dependency on google-genai.
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
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.voice = voice
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.play_cmd = play_cmd
        self._api_key = api_key
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            from google import genai

            self._client = (
                genai.Client(api_key=self._api_key) if self._api_key else genai.Client()
            )
        return self._client

    async def say(self, text: str) -> None:
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

        print(f"\n    \033[1m\033[96m💬  {text}\033[0m   🔊 {path}\n")

        if self.play_cmd:
            proc = await asyncio.create_subprocess_exec(*self.play_cmd.split(), str(path))
            await proc.wait()
