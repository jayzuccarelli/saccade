"""Synthesize speech with Gemini TTS and write it to a wav.

The hosted upgrade over the default Piper speaker: better voices, at the cost of
an API key and a network round trip per utterance. Worth it when you care how it
sounds; overkill for "someone's at the door".

Playback (device or command) is shared with the other speakers — see
`_playback.py`. The SDK import is lazy so the harness has no hard dependency on
google-genai.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path
from typing import Any

from saccade.speakers._playback import play

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
        self._client: Any = None

    def _client_lazy(self) -> Any:
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

    async def say(self, text: str) -> None:
        path = await self.synthesize(text)
        print(f"\n    \033[1m\033[96m💬  {text}\033[0m   🔊 {path}\n")
        await play(path, self.play_cmd, self.out_index)
