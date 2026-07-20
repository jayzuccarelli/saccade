"""Turn captured audio into text, on this machine.

    uv pip install -e '.[stt]'
    SACCADE_SENSOR=mic SACCADE_STT=whisper

Hearing used to be a property of one vendor: Gemini was the only backend that
accepted audio, so putting a microphone in the room meant uploading the room.
Transcribing here inverts that: the audio never leaves, and what reaches the
model is a line of text that *any* backend can read, local ones included.

faster-whisper is the engine: MIT, `pip install faster-whisper`, and it bundles
FFmpeg through PyAV so Windows needs no system install. It's imported normally
rather than shelled out to (unlike Piper, which is GPL); it lives behind the
`stt` extra so the base harness stays stdlib-only.

The model is loaded once and reused: it's a few hundred MB and reloading it per
clip would cost more than the transcription.
"""

from __future__ import annotations

import asyncio
from typing import Any

# `base` is the accuracy/speed knee for short ambient clips on CPU; `tiny` is
# noticeably worse at exactly the thing that matters here (catching a sentence
# said across a room). Override with SACCADE_STT_MODEL.
DEFAULT_MODEL = "base"


class Transcriber:
    """Lazily-loaded local speech-to-text. `transcribe` returns "" for silence,
    which is most clips: an ambient mic is mostly nothing happening."""

    def __init__(self, model: str = DEFAULT_MODEL, compute_type: str = "int8") -> None:
        self.model_name = model
        self.compute_type = compute_type
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_name, device="cpu", compute_type=self.compute_type)
        return self._model

    def _run(self, wav: bytes) -> str:
        import io

        segments, _ = self._load().transcribe(io.BytesIO(wav), beam_size=1)
        return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(self, wav: bytes) -> str:
        """Transcribe WAV bytes off-thread: the model call is CPU-bound and would
        otherwise stall the capture loop it's feeding."""
        return await asyncio.to_thread(self._run, wav)
