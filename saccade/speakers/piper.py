"""Synthesize speech with Piper: local, offline, no API key, same on every OS.

The default way saccade talks. An ambient agent says a handful of short lines a
day, and needing a hosted key and a network round trip to make any sound at all
is the wrong shape for something running on a box in your kitchen.

Piper is run as a subprocess, never imported, and that is deliberate:
piper-tts is GPL-3.0-or-later while saccade is MIT. Importing it into our
process is linkage; running it as a separate program is aggregation, which keeps
the two licenses at arm's length. It also keeps onnxruntime out of saccade's
dependency tree. For the same reason piper-tts is *not* in our extras — the user
installs it themselves and we call whatever they installed:

    pip install piper-tts
    python -m piper.download_voices en_US-lessac-medium

Voices are per-language and per-quality; see `python -m piper.download_voices`
with no argument for the list.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from saccade.speakers._playback import play


class PiperError(RuntimeError):
    """Piper couldn't speak, with the command that fixes it. The loop prints this
    verbatim, so it has to read like an instruction rather than a stack trace."""


class PiperSpeaker:
    def __init__(
        self,
        voice: str,
        out_dir: str,
        play_cmd: str = "",
        out_index: int = -1,
        data_dir: str = "",
    ) -> None:
        self.voice = voice
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.play_cmd = play_cmd
        self.out_index = out_index  # -1 = OS default (via play_cmd); >=0 = that device
        self.data_dir = data_dir  # where voices live; blank = piper's default

    async def synthesize(self, text: str) -> Path:
        """Synthesize `text` to a wav and return its path. Same shape as
        GeminiTTSSpeaker.synthesize, so HomeAssistantSpeaker takes either."""
        path = self.out_dir / f"utt_{int(time.time() * 1000)}.wav"
        # sys.executable, not a bare `piper`: the venv that's running saccade is
        # the one that has piper installed, and on Windows there may be no
        # console script on PATH at all.
        cmd = [sys.executable, "-m", "piper", "-m", self.voice, "-f", str(path)]
        if self.data_dir:
            cmd += ["--data-dir", self.data_dir]
        cmd += ["--", text]  # `--` so a line starting with "-" isn't read as a flag
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise PiperError(self._diagnose(err.decode(errors="replace")))
        return path

    def _diagnose(self, stderr: str) -> str:
        """Turn Piper's failure into the command that fixes it. The two that
        actually happen are 'never installed it' and 'never downloaded a voice',
        and both are one command away from working."""
        # Name the interpreter, not a bare `pip`/`python`. A Mac has no `python`
        # on PATH, and installing with the wrong pip lands piper somewhere the
        # running saccade can't import from — which is how you get piper
        # installed and "No module named piper" in the same terminal. Both
        # installers are offered because a uv-made venv ships without pip.
        if "No module named" in stderr:
            return (
                f"Piper isn't installed in {sys.executable} — "
                f"`uv pip install piper-tts`, or `pip install piper-tts` in an activated venv"
            )
        if self.voice in stderr or "voice" in stderr.lower():
            return (
                f"Piper has no voice {self.voice!r} — download it: "
                f"{sys.executable} -m piper.download_voices {self.voice}"
            )
        return f"Piper failed: {stderr.strip().splitlines()[-1] if stderr.strip() else 'no output'}"

    async def say(self, text: str) -> None:
        path = await self.synthesize(text)
        print(f"\n    \033[1m\033[96m💬  {text}\033[0m   🔊 {path}\n")
        await play(path, self.play_cmd, self.out_index)
