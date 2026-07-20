"""Play a wav — to a chosen output device, through a command, or not at all.

Shared by every speaker that synthesizes to a file. The box saccade watches from
may have no audio out at all, so "wrote the clip" is a valid outcome and
playback is the optional half.

  - `out_index` (SACCADE_AUDIO_OUT_INDEX): a specific device by index (the
    numbers `saccade devices` lists), via sounddevice. The symmetric twin of
    picking a mic. Wins over play_cmd when set.
  - `play_cmd` (SACCADE_PLAY_CMD): a command taking the file path — `aplay`,
    `afplay`, or a wrapper that pushes it somewhere. Uses the OS default device.
"""

from __future__ import annotations

import asyncio
import wave
from pathlib import Path


def _to_device(path: Path, out_index: int) -> None:
    """Blocking playback to one specific device — callers run it off-thread.

    sounddevice/numpy are imported here so the audio extra stays optional: a
    speaker that only writes files shouldn't need PortAudio installed."""
    import numpy as np
    import sounddevice as sd

    with wave.open(str(path), "rb") as w:
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = w.readframes(w.getnframes())
    data = np.frombuffer(pcm, dtype=np.int16)
    if channels > 1:
        data = data.reshape(-1, channels)
    sd.play(data, samplerate=rate, device=out_index)
    sd.wait()


# A player that never exits (wedged audio daemon, a device that vanished mid-clip)
# would otherwise hang here forever, and the loop awaits the speaker — so one stuck
# `afplay` stops the agent watching the room, permanently and silently. Utterances
# are a few seconds; a minute means something is wrong, not slow.
PLAY_TIMEOUT_S = 60.0


async def play(path: Path, play_cmd: str, out_index: int) -> None:
    """Play `path`, if this box has any way to. Silent no-op when it doesn't."""
    if out_index >= 0:
        await asyncio.to_thread(_to_device, path, out_index)
    elif play_cmd:
        proc = await asyncio.create_subprocess_exec(*play_cmd.split(), str(path))
        try:
            await asyncio.wait_for(proc.wait(), PLAY_TIMEOUT_S)
        except asyncio.TimeoutError:
            # Losing one utterance beats losing the agent: kill it and keep going.
            proc.kill()
            await proc.wait()
            print(f"warning: {play_cmd.split()[0]} hung on {path.name} — killed it")
