"""Play a wav: to a chosen output device, through a command, or not at all.

Shared by every speaker that synthesizes to a file. The box saccade watches from
may have no audio out at all, so "wrote the clip" is a valid outcome and
playback is the optional half.

  - `out_index` (SACCADE_AUDIO_OUT_INDEX): a specific device by index (the
    numbers `saccade devices` lists), via sounddevice. The symmetric twin of
    picking a mic. Wins over play_cmd when set.
  - `play_cmd` (SACCADE_PLAY_CMD): a command taking the file path: `aplay`,
    `afplay`, or a wrapper that pushes it somewhere. Uses the OS default device.
"""

from __future__ import annotations

import asyncio
import wave
from pathlib import Path


def _to_device(path: Path, out_index: int) -> bool:
    """Blocking playback to one specific device; callers run it off-thread.
    False means this device can't take the clip and the caller should fall back.

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

    # What the clip has and what the device accepts are two different numbers,
    # and we used to send the first without asking about the second. Piper writes
    # mono, plenty of CoreAudio outputs will only open a stereo stream, and the
    # result is `PortAudioError: Invalid number of channels [-9998]` on every
    # utterance: the agent watches the room all day and never makes a sound.
    wants = int(sd.query_devices(out_index)["max_output_channels"])
    if wants < 1:
        print(f"warning: audio device {out_index} has no output; using the default instead")
        return False
    if channels != wants:
        # Down to one, then out to however many it wants: that lands on exactly
        # `wants` for every pair, where scaling by `wants // channels` quietly
        # didn't (2 into 3 stayed 2, and PortAudio refuses it just the same).
        # mean() accumulates in float64 and is cast after, because averaging
        # int16 in int16 overflows on anything loud.
        mono = data.reshape(-1, channels).mean(axis=1) if channels > 1 else data
        data = np.repeat(mono.reshape(-1, 1).astype(np.int16), wants, axis=1)
    sd.play(data, samplerate=rate, device=out_index)
    sd.wait()
    return True


# A player that never exits (wedged audio daemon, a device that vanished mid-clip)
# would otherwise hang here forever, and the loop awaits the speaker, so one stuck
# `afplay` stops the agent watching the room, permanently and silently. Utterances
# are a few seconds; a minute means something is wrong, not slow.
PLAY_TIMEOUT_S = 60.0


async def play(path: Path, play_cmd: str, out_index: int) -> None:
    """Play `path`, if this box has any way to. Silent no-op when it doesn't."""
    if out_index >= 0 and await asyncio.to_thread(_to_device, path, out_index):
        return
    if play_cmd:
        proc = await asyncio.create_subprocess_exec(*play_cmd.split(), str(path))
        try:
            await asyncio.wait_for(proc.wait(), PLAY_TIMEOUT_S)
        except asyncio.TimeoutError:
            # Losing one utterance beats losing the agent: kill it and keep going.
            proc.kill()
            await proc.wait()
            print(f"warning: {play_cmd.split()[0]} hung on {path.name}; killed it")
