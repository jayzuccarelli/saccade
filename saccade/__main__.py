"""Wire it together and run.

    python -m saccade                          # scripted stub, no key, no camera
    SACCADE_GLANCE_BACKEND=gemini \
    SACCADE_FOCUS_BACKEND=gemini \
    GEMINI_API_KEY=... python -m saccade        # real models

Settings come from the environment (see config.py). Drop them in a gitignored
.env (copy .env.example) and `python -m saccade` just runs: no launcher script.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import TYPE_CHECKING, Any

from saccade import __version__

if TYPE_CHECKING:
    from saccade.backends.base import Backend
    from saccade.config import Config
    from saccade.sensors.base import Sensor
    from saccade.speakers.base import Speaker

# saccade imports live inside the functions below: importing config parses env
# vars, so `saccade devices` must not trigger it; it's the tool you reach for
# when your .env is broken.


def make_sensor(c: Config) -> Sensor:
    """One sensor, or several merged. `SACCADE_SENSOR=screen,mic` runs both and
    interleaves their frames; a single name behaves exactly as before."""
    kinds = [k.strip() for k in c.sensor.split(",") if k.strip()]
    if len(kinds) > 1:
        from saccade.sensors.multi import MultiSensor

        return MultiSensor([_one_sensor(k, c) for k in kinds], labels=kinds)
    return _one_sensor(kinds[0] if kinds else "stub", c)


def _one_sensor(kind: str, c: Config) -> Sensor:
    if kind == "webcam":
        from saccade.sensors.webcam import WebcamSensor

        return WebcamSensor(c.webcam_index, c.capture_fps)
    if kind == "screen":
        from saccade.sensors.screen import ScreenSensor

        return ScreenSensor(c.screen_index, c.capture_fps)
    if kind == "mic":
        from saccade.sensors.mic import MicSensor

        return MicSensor(
            c.mic_index if c.mic_index >= 0 else None, c.capture_fps, transcriber=make_transcriber(c)
        )
    if kind == "av":
        from saccade.sensors.av import AVSensor

        return AVSensor(
            c.webcam_index,
            c.mic_index if c.mic_index >= 0 else None,
            c.capture_fps,
            transcriber=make_transcriber(c),
        )
    if kind == "reolink":
        from saccade.sensors.reolink import ReolinkSensor

        return ReolinkSensor(c.rtsp_url, c.capture_fps)
    if kind == "replay":
        from saccade.sensors.replay import ReplaySensor

        return ReplaySensor(c.replay_dir, c.capture_fps)
    from saccade.sensors.stub import StubSensor

    return StubSensor(c.capture_fps)


def make_transcriber(c: Config) -> Any:
    """Local speech-to-text, or None to hand raw audio to the backend instead."""
    if c.stt != "whisper":
        return None
    from saccade.stt import Transcriber

    return Transcriber(c.stt_model)


# Sensible per-provider defaults for each tier. Glance = cheap/fast, Focus = smart.
# Gemini + Anthropic IDs verified; OpenAI IDs may need adjusting to current names.
DEFAULT_MODELS = {
    ("gemini", "glance"): "gemini-3.1-flash-lite",
    ("gemini", "focus"): "gemini-3.5-flash",
    ("openai", "glance"): "gpt-4.1-nano",
    ("openai", "focus"): "gpt-5.5",
    ("anthropic", "glance"): "claude-haiku-4-5",
    ("anthropic", "focus"): "claude-sonnet-4-6",
    ("ollama", "glance"): "gemma3:4b",
    ("ollama", "focus"): "gemma3:12b",
}


def make_backend(kind: str, role: str, c: Config) -> Backend:
    override = c.glance_model if role == "glance" else c.focus_model
    model = override or DEFAULT_MODELS.get((kind, role), "")
    if kind == "gemini":
        from saccade.backends.gemini import GeminiBackend

        return GeminiBackend(model)
    if kind == "openai":
        from saccade.backends.openai import OpenAIBackend

        return OpenAIBackend(model)
    if kind == "anthropic":
        from saccade.backends.anthropic import AnthropicBackend

        return AnthropicBackend(model)
    if kind == "ollama":
        from saccade.backends.ollama import OllamaBackend

        return OllamaBackend(model, host=c.ollama_host or None)
    from saccade.backends.stub import StubBackend

    return StubBackend(role=role)


def make_speaker(c: Config) -> Speaker:
    if c.speaker == "piper":
        from saccade.speakers.piper import PiperSpeaker

        return PiperSpeaker(
            c.piper_voice,
            c.tts_dir,
            c.play_cmd,
            out_index=c.audio_out_index,
            data_dir=c.piper_data_dir,
        )
    if c.speaker == "gemini_tts":
        from saccade.speakers.gemini_tts import GeminiTTSSpeaker

        return GeminiTTSSpeaker(
            c.tts_model, c.tts_voice, c.tts_dir, c.play_cmd, out_index=c.audio_out_index
        )
    if c.speaker == "home_assistant":
        from saccade.speakers.home_assistant import HomeAssistantSpeaker

        # Built without play_cmd/out_index on purpose: the clip is played on the
        # media_player, so synthesizing here must not also play it on this box.
        tts: Any
        if c.ha_tts == "gemini_tts":
            from saccade.speakers.gemini_tts import GeminiTTSSpeaker

            tts = GeminiTTSSpeaker(c.tts_model, c.tts_voice, c.tts_dir)
        else:
            from saccade.speakers.piper import PiperSpeaker

            tts = PiperSpeaker(c.piper_voice, c.tts_dir, data_dir=c.piper_data_dir)
        return HomeAssistantSpeaker(
            tts, c.ha_url, c.ha_token, c.ha_entity, c.serve_host, c.serve_port
        )
    from saccade.speakers.print import PrintSpeaker

    return PrintSpeaker()


# A file extension -> its audio MIME. Anything else is treated as an image.
AUDIO_MIMES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


async def snapshot(path: str) -> None:
    """Run one image or audio clip through Glance (and Focus if it escalates). The
    fastest way to see a real Percept the moment a key is wired:
    `python -m saccade snapshot pic.jpg` or `... snapshot clip.wav` (audio needs
    the gemini backend)."""
    from saccade.config import Config
    from saccade.focus import Focus
    from saccade.glance import Glance
    from saccade.memory import Memory
    from saccade.schema import Frame, Window

    c = Config()
    with open(path, "rb") as f:
        data = f.read()
    lower = path.lower()
    audio_mime = next((m for ext, m in AUDIO_MIMES.items() if lower.endswith(ext)), None)
    if audio_mime:
        frame = Frame(ts=time.time(), audio=data, audio_mime=audio_mime)
    else:
        mime = "image/png" if lower.endswith(".png") else "image/jpeg"
        frame = Frame(ts=time.time(), image=data, mime=mime)
    window = Window(frames=[frame])
    glance = Glance(make_backend(c.glance_backend, "glance", c), max_dim=c.glance_max_dim)
    focus = Focus(make_backend(c.focus_backend, "focus", c), c.recent_said_window_s)
    memory = Memory(c.episodic_path, c.preferences_path)

    percept = await glance.perceive(window, memory)
    print(f"PERCEPT:  {percept}")
    if percept.escalate:
        decision = await focus.reason(percept, window, memory)
        print(f"DECISION: {decision}")
    else:
        print("(not salient, Focus not invoked)")


async def main() -> None:
    from saccade import loop as looplib
    from saccade.config import Config
    from saccade.focus import Focus
    from saccade.glance import Glance
    from saccade.memory import Memory

    c = Config()
    if c.sensor == "stub" and c.glance_backend == "stub" and c.focus_backend == "stub":
        # Nothing is configured at all: the first thing a fresh clone hits.
        # Don't leave them watching a canned scene with no way forward.
        print(
            "Nothing configured yet: this is a scripted demo with a stub model,\n"
            "not your camera. To point saccade at real hardware and a real model:\n\n"
            f"    {sys.executable} -m saccade setup\n"
        )
    elif c.sensor == "stub":
        print(
            "note: SACCADE_SENSOR is unset, so the stub sensor is feeding the real "
            f"model no images.\nPoint it at something: `{sys.executable} -m saccade setup`, or set "
            "SACCADE_SENSOR=webcam / screen / reolink.\n"
        )
    sensor = make_sensor(c)
    glance = Glance(make_backend(c.glance_backend, "glance", c), max_dim=c.glance_max_dim)
    focus = Focus(make_backend(c.focus_backend, "focus", c), c.recent_said_window_s)
    memory = Memory(
        c.episodic_path, c.preferences_path, sensory_n=c.sensory_buffer, working_n=c.working_memory
    )

    speaker = make_speaker(c)

    print(
        f"saccade v{__version__}: sensor={c.sensor} glance={c.glance_backend} "
        f"focus={c.focus_backend} speaker={c.speaker}\n"
    )
    await looplib.run(
        sensor,
        glance,
        focus,
        memory,
        on_action=speaker.say,
        glance_fps=c.glance_fps,
        focus_clip_frames=c.focus_clip_frames,
        adaptive_cadence=c.adaptive_cadence,
        glance_max_interval=c.glance_max_interval,
        concurrent_focus=c.concurrent_focus,
    )


USAGE = """usage: saccade [command]

  (no command)      run the ambient loop (sensor/models/speaker from env or .env)
  setup             pick your camera/mic/screen and model, write .env
  devices           list cameras, screens, mics, and audio outputs
  snapshot <file>   run one image or audio clip through Glance (then Focus if salient)"""


# Which extra ships each lazily-imported dependency. Sensors and speakers import
# these inside the function that needs them, so a missing one surfaces at the
# first frame rather than at startup.
EXTRA_FOR_MODULE = {
    "cv2": "camera",
    "mss": "screen",
    "PIL": "camera",
    "sounddevice": "audio",
    "numpy": "audio",
    "google": "gemini",
    "openai": "openai",
    "anthropic": "anthropic",
    # Reached only when someone picked local transcription without the extra, which
    # is exactly the case that produced a 40-line asyncio traceback ending in
    # `No module named 'faster_whisper'` on the first audio frame.
    "faster_whisper": "stt",
}


def _dependency_hint(exc: ModuleNotFoundError) -> str:
    """Turn a missing optional dep into the line that fixes it, or "" if it isn't
    one of ours. `SACCADE_SENSOR=webcam` with no cv2 installed is a completely
    ordinary thing to do, and answering it with a twenty-line asyncio traceback
    tells the user they broke something when they just haven't installed the
    extra yet."""
    extra = EXTRA_FOR_MODULE.get((exc.name or "").split(".")[0])
    if not extra:
        return ""
    return (
        f"saccade needs the '{extra}' extra for this configuration "
        f"(no module named {exc.name!r}).\n\n"
        f"    uv pip install -e '.[{extra}]'\n\n"
        f"Then rerun. `{sys.executable} -m saccade devices` shows what's available."
    )


def cli() -> None:
    """Sync entry point: both `python -m saccade` and the installed `saccade`
    script land here. Unknown input gets usage, not the infinite loop."""
    try:
        _cli()
    except KeyboardInterrupt:
        # Ctrl-C is how you stop an ambient agent, so it exits, now: not a stack
        # trace, and not a wait either. A model call is usually in flight in a
        # worker thread, a blocking HTTP request can't be cancelled, and the
        # interpreter joins those threads at exit; with a 120s client timeout that
        # is two minutes of a dead-looking terminal eating further Ctrl-Cs. So
        # flush what we have and go. Nothing here outlives the process: percepts
        # and episodes are written as they happen, and a half-finished glance is
        # worth nothing anyway.
        print("\nstopped", file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(130)
    except ModuleNotFoundError as e:
        hint = _dependency_hint(e)
        if not hint:
            raise
        print(f"\n{hint}\n", file=sys.stderr)
        raise SystemExit(1) from None


def _cli() -> None:
    argv = sys.argv[1:]
    if not argv:
        asyncio.run(main())
    elif argv[0] in ("-h", "--help"):
        print(USAGE)
    elif argv[0] == "setup":
        from saccade.setup import main as setup_main

        setup_main()
    elif argv[0] == "devices":
        from saccade.devices import main as devices_main

        devices_main()
    elif argv[0] == "snapshot" and len(argv) == 2:
        asyncio.run(snapshot(argv[1]))
    else:
        print(USAGE, file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    cli()
