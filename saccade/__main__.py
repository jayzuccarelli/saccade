"""Wire it together and run.

    python -m saccade                          # scripted stub, no key, no camera
    SACCADE_GLANCE_BACKEND=gemini \
    SACCADE_FOCUS_BACKEND=gemini \
    GEMINI_API_KEY=... python -m saccade        # real models

Settings come from the environment (see config.py). Drop them in a gitignored
.env (copy .env.example) and `python -m saccade` just runs — no launcher script.
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saccade.config import Config

# saccade imports live inside the functions below: importing config parses env
# vars, so `saccade devices` must not trigger it — it's the tool you reach for
# when your .env is broken.


def make_sensor(c: Config):
    if c.sensor == "webcam":
        from saccade.sensors.webcam import WebcamSensor

        return WebcamSensor(c.webcam_index, c.capture_fps)
    if c.sensor == "screen":
        from saccade.sensors.screen import ScreenSensor

        return ScreenSensor(c.screen_index, c.capture_fps)
    if c.sensor == "mic":
        from saccade.sensors.mic import MicSensor

        return MicSensor(c.mic_index if c.mic_index >= 0 else None, c.capture_fps)
    if c.sensor == "av":
        from saccade.sensors.av import AVSensor

        return AVSensor(
            c.webcam_index, c.mic_index if c.mic_index >= 0 else None, c.capture_fps
        )
    if c.sensor == "reolink":
        from saccade.sensors.reolink import ReolinkSensor

        return ReolinkSensor(c.rtsp_url, c.capture_fps)
    if c.sensor == "replay":
        from saccade.sensors.replay import ReplaySensor

        return ReplaySensor(c.replay_dir, c.capture_fps)
    from saccade.sensors.stub import StubSensor

    return StubSensor(c.capture_fps)


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


def make_backend(kind: str, role: str, c: Config):
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


def make_speaker(c: Config):
    if c.speaker == "gemini_tts":
        from saccade.speakers.gemini_tts import GeminiTTSSpeaker

        return GeminiTTSSpeaker(
            c.tts_model, c.tts_voice, c.tts_dir, c.play_cmd, out_index=c.audio_out_index
        )
    if c.speaker == "home_assistant":
        from saccade.speakers.gemini_tts import GeminiTTSSpeaker
        from saccade.speakers.home_assistant import HomeAssistantSpeaker

        tts = GeminiTTSSpeaker(c.tts_model, c.tts_voice, c.tts_dir)
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
        print("(not salient — Focus not invoked)")


async def main() -> None:
    from saccade import loop as looplib
    from saccade.config import Config
    from saccade.focus import Focus
    from saccade.glance import Glance
    from saccade.memory import Memory

    c = Config()
    if c.sensor == "stub" and (c.glance_backend != "stub" or c.focus_backend != "stub"):
        print(
            "note: SACCADE_SENSOR is unset, so the stub sensor is feeding the real "
            "model no images.\nPoint it at something: SACCADE_SENSOR=webcam / screen / "
            "reolink, or try `python -m saccade snapshot pic.jpg`.\n"
        )
    sensor = make_sensor(c)
    glance = Glance(make_backend(c.glance_backend, "glance", c), max_dim=c.glance_max_dim)
    focus = Focus(make_backend(c.focus_backend, "focus", c), c.recent_said_window_s)
    memory = Memory(
        c.episodic_path, c.preferences_path, sensory_n=c.sensory_buffer, working_n=c.working_memory
    )

    speaker = make_speaker(c)

    print(
        f"saccade v0 — sensor={c.sensor} glance={c.glance_backend} "
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
  devices           list cameras, screens, mics, and audio outputs
  snapshot <file>   run one image or audio clip through Glance (then Focus if salient)"""


def cli() -> None:
    """Sync entry point — both `python -m saccade` and the installed `saccade`
    script land here. Unknown input gets usage, not the infinite loop."""
    argv = sys.argv[1:]
    if not argv:
        asyncio.run(main())
    elif argv[0] in ("-h", "--help"):
        print(USAGE)
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
