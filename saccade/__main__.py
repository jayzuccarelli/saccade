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

from saccade import loop as looplib
from saccade.config import Config
from saccade.focus import Focus
from saccade.glance import Glance
from saccade.memory import Memory
from saccade.schema import Frame, Window


def make_sensor(c: Config):
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
    ("gemini", "glance"): "gemini-2.5-flash-lite",
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

        return GeminiTTSSpeaker(c.tts_model, c.tts_voice, c.tts_dir, c.play_cmd)
    if c.speaker == "home_assistant":
        from saccade.speakers.gemini_tts import GeminiTTSSpeaker
        from saccade.speakers.home_assistant import HomeAssistantSpeaker

        tts = GeminiTTSSpeaker(c.tts_model, c.tts_voice, c.tts_dir)
        return HomeAssistantSpeaker(
            tts, c.ha_url, c.ha_token, c.ha_entity, c.serve_host, c.serve_port
        )
    from saccade.speakers.print import PrintSpeaker

    return PrintSpeaker()


async def snapshot(path: str) -> None:
    """Run one image through Glance (and Focus if it escalates). The fastest way
    to see a real Percept the moment a key is wired: `python -m saccade snapshot pic.jpg`."""
    c = Config()
    with open(path, "rb") as f:
        data = f.read()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
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
    c = Config()
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
        sensor, glance, focus, memory, on_action=speaker.say,
        glance_fps=c.glance_fps, focus_clip_frames=c.focus_clip_frames,
    )


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "snapshot":
        asyncio.run(snapshot(sys.argv[2]))
    else:
        asyncio.run(main())
