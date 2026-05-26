"""Wire it together and run.

    python -m saccade                          # scripted stub, no key, no camera
    SACCADE_GLANCE_BACKEND=gemini \
    SACCADE_FOCUS_BACKEND=gemini \
    GEMINI_API_KEY=... python -m saccade        # real models
    SACCADE_SENSOR=reolink SACCADE_RTSP_URL=rtsp://... python -m saccade
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

        return ReolinkSensor(c.rtsp_url, c.fps)
    if c.sensor == "replay":
        from saccade.sensors.replay import ReplaySensor

        return ReplaySensor(c.replay_dir, c.fps)
    from saccade.sensors.stub import StubSensor

    return StubSensor(c.fps)


# Sensible per-provider defaults for each tier. Glance = cheap/fast, Focus = smart.
# Gemini + Anthropic IDs verified; OpenAI IDs may need adjusting to current names.
DEFAULT_MODELS = {
    ("gemini", "glance"): "gemini-2.5-flash-lite",
    ("gemini", "focus"): "gemini-3.5-flash",
    ("openai", "glance"): "gpt-4.1-nano",
    ("openai", "focus"): "gpt-5.5",
    ("anthropic", "glance"): "claude-haiku-4-5",
    ("anthropic", "focus"): "claude-sonnet-4-6",
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
    from saccade.backends.stub import StubBackend

    return StubBackend(role=role)


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
    focus = Focus(make_backend(c.focus_backend, "focus", c))
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
    focus = Focus(make_backend(c.focus_backend, "focus", c))
    memory = Memory(
        c.episodic_path, c.preferences_path, sensory_n=c.sensory_buffer, working_n=c.working_memory
    )

    def speak(msg: str) -> None:
        print(f"\n    \033[1m\033[96m💬  {msg}\033[0m\n")

    print(f"saccade v0 — sensor={c.sensor} glance={c.glance_backend} focus={c.focus_backend}\n")
    await looplib.run(sensor, glance, focus, memory, on_action=speak, focus_clip_frames=c.focus_clip_frames)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "snapshot":
        asyncio.run(snapshot(sys.argv[2]))
    else:
        asyncio.run(main())
