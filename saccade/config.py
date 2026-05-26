"""Runtime config. Everything tunable lives here — no magic constants in code.

Swap to the real camera/models by changing these (or env), nothing else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    # Sensor: "stub" (scripted), "reolink" (RTSP), or "replay" (folder of images)
    sensor: str = os.environ.get("SACCADE_SENSOR", "stub")
    rtsp_url: str = os.environ.get("SACCADE_RTSP_URL", "")
    replay_dir: str = os.environ.get("SACCADE_REPLAY_DIR", "frames")
    # Two clocks: capture = how fast frames stream into the buffer; glance = how
    # often we actually call the model. Start aligned (1/1); widen glance (e.g.
    # 0.14 ≈ every 7s) to fit a rate limit while still capturing a dense clip.
    capture_fps: float = float(os.environ.get("SACCADE_CAPTURE_FPS", "1.0"))
    glance_fps: float = float(os.environ.get("SACCADE_GLANCE_FPS", "1.0"))
    # Glance downscales its input (peripheral = low acuity, saves tokens). Focus
    # always gets full resolution (it reasons carefully and runs rarely). 0 = off.
    glance_max_dim: int = int(os.environ.get("SACCADE_GLANCE_MAX_DIM", "768"))

    # Memory buffer sizes + how many recent frames Focus sees as a clip.
    sensory_buffer: int = int(os.environ.get("SACCADE_SENSORY_BUFFER", "16"))
    working_memory: int = int(os.environ.get("SACCADE_WORKING_MEMORY", "30"))
    focus_clip_frames: int = int(os.environ.get("SACCADE_FOCUS_CLIP_FRAMES", "6"))

    # Backends: "stub" (no key), "gemini", "openai", "anthropic"
    glance_backend: str = os.environ.get("SACCADE_GLANCE_BACKEND", "stub")
    focus_backend: str = os.environ.get("SACCADE_FOCUS_BACKEND", "stub")
    # Empty = use that provider's default model (see DEFAULT_MODELS in __main__)
    glance_model: str = os.environ.get("SACCADE_GLANCE_MODEL", "")
    focus_model: str = os.environ.get("SACCADE_FOCUS_MODEL", "")

    episodic_path: str = os.environ.get("SACCADE_EPISODIC", "episodic.jsonl")
    preferences_path: str = os.environ.get("SACCADE_PREFS", "preferences.md")
