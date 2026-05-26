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
    fps: float = float(os.environ.get("SACCADE_FPS", "1.0"))
    # Glance downscales its input (peripheral = low acuity, saves tokens). Focus
    # always gets full resolution (it reasons carefully and runs rarely). 0 = off.
    glance_max_dim: int = int(os.environ.get("SACCADE_GLANCE_MAX_DIM", "768"))

    # Backends: "stub" (no key), "gemini", "openai", "anthropic"
    glance_backend: str = os.environ.get("SACCADE_GLANCE_BACKEND", "stub")
    focus_backend: str = os.environ.get("SACCADE_FOCUS_BACKEND", "stub")
    # Empty = use that provider's default model (see DEFAULT_MODELS in __main__)
    glance_model: str = os.environ.get("SACCADE_GLANCE_MODEL", "")
    focus_model: str = os.environ.get("SACCADE_FOCUS_MODEL", "")

    episodic_path: str = os.environ.get("SACCADE_EPISODIC", "episodic.jsonl")
    preferences_path: str = os.environ.get("SACCADE_PREFS", "preferences.md")
