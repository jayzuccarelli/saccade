"""Runtime config. Everything tunable lives here — no magic constants in code.

Swap to the real camera/models by changing these (or env), nothing else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _apply_dotenv(path: str) -> dict[str, str]:
    """Parse a KEY=VALUE file and set any key not already in the environment — the
    real environment always wins, like every other dotenv. Returns what it parsed.
    Stdlib only; no python-dotenv dependency."""
    parsed: dict[str, str] = {}
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return parsed
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]  # unquote
        else:
            # python-dotenv semantics: an unquoted value ends at ` #` — otherwise
            # a trailing comment silently becomes part of the value (e.g. an RTSP
            # path that can never connect). Quote the value to keep a literal #.
            val = val.split(" #", 1)[0].rstrip()
        parsed[key] = val
        os.environ.setdefault(key, val)
    return parsed


def _autoload_dotenv() -> None:
    """Load the first .env found: $SACCADE_ENV_FILE, then ./.env, then the repo
    root. This is what kills the launcher script — `python -m saccade` just runs,
    with secrets in a gitignored .env instead of a hand-run shell file."""
    candidates = []
    if explicit := os.environ.get("SACCADE_ENV_FILE"):
        candidates.append(explicit)
    candidates.append(os.path.join(os.getcwd(), ".env"))
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(repo_root, ".env"))
    for path in candidates:
        if os.path.isfile(path):
            _apply_dotenv(path)
            return


_autoload_dotenv()  # before the dataclass: its field defaults read os.environ


@dataclass
class Config:
    # Sensor: "stub", "webcam" (local cam), "screen", "mic" (local mic), "av"
    # (webcam + mic fused), "reolink" (RTSP), or "replay" (folder)
    sensor: str = os.environ.get("SACCADE_SENSOR", "stub")
    webcam_index: int = int(os.environ.get("SACCADE_WEBCAM_INDEX", "0"))
    # Mic input device; -1 = system default (see `saccade devices` for indices).
    mic_index: int = int(os.environ.get("SACCADE_MIC_INDEX", "-1"))
    screen_index: int = int(os.environ.get("SACCADE_SCREEN_INDEX", "1"))
    rtsp_url: str = os.environ.get("SACCADE_RTSP_URL", "")
    # Or give the parts and let saccade assemble + URL-encode the URL — so a
    # password with @ : / # symbols can't break it and creds stay out of shell
    # history. Used only when SACCADE_RTSP_URL is empty (see __post_init__).
    rtsp_user: str = os.environ.get("SACCADE_RTSP_USER", "admin")
    rtsp_password: str = os.environ.get("SACCADE_RTSP_PASSWORD", "")
    rtsp_host: str = os.environ.get("SACCADE_RTSP_HOST", "")  # host or host:port
    rtsp_path: str = os.environ.get("SACCADE_RTSP_PATH", "/h264Preview_01_sub")
    replay_dir: str = os.environ.get("SACCADE_REPLAY_DIR", "frames")
    # Two clocks: capture = how fast frames stream into the buffer; glance = how
    # often we actually call the model. Start aligned (1/1); widen glance (e.g.
    # 0.14 ≈ every 7s) to fit a rate limit while still capturing a dense clip.
    capture_fps: float = float(os.environ.get("SACCADE_CAPTURE_FPS", "1.0"))
    glance_fps: float = float(os.environ.get("SACCADE_GLANCE_FPS", "1.0"))
    # Glance downscales its input (peripheral = low acuity, saves tokens). Focus
    # always gets full resolution (it reasons carefully and runs rarely). 0 = off.
    glance_max_dim: int = int(os.environ.get("SACCADE_GLANCE_MAX_DIM", "768"))
    # Adaptive cadence: let Glance decide how soon to look again (per its
    # next_glance_s). It only ever slows below glance_fps, never faster — quiet
    # scene = rest (up to glance_max_interval s), action = every tick. 0 to disable.
    adaptive_cadence: bool = os.environ.get("SACCADE_ADAPTIVE_CADENCE", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    glance_max_interval: float = float(os.environ.get("SACCADE_GLANCE_MAX_INTERVAL", "15.0"))

    # Concurrent Focus: a salient frame spawns Focus in the background so Glance
    # keeps watching while the big model reasons (single-slot — one at a time). 0
    # to reason inline (Glance pauses until Focus + the action finish).
    concurrent_focus: bool = os.environ.get("SACCADE_CONCURRENT_FOCUS", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

    # Memory buffer sizes + how many recent frames Focus sees as a clip.
    sensory_buffer: int = int(os.environ.get("SACCADE_SENSORY_BUFFER", "16"))
    working_memory: int = int(os.environ.get("SACCADE_WORKING_MEMORY", "30"))
    focus_clip_frames: int = int(os.environ.get("SACCADE_FOCUS_CLIP_FRAMES", "6"))

    # Backends: "stub" (no key), "ollama" (local), "gemini", "openai", "anthropic"
    glance_backend: str = os.environ.get("SACCADE_GLANCE_BACKEND", "stub")
    focus_backend: str = os.environ.get("SACCADE_FOCUS_BACKEND", "stub")
    # Empty = use that provider's default model (see DEFAULT_MODELS in __main__)
    glance_model: str = os.environ.get("SACCADE_GLANCE_MODEL", "")
    focus_model: str = os.environ.get("SACCADE_FOCUS_MODEL", "")
    # Ollama endpoint (empty = use OLLAMA_HOST or http://localhost:11434).
    ollama_host: str = os.environ.get("SACCADE_OLLAMA_HOST", "")

    episodic_path: str = os.environ.get("SACCADE_EPISODIC", "episodic.jsonl")
    preferences_path: str = os.environ.get("SACCADE_PREFS", "preferences.md")
    # How far back Focus treats its own past utterances as "recent" (the anti-nag
    # window). Older lines drop out, so a fresh run isn't muted by what it said in
    # a prior session — episodic is on disk and persists across runs. Seconds.
    recent_said_window_s: float = float(os.environ.get("SACCADE_RECENT_SAID_WINDOW", "180"))

    # Speaker (output): "print" (default), "piper" (local TTS, no key), "gemini_tts"
    # (hosted TTS, better voices), or "home_assistant" (play on a media_player via HA).
    speaker: str = os.environ.get("SACCADE_SPEAKER", "print")
    # piper speaker: which downloaded voice to use, and where voices live (blank =
    # piper's own default dir). `python -m piper.download_voices` lists them.
    piper_voice: str = os.environ.get("SACCADE_PIPER_VOICE", "en_US-lessac-medium")
    piper_data_dir: str = os.environ.get("SACCADE_PIPER_DATA_DIR", "")
    tts_model: str = os.environ.get("SACCADE_TTS_MODEL", "gemini-2.5-flash-preview-tts")
    tts_voice: str = os.environ.get("SACCADE_TTS_VOICE", "Kore")
    tts_dir: str = os.environ.get("SACCADE_TTS_DIR", "utterances")
    # A command that takes a wav path and plays it (aplay/afplay/a push wrapper).
    # Empty = synthesize and save only (the watching box may have no audio out).
    play_cmd: str = os.environ.get("SACCADE_PLAY_CMD", "")
    # Pick a specific output device by index (see `saccade devices`); -1 = OS
    # default via play_cmd. Needs the audio extra. Wins over play_cmd when >=0.
    audio_out_index: int = int(os.environ.get("SACCADE_AUDIO_OUT_INDEX", "-1"))

    # home_assistant speaker: play the clip on a media_player. saccade serves the
    # audio itself, so HA just fetches serve_host:serve_port — no HA www needed.
    ha_url: str = os.environ.get("SACCADE_HA_URL", "http://localhost:8123")
    ha_token: str = os.environ.get("SACCADE_HA_TOKEN", "")
    ha_entity: str = os.environ.get("SACCADE_HA_ENTITY", "")
    serve_host: str = os.environ.get("SACCADE_SERVE_HOST", "")  # blank = auto-detect LAN IP
    serve_port: int = int(os.environ.get("SACCADE_SERVE_PORT", "8189"))

    def __post_init__(self) -> None:
        if not self.rtsp_url and self.rtsp_host:
            from urllib.parse import quote

            userinfo = self.rtsp_user
            if self.rtsp_password:
                userinfo += ":" + quote(self.rtsp_password, safe="")
            self.rtsp_url = f"rtsp://{userinfo}@{self.rtsp_host}{self.rtsp_path}"
