"""Interactive setup: probe the machine, pick devices, write .env.

    python -m saccade setup

Answers three questions — what saccade watches or hears, which model thinks, how
it answers — and writes the `.env` that `python -m saccade` reads. Everything it
writes is a plain env var you can edit by hand afterwards (see `.env.example`);
the wizard is a convenience over `python -m saccade devices`, not a new config
system.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from saccade.devices import _audio, _cameras, _screens

# A menu entry: what the user sees, and the env vars picking it writes.
Choice = tuple[str, dict[str, str]]

Devices = tuple[list[tuple[int, str]], list[tuple[int, str]], list[tuple[int, str]]]


def _ask(question: str, choices: list[Choice]) -> dict[str, str]:
    """Print a numbered menu and return the chosen entry's env vars. Reprompts
    until the answer is a valid index; empty input takes the first entry."""
    print(f"\n{question}")
    for n, (label, _) in enumerate(choices, start=1):
        print(f"  [{n}] {label}")
    while True:
        raw = input(f"  > [1-{len(choices)}, default 1] ").strip()
        if not raw:
            return choices[0][1]
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1][1]
        print("  (pick one of the numbers above)")


def _sensor_choices(devs: Devices) -> list[Choice]:
    cams, screens, mics = devs
    out: list[Choice] = []
    for i, desc in cams:
        out.append(
            (f"Camera {i} — {desc}", {"SACCADE_SENSOR": "webcam", "SACCADE_WEBCAM_INDEX": str(i)})
        )
    for i, desc in screens:
        out.append(
            (f"Screen {i} — {desc}", {"SACCADE_SENSOR": "screen", "SACCADE_SCREEN_INDEX": str(i)})
        )
    for i, name in mics:
        out.append((f"Mic {i} — {name}", {"SACCADE_SENSOR": "mic", "SACCADE_MIC_INDEX": str(i)}))
    if cams and mics:
        cam, mic = cams[0][0], mics[0][0]
        out.append(
            (
                f"Camera {cam} + Mic {mic} — see and hear at once",
                {
                    "SACCADE_SENSOR": "av",
                    "SACCADE_WEBCAM_INDEX": str(cam),
                    "SACCADE_MIC_INDEX": str(mic),
                },
            )
        )
    out.append(("Nothing — scripted demo, no hardware", {"SACCADE_SENSOR": "stub"}))
    return out


def _backend_choices() -> list[Choice]:
    """Local-first: Ollama leads when it's installed, since it's free and the
    frames never leave the machine."""
    ollama = shutil.which("ollama") is not None
    tag = "installed" if ollama else "not installed — see https://ollama.com"
    out: list[Choice] = [
        (
            f"Ollama — local, free, private ({tag})",
            {"SACCADE_GLANCE_BACKEND": "ollama", "SACCADE_FOCUS_BACKEND": "ollama"},
        ),
        (
            "Gemini — hosted, needs an API key (the only backend that hears audio)",
            {"SACCADE_GLANCE_BACKEND": "gemini", "SACCADE_FOCUS_BACKEND": "gemini"},
        ),
        (
            "OpenAI — hosted, needs an API key",
            {"SACCADE_GLANCE_BACKEND": "openai", "SACCADE_FOCUS_BACKEND": "openai"},
        ),
        (
            "Anthropic — hosted, needs an API key",
            {"SACCADE_GLANCE_BACKEND": "anthropic", "SACCADE_FOCUS_BACKEND": "anthropic"},
        ),
        (
            "Stub — no model, scripted output",
            {"SACCADE_GLANCE_BACKEND": "stub", "SACCADE_FOCUS_BACKEND": "stub"},
        ),
    ]
    if not ollama:
        out.append(out.pop(0))  # don't lead with something they can't run
    return out


KEY_VARS = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def _speaker_choices(outs: list[tuple[int, str]]) -> list[Choice]:
    out: list[Choice] = [("Text in the terminal", {"SACCADE_SPEAKER": "print"})]
    play = "afplay" if sys.platform == "darwin" else "aplay"
    for i, name in outs:
        out.append(
            (
                f"Speak out loud via {name} (Gemini TTS, needs GEMINI_API_KEY)",
                {
                    "SACCADE_SPEAKER": "gemini_tts",
                    "SACCADE_AUDIO_OUT_INDEX": str(i),
                    "SACCADE_PLAY_CMD": play,
                },
            )
        )
    return out


def _missing_extras(devs: Devices, hints: tuple[str, str, str]) -> list[str]:
    """Extras worth installing given what the probes couldn't see. Only the
    ImportError hints in devices.py start with `uv pip install` — match that,
    not a bare "install", or the PortAudio hint ("apt install libportaudio2")
    gets misread as a missing Python package."""
    names = ("camera", "screen", "audio")
    found = (devs[0], devs[1], devs[2])
    return [
        name
        for name, items, hint in zip(names, found, hints, strict=True)
        if not items and hint.startswith("uv pip install")
    ]


def _write_env(path: Path, env: dict[str, str]) -> bool:
    """Write env as KEY=value lines. Backs up an existing file rather than
    silently clobbering someone's hand-tuned config."""
    if path.exists():
        ans = input(f"\n{path} already exists. Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("\nLeft it alone. Your picks, to paste in yourself:\n")
            for k, v in env.items():
                print(f"  {k}={v}")
            return False
        backup = path.with_suffix(".env.bak")
        shutil.copyfile(path, backup)
        print(f"  (backed up to {backup})")
    body = "\n".join(f"{k}={v}" for k, v in env.items())
    path.write_text(f"# Written by `python -m saccade setup`. Edit freely.\n{body}\n")
    return True


def main() -> None:
    if not sys.stdin.isatty():
        print(
            "setup is interactive and stdin isn't a terminal.\n"
            "Run `python -m saccade devices` and set the env vars yourself.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print("\nsaccade setup — probing this machine...")
    cams, cam_hint = _cameras()
    screens, screen_hint = _screens()
    mics, outs, audio_hint = _audio()
    devs: Devices = (cams, screens, mics)

    extras = _missing_extras(devs, (cam_hint, screen_hint, audio_hint))
    if extras:
        joined = ",".join(extras)
        print(
            f"\n  Heads up: no {' / '.join(extras)} support installed, so those "
            f"devices can't be listed.\n  Install it, then rerun setup:\n\n"
            f"    uv pip install -e '.[{joined}]'\n"
        )
    for hint in (cam_hint, screen_hint, audio_hint):
        if hint and "install" not in hint:
            print(f"  note: {hint}")

    env: dict[str, str] = {}
    env.update(_ask("What should saccade watch or hear?", _sensor_choices(devs)))
    env.update(_ask("Which model should think?", _backend_choices()))
    env.update(_ask("How should saccade answer?", _speaker_choices(outs)))

    backend = env.get("SACCADE_GLANCE_BACKEND", "stub")
    needs_key = KEY_VARS.get(backend) or (
        KEY_VARS["gemini"] if env.get("SACCADE_SPEAKER") == "gemini_tts" else None
    )
    if needs_key:
        key = input(f"\n{needs_key} (blank to set it later): ").strip()
        if key:
            env[needs_key] = key

    if not _write_env(Path(".env"), env):
        return

    print("\nWrote .env. Start it with:\n\n    python -m saccade\n")
    if env.get("SACCADE_SENSOR") == "webcam" and sys.platform == "darwin":
        print("macOS: approve the Camera prompt on first run, then rerun.\n")
