"""Interactive setup: probe the machine, pick devices, write .env.

    python -m saccade setup

Answers three questions (what saccade watches or hears, which model thinks, how
it answers) and writes the `.env` that `python -m saccade` reads. Everything it
writes is a plain env var you can edit by hand afterwards (see `.env.example`);
the wizard is a convenience over `python -m saccade devices`, not a new config
system.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib import request

from saccade.devices import _audio, _cameras, _screens

_OLLAMA_HOST = "http://localhost:11434"

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


def _one_sensor_choices(devs: Devices) -> list[Choice]:
    """Every individual device, one entry each. The building block for both the
    single pick and the several-at-once pick."""
    cams, screens, mics = devs
    out: list[Choice] = []
    for i, desc in cams:
        out.append(
            (f"Camera {i}: {desc}", {"SACCADE_SENSOR": "webcam", "SACCADE_WEBCAM_INDEX": str(i)})
        )
    for i, desc in screens:
        out.append(
            (f"Screen {i}: {desc}", {"SACCADE_SENSOR": "screen", "SACCADE_SCREEN_INDEX": str(i)})
        )
    for i, name in mics:
        out.append((f"Mic {i}: {name}", {"SACCADE_SENSOR": "mic", "SACCADE_MIC_INDEX": str(i)}))
    return out


def _sensor_kinds(env: dict[str, str]) -> set[str]:
    """The sensor kinds this env selects. A single name, or several after a
    "several at once" pick: every downstream question (does it hear? does macOS
    need a camera prompt?) has to look at all of them, not just the string."""
    return {k.strip() for k in env.get("SACCADE_SENSOR", "").split(",") if k.strip()}


def _merge_sensors(picked: list[Choice]) -> tuple[dict[str, str], list[str]]:
    """Fold several single-device picks into one env block, e.g. screen + mic ->
    SACCADE_SENSOR=screen,mic plus each index var.

    Two of the same kind can't both be expressed: there's one
    SACCADE_WEBCAM_INDEX, so a second camera would silently overwrite the first.
    Rather than quietly watching a camera they didn't choose, the extras come
    back as `dropped` for the caller to say out loud."""
    env: dict[str, str] = {}
    kinds: list[str] = []
    dropped: list[str] = []
    for label, choice in picked:
        kind = choice["SACCADE_SENSOR"]
        if kind in kinds:
            dropped.append(label)
            continue
        kinds.append(kind)
        env.update({k: v for k, v in choice.items() if k != "SACCADE_SENSOR"})
    env["SACCADE_SENSOR"] = ",".join(kinds)
    return env, dropped


def _sensor_choices(devs: Devices) -> list[Choice]:
    cams, screens, mics = devs
    out: list[Choice] = _one_sensor_choices(devs)
    singles = len(out)
    if cams and mics:
        # Which camera and which mic is a follow-up question: pairing the first
        # of each looks reasonable until you meet a Mac whose first mic is the
        # user's iPhone and whose first camera is the built-in webcam.
        out.append(("A camera and a mic together: see and hear at once", {"SACCADE_SENSOR": "av"}))
    if singles >= 2:
        # Distinct from `av`: that fuses one camera grab and one mic clip into a
        # single Frame describing one instant. This just runs several inputs at
        # their own pace and interleaves them: watch the screen, hear the room.
        out.append(("Several at once: pick which on the next screen", {"SACCADE_SENSOR": "multi"}))
    out.append(("Nothing: scripted demo, no hardware", {"SACCADE_SENSOR": "stub"}))
    return out


def _ask_many(question: str, choices: list[Choice]) -> list[Choice]:
    """Like _ask, but takes several: comma-separated numbers. Order is preserved
    and repeats collapse, so "3,1,3" means the third and the first."""
    print(f"\n{question}")
    for n, (label, _) in enumerate(choices, start=1):
        print(f"  [{n}] {label}")
    while True:
        raw = input(f"  > [comma-separated, 1-{len(choices)}, e.g. 1,3] ").strip()
        if not raw:
            return [choices[0]]
        picks = [p.strip() for p in raw.split(",") if p.strip()]
        if picks and all(p.isdigit() and 1 <= int(p) <= len(choices) for p in picks):
            return [choices[i] for i in dict.fromkeys(int(p) - 1 for p in picks)]
        print("  (numbers from the list, separated by commas)")


def _device_choices(kind: str, var: str, items: list[tuple[int, str]]) -> list[Choice]:
    """Menu over one device family, setting just that family's index var."""
    return [(f"{kind} {i}: {desc}", {var: str(i)}) for i, desc in items]


def _ollama_state() -> tuple[bool, str]:
    """Whether Ollama can actually answer, and what to say about it. Ask the
    daemon, don't ask `which ollama`: a Mac with the binary installed and the
    server down is the common case, and it fails as connection-refused on every
    tick rather than at setup time, which is where you'd want to hear it."""
    try:
        with request.urlopen(f"{_OLLAMA_HOST}/api/tags", timeout=0.5) as resp:
            models = json.loads(resp.read()).get("models", [])
    except (OSError, ValueError):  # URLError subclasses OSError
        if shutil.which("ollama"):
            return False, "not running; start it: ollama serve"
        return False, "not installed; see https://ollama.com"
    if not models:
        return False, "running, but no models pulled: ollama pull gemma3:4b"
    return True, f"ready, {len(models)} model(s) pulled"


GLANCE_VAR = "SACCADE_GLANCE_BACKEND"
FOCUS_VAR = "SACCADE_FOCUS_BACKEND"

_NO_MODEL = "No model: scripted demo output, nothing is called"


def _ordered(order: list[str], first: str = "", last: str = "") -> list[str]:
    """Same options every time, reordered so the sensible pick is [1]."""
    out = list(order)
    if last:
        out.append(out.pop(out.index(last)))
    if first:
        out.insert(0, out.pop(out.index(first)))
    return out


STT_VAR = "SACCADE_STT"


def _stt_state() -> tuple[bool, str]:
    """Whether local transcription can run here."""
    if _importable("faster_whisper"):
        return True, "ready"
    return False, "needs the stt extra"


def _stt_choices(stt: tuple[bool, str]) -> list[Choice]:
    """Where the audio gets understood. Leading with local isn't a preference:
    it's the only option where the microphone in your room doesn't become an
    upload, and it's also the one that frees you from the single backend that
    accepts audio at all."""
    ready, tag = stt
    return [
        (
            f"Transcribe on this machine: the audio never leaves, any model can read it ({tag})",
            {STT_VAR: "whisper"},
        ),
        ("Send the recording to the model: only Gemini accepts audio", {STT_VAR: ""}),
    ]


def _glance_choices(ollama: tuple[bool, str], hears_audio: bool = False) -> list[Choice]:
    """Glance is the tier that runs about once a second and looks at *every* frame,
    so this pick decides whether a camera pointed at your kitchen streams to a
    vendor all day. Local leads whenever it can actually run.

    Gemini takes the lead instead when the sensor captures audio, because it's the
    only backend that forwards it; anything else accepts the mic pick and then
    silently drops the audio half."""
    usable, tag = ollama
    labels = {
        "ollama": f"Ollama: runs on this machine, frames never leave it ({tag})",
        "gemini": "Gemini: hosted; every frame it looks at is uploaded (only one that hears audio)",
        "openai": "OpenAI: hosted; every frame it looks at is uploaded",
        "anthropic": "Anthropic: hosted; every frame it looks at is uploaded",
        "stub": _NO_MODEL,
    }
    order = _ordered(
        ["ollama", "gemini", "openai", "anthropic", "stub"],
        first="gemini" if hears_audio else "",
        last="" if usable else "ollama",  # don't lead with something they can't run
    )
    return [(labels[k], {GLANCE_VAR: k}) for k in order]


def _focus_choices(ollama: tuple[bool, str]) -> list[Choice]:
    """Focus only runs when Glance escalates, which is rare by design. That's why
    the expensive, capable model belongs here and not on the 1 Hz tier: you pay
    for it a few times a day, and the only thing it ever sees is a moment that
    already cleared the bar."""
    usable, tag = ollama
    sees = "hosted; sees only the moments Glance escalates"
    labels = {
        "gemini": f"Gemini: {sees}",
        "anthropic": f"Anthropic: {sees}",
        "openai": f"OpenAI: {sees}",
        "ollama": f"Ollama: runs on this machine, nothing leaves it ({tag})",
        "stub": _NO_MODEL,
    }
    order = _ordered(
        ["gemini", "anthropic", "openai", "ollama", "stub"], last="" if usable else "ollama"
    )
    return [(labels[k], {FOCUS_VAR: k}) for k in order]


KEY_VARS = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

# The import each hosted backend needs, so the wizard can notice a pick this
# machine can't actually run.
SDK_MODULES = {"gemini": "google.genai", "openai": "openai", "anthropic": "anthropic"}


def _missing_sdks(env: dict[str, str]) -> list[str]:
    """Extras the picks need but this interpreter can't import.

    Picking Gemini without the extra installed used to write a perfectly valid
    .env and then fail on every single tick with 'No module named google', which
    reads like saccade is broken rather than one install short."""
    missing: list[str] = []
    kinds = [env.get(GLANCE_VAR, "stub"), env.get(FOCUS_VAR, "stub")]
    if env.get("SACCADE_SPEAKER") == "gemini_tts":
        kinds.append("gemini")
    for kind in kinds:
        module = SDK_MODULES.get(kind)
        if module and kind not in missing and not _importable(module):
            missing.append(kind)
    return missing


def _mask(secret: str) -> str:
    """Enough to recognize a key, not enough to be one."""
    return f"...{secret[-4:]}" if len(secret) > 8 else "(short)"


def _ask_key(var: str) -> str:
    """Ask for a key, but look for one first.

    Sending someone to go fetch a credential they already exported is a pointless
    errand, and the usual outcome is a second key pasted next to the working one."""
    found = os.environ.get(var, "").strip()
    if found:
        ans = input(f"\nFound {var} in your environment ({_mask(found)}). Use it? [Y/n] ")
        if ans.strip().lower() in ("", "y", "yes"):
            return found
    return input(f"\n{var} (blank to set it later): ").strip()

_OUT_VAR = "SACCADE_AUDIO_OUT_INDEX"


def _importable(module: str) -> bool:
    """Whether *this* interpreter can import `module`, asked in a subprocess so
    we never load it into our own process (piper is GPL; we run it, not link it)."""
    probe = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True)
    return probe.returncode == 0


def _piper_state() -> tuple[bool, str]:
    """Whether Piper can speak, and what to say about it."""
    return (True, "local, free, no key") if _importable("piper") else (False, "not installed")


def _piper_setup_commands() -> str:
    """The two commands to get Piper working, aimed at *this* interpreter.

    Every shortcut here has already bitten someone on a Mac. A bare `python`
    isn't on PATH at all, so `python -m piper.download_voices` picks Homebrew's
    3.14 and reports 'No module named piper' while piper sits happily in .venv.
    And `python -m pip install` fails in a uv-made venv, which ships without pip
    (a different confusing error for the same user). So: name the interpreter,
    and ask which installer this environment actually has."""
    exe = sys.executable
    if _importable("pip"):
        install = f"{exe} -m pip install piper-tts"
    elif shutil.which("uv"):
        install = "uv pip install piper-tts"
    else:
        install = f"{exe} -m ensurepip --upgrade && {exe} -m pip install piper-tts"
    return f"    {install}\n    {exe} -m piper.download_voices en_US-lessac-medium\n"


def _speaker_choices(piper: tuple[bool, str]) -> list[Choice]:
    """Text, then the two ways to make sound. Which output device is a follow-up
    question, so adding a second engine doesn't multiply the menu by every
    speaker on the machine.

    Speaking is offered whether or not any output *devices* were enumerated.
    Listing them needs the audio extra (sounddevice); actually making a noise
    doesn't: SACCADE_PLAY_CMD hands the wav to afplay/aplay and the OS picks the
    default device. Gating on the device list denied a fresh install the one
    speaker that needs no key and no extra at all."""
    out: list[Choice] = [("Text in the terminal", {"SACCADE_SPEAKER": "print"})]
    play = "afplay" if sys.platform == "darwin" else "aplay"
    out.append(
        (
            f"Speak out loud: Piper ({piper[1]})",
            {"SACCADE_SPEAKER": "piper", "SACCADE_PLAY_CMD": play},
        )
    )
    out.append(
        (
            "Speak out loud: Gemini TTS (better voice, needs GEMINI_API_KEY)",
            {"SACCADE_SPEAKER": "gemini_tts", "SACCADE_PLAY_CMD": play},
        )
    )
    return out


def _missing_extras(devs: Devices, hints: tuple[str, str, str]) -> list[str]:
    """Extras worth installing given what the probes couldn't see. Only the
    ImportError hints in devices.py start with `uv pip install`; match that,
    not a bare "install", or the PortAudio hint ("apt install libportaudio2")
    gets misread as a missing Python package."""
    names = ("camera", "screen", "audio")
    found = (devs[0], devs[1], devs[2])
    return [
        name
        for name, items, hint in zip(names, found, hints, strict=True)
        if not items and hint.startswith("uv pip install")
    ]


def _notes(hints: tuple[str, str, str]) -> list[str]:
    """Hints worth printing after the extras block. That block already covered the
    `uv pip install` ones, so match on that prefix rather than a bare "install",
    or the PortAudio hint ("apt install libportaudio2") gets swallowed and the
    user is left with no audio devices and nothing to act on."""
    return [hint for hint in hints if hint and not hint.startswith("uv pip install")]


def _needed_keys(env: dict[str, str]) -> list[str]:
    """Key vars the picks actually require. The backend and the speaker are
    separate asks: thinking with OpenAI while speaking with Gemini TTS needs both,
    and prompting for only the backend's key wrote an .env whose speaker failed on
    the first spoken word."""
    needed: list[str] = []
    for var in (GLANCE_VAR, FOCUS_VAR):
        key = KEY_VARS.get(env.get(var, "stub"))
        if key and key not in needed:
            needed.append(key)
    if env.get("SACCADE_SPEAKER") == "gemini_tts" and KEY_VARS["gemini"] not in needed:
        needed.append(KEY_VARS["gemini"])
    return needed


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
        # .env has no suffix to replace (it's all stem), so with_suffix would
        # make ".env.env.bak"; append instead.
        backup = path.with_name(path.name + ".bak")
        shutil.copyfile(path, backup)
        print(f"  (backed up to {backup})")
    body = "\n".join(f"{k}={v}" for k, v in env.items())
    path.write_text(f"# Written by `python -m saccade setup`. Edit freely.\n{body}\n")
    return True


def main() -> None:
    if not sys.stdin.isatty():
        print(
            "setup is interactive and stdin isn't a terminal.\n"
            f"Run `{sys.executable} -m saccade devices` and set the env vars yourself.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print("\nsaccade setup: probing this machine...")
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
    for hint in _notes((cam_hint, screen_hint, audio_hint)):
        print(f"  note: {hint}")

    ollama = _ollama_state()
    env: dict[str, str] = {}
    env.update(_ask("What should saccade watch or hear?", _sensor_choices(devs)))
    if env.get("SACCADE_SENSOR") == "multi":
        picked = _ask_many("Which ones?", _one_sensor_choices(devs))
        merged, dropped = _merge_sensors(picked)
        for label in dropped:
            print(f"  note: skipped {label} (only one of each kind for now)")
        env.update(merged)
    if env.get("SACCADE_SENSOR") == "av":
        env.update(_ask("Which camera?", _device_choices("Camera", "SACCADE_WEBCAM_INDEX", cams)))
        env.update(_ask("Which mic?", _device_choices("Mic", "SACCADE_MIC_INDEX", mics)))
    # Two tiers, two questions. One question set both, which quietly threw away
    # the whole point of the split: cheap eyes that run constantly, an expensive
    # brain that runs almost never.
    stt = _stt_state()
    if {"mic", "av"} & _sensor_kinds(env):
        env.update(_ask("What should happen to what it hears?", _stt_choices(stt)))
    # Only *raw* audio pins you to Gemini. Once it's transcribed here, the model
    # is reading text and every backend is back on the table.
    hears_audio = bool({"mic", "av"} & _sensor_kinds(env)) and env.get(STT_VAR) != "whisper"
    env.update(
        _ask(
            "What keeps watching?  (runs ~1x/sec, looks at every frame)",
            _glance_choices(ollama, hears_audio),
        )
    )
    env.update(
        _ask(
            "What thinks when something happens?  (runs only when the watcher escalates)",
            _focus_choices(ollama),
        )
    )
    piper = _piper_state()
    env.update(_ask("How should saccade answer?", _speaker_choices(piper)))
    if env.get("SACCADE_SPEAKER") in ("piper", "gemini_tts") and outs:
        env.update(_ask("Out of which speaker?", _device_choices("Output", _OUT_VAR, outs)))

    if "ollama" in (env.get(GLANCE_VAR), env.get(FOCUS_VAR)) and not ollama[0]:
        print(f"\n  Heads up: Ollama is {ollama[1]}\n  saccade will keep retrying until it's up.")
    if env.get(STT_VAR) == "whisper" and not stt[0]:
        print(
            "\n  Local transcription isn't installed yet:\n\n"
            "    uv pip install -e '.[stt]'\n"
        )
    if env.get("SACCADE_SPEAKER") == "piper" and not piper[0]:
        print(f"\n  Piper isn't installed yet. Two commands and it can talk:\n\n{_piper_setup_commands()}")

    for extra in _missing_sdks(env):
        print(
            f"\n  The {extra} backend needs its SDK, which isn't installed here:\n"
            f"\n    uv pip install -e '.[{extra}]'\n"
            f"\n  Without it every tick fails with 'No module named ...'.\n"
        )

    for var in _needed_keys(env):
        key = _ask_key(var)
        if key:
            env[var] = key

    if not _write_env(Path(".env"), env):
        return

    # sys.executable, not a bare `python`: the shell that just ran setup may have
    # no `python` on it at all (macOS), and the one it does have is often not the
    # venv holding saccade's deps.
    print(f"\nWrote .env. Start it with:\n\n    {sys.executable} -m saccade\n")
    if {"webcam", "av"} & _sensor_kinds(env) and sys.platform == "darwin":
        print("macOS: approve the Camera prompt on first run, then rerun.\n")
