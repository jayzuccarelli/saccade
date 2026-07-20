"""Discover local peripherals: cameras, screens, microphones, audio outputs.

    python -m saccade devices

Each probe is lazy and self-contained: a missing dep or a failing subsystem
prints a hint for that section instead of crashing the command. The `.env
lines:` under each section paste verbatim into `.env`.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator


@contextlib.contextmanager
def _quiet_stderr() -> Iterator[None]:
    """OpenCV's C layer logs camera-probe noise straight to fd 2; silence it
    around the probe so the device table stays readable."""
    fd = os.dup(2)
    try:
        with open(os.devnull, "wb") as null:
            os.dup2(null.fileno(), 2)
        yield
    finally:
        os.dup2(fd, 2)
        os.close(fd)


def _mac_camera_names() -> list[str]:
    """Camera names from system_profiler, in AVFoundation order. cv2 exposes no
    name API, and a Mac with Continuity Camera reports the built-in and the
    iPhone as 1920x1080 alike; with only resolution to go on the menu is two
    identical rows and you can't pick. Empty on anything unexpected; the caller
    then falls back to resolution."""
    if sys.platform != "darwin":
        return []
    try:
        raw = subprocess.run(
            ["system_profiler", "-json", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        cams = json.loads(raw).get("SPCameraDataType", [])
        return [str(c["_name"]) for c in cams if c.get("_name")]
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return []


def _label_cameras(found: list[tuple[int, str]], names: list[str]) -> list[tuple[int, str]]:
    """Pair probed indices with names, positionally: cv2 and system_profiler
    both enumerate AVFoundation, so the orders line up. Only when the counts
    agree: if they don't, the index->name mapping is unknowable, and a
    confidently wrong name ("that's my Mac's camera") is worse than a bare
    resolution."""
    if not names or len(names) != len(found):
        return found
    return [(i, f"{n} ({desc})") for (i, desc), n in zip(found, names, strict=True)]


def _cameras() -> tuple[list[tuple[int, str]], str]:
    """Probe cv2 indices upward, naming them via the OS where we can."""
    try:
        import cv2
    except ImportError:
        return [], "uv pip install -e '.[camera]'  # opencv-python-headless + pillow"
    out: list[tuple[int, str]] = []
    misses = 0
    with _quiet_stderr():
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out.append((i, f"{w}x{h}" if w and h else "unknown resolution"))
                misses = 0
            else:
                misses += 1
            cap.release()
            if misses >= 2:
                break  # indices are contiguous in practice; stop probing air
    if not out and sys.platform == "darwin":
        return [], (
            "no camera opened; grant Camera access to your terminal app "
            "(System Settings > Privacy & Security > Camera), then rerun"
        )
    return _label_cameras(out, _mac_camera_names()), ""


def _screens() -> tuple[list[tuple[int, str]], str]:
    try:
        import mss
    except ImportError:
        return [], "uv pip install -e '.[screen]'  # mss + pillow"
    try:
        with mss.mss() as sct:
            # sct.monitors[0] is the combined virtual screen; [1:] are individual monitors.
            return [
                (i, f"{m['width']}x{m['height']} at ({m['left']},{m['top']})")
                for i, m in enumerate(sct.monitors[1:], start=1)
            ], ""
    except Exception as e:  # headless box, no $DISPLAY, ...; keep the other sections
        return [], f"screen probe failed: {e}"


def _audio() -> tuple[list[tuple[int, str]], list[tuple[int, str]], str]:
    try:
        import sounddevice as sd
    except ImportError:
        return [], [], "uv pip install -e '.[audio]'  # sounddevice"
    except OSError:
        # sounddevice imports but PortAudio (the C lib) is missing: bundled in
        # the Mac/Windows wheels, apt/brew territory on Linux.
        return [], [], "PortAudio not found: `sudo apt install libportaudio2` (Linux)"
    try:
        devs = sd.query_devices()
    except Exception as e:  # no PortAudio backend / no audio subsystem
        return [], [], f"audio probe failed: {e}"
    mics = [(i, d["name"]) for i, d in enumerate(devs) if d["max_input_channels"] > 0]
    outs = [(i, d["name"]) for i, d in enumerate(devs) if d["max_output_channels"] > 0]
    return mics, outs, ""


def _print(
    title: str, items: list[tuple[int, str]], hint: str, env_lines: tuple[str, ...] = ()
) -> None:
    print(f"\n{title}")
    if hint:
        print(f"  (none: {hint})")
        return
    if not items:
        print("  (none found)")
        return
    for idx, name in items:
        print(f"  [{idx}] {name}")
    if env_lines:
        print("  # .env lines:")
        for line in env_lines:
            print(f"  {line}")


def main() -> None:
    cams, cam_hint = _cameras()
    screens, screen_hint = _screens()
    mics, outs, audio_hint = _audio()

    _print(
        "Cameras",
        cams,
        cam_hint,
        ("SACCADE_SENSOR=webcam", f"SACCADE_WEBCAM_INDEX={cams[0][0] if cams else 0}"),
    )
    _print(
        "Screens",
        screens,
        screen_hint,
        ("SACCADE_SENSOR=screen", f"SACCADE_SCREEN_INDEX={screens[0][0] if screens else 1}"),
    )
    if screens and sys.platform == "darwin":
        print(
            "  # black captures? grant Screen Recording to your terminal app"
            " (System Settings > Privacy & Security)"
        )
    _print(
        "Mics",
        mics,
        audio_hint,
        ("SACCADE_SENSOR=mic", f"SACCADE_MIC_INDEX={mics[0][0] if mics else 0}"),
    )
    if mics:
        print("  # hearing needs an audio-capable backend: SACCADE_GLANCE_BACKEND=gemini")
    play = "afplay" if sys.platform == "darwin" else "aplay"
    _print(
        "Audio out",
        outs,
        audio_hint,
        (
            f"SACCADE_PLAY_CMD={play}  # OS default device, or:",
            f"SACCADE_AUDIO_OUT_INDEX={outs[0][0] if outs else 0}  # play to a specific device",
        ),
    )
    print()


if __name__ == "__main__":
    main()
