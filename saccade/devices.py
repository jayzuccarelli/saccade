"""Discover local peripherals: cameras, screens, microphones, audio outputs.

    python -m saccade devices

Each probe is lazy and self-contained — a missing dep or a failing subsystem
prints a hint for that section instead of crashing the command. The `.env
lines:` under each section paste verbatim into `.env`.
"""

from __future__ import annotations

import contextlib
import os
import sys


@contextlib.contextmanager
def _quiet_stderr():
    """OpenCV's C layer logs camera-probe noise straight to fd 2 — silence it
    around the probe so the device table stays readable."""
    fd = os.dup(2)
    try:
        with open(os.devnull, "wb") as null:
            os.dup2(null.fileno(), 2)
        yield
    finally:
        os.dup2(fd, 2)
        os.close(fd)


def _cameras() -> tuple[list[tuple[int, str]], str]:
    """Probe cv2 indices upward. Names aren't cross-platform via cv2, so show
    resolution as the tell (e.g. built-in is often 1280x720, USB may differ)."""
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
            "no camera opened — grant Camera access to your terminal app "
            "(System Settings > Privacy & Security > Camera), then rerun"
        )
    return out, ""


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
    except Exception as e:  # headless box, no $DISPLAY, ... — keep the other sections
        return [], f"screen probe failed: {e}"


def _audio() -> tuple[list[tuple[int, str]], list[tuple[int, str]], str]:
    try:
        import sounddevice as sd
    except ImportError:
        return [], [], "uv pip install -e '.[audio]'  # sounddevice"
    except OSError:
        # sounddevice imports but PortAudio (the C lib) is missing — bundled in
        # the Mac/Windows wheels, apt/brew territory on Linux.
        return [], [], "PortAudio not found — `sudo apt install libportaudio2` (Linux)"
    try:
        devs = sd.query_devices()
    except Exception as e:  # no PortAudio backend / no audio subsystem
        return [], [], f"audio probe failed: {e}"
    mics = [(i, d["name"]) for i, d in enumerate(devs) if d["max_input_channels"] > 0]
    outs = [(i, d["name"]) for i, d in enumerate(devs) if d["max_output_channels"] > 0]
    return mics, outs, ""


def _print(title: str, items: list[tuple[int, str]], hint: str, env_lines: tuple = ()) -> None:
    print(f"\n{title}")
    if hint:
        print(f"  (none — {hint})")
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
    _print("Mics (audio sensor lands in v0.2 — Frame is image-shaped today)", mics, audio_hint)
    play = "afplay" if sys.platform == "darwin" else "aplay"
    _print(
        "Audio out (played via SACCADE_PLAY_CMD; per-device routing = v0.2)",
        outs,
        audio_hint,
        (f"SACCADE_PLAY_CMD={play}",),
    )
    print()


if __name__ == "__main__":
    main()
