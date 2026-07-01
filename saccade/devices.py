"""Discover local peripherals: cameras, screens, microphones, audio outputs.

    python -m saccade devices

Each probe uses a lazy import — if the optional dep is missing, prints an
install hint instead of crashing. Copy the printed env line straight into `.env`.
"""

from __future__ import annotations


def _cameras() -> tuple[list[tuple[int, str]], str]:
    """Probe cv2 indices 0..5. Names aren't cross-platform via cv2, so show
    resolution as the tell (e.g. built-in is often 1280x720, USB may differ)."""
    try:
        import cv2
    except ImportError:
        return [], "uv pip install saccade[camera]  # opencv-python-headless"
    out: list[tuple[int, str]] = []
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out.append((i, f"{w}x{h}" if w and h else "unknown resolution"))
        cap.release()
    return out, ""


def _screens() -> tuple[list[tuple[int, str]], str]:
    try:
        import mss
    except ImportError:
        return [], "uv pip install saccade[screen]  # mss"
    with mss.mss() as sct:
        # sct.monitors[0] is the combined virtual screen; [1:] are individual monitors.
        return [
            (i, f"{m['width']}x{m['height']} at ({m['left']},{m['top']})")
            for i, m in enumerate(sct.monitors[1:], start=1)
        ], ""


def _audio() -> tuple[list[tuple[int, str]], list[tuple[int, str]], str]:
    try:
        import sounddevice as sd
    except ImportError:
        return [], [], "uv pip install saccade[audio]  # sounddevice"
    devs = sd.query_devices()
    mics = [(i, d["name"]) for i, d in enumerate(devs) if d["max_input_channels"] > 0]
    outs = [(i, d["name"]) for i, d in enumerate(devs) if d["max_output_channels"] > 0]
    return mics, outs, ""


def _print(title: str, items: list[tuple[int, str]], hint: str) -> None:
    print(f"\n{title}")
    if hint:
        print(f"  (none — {hint})")
        return
    if not items:
        print("  (none found)")
        return
    for idx, name in items:
        print(f"  [{idx}] {name}")


def main() -> None:
    cams, cam_hint = _cameras()
    screens, screen_hint = _screens()
    mics, outs, audio_hint = _audio()

    _print("Cameras   (SACCADE_SENSOR=webcam SACCADE_WEBCAM_INDEX=N)", cams, cam_hint)
    _print("Screens   (SACCADE_SENSOR=screen SACCADE_SCREEN_INDEX=N)", screens, screen_hint)
    _print("Mics      (audio sensor coming in v0.2 — schema is image-shaped today)", mics, audio_hint)
    _print("Audio out (SACCADE_PLAY_CMD=afplay for OS default; per-device = v0.2)", outs, audio_hint)
    print()


if __name__ == "__main__":
    main()
