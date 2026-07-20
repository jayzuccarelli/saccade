"""Probing needs hardware, but naming what was probed is pure logic, and it's
where the menu earns its keep. A Mac with Continuity Camera reports the built-in
webcam and the iPhone as 1920x1080 alike, so without names you get two identical
rows and no way to pick the one you meant."""

import json
import subprocess
from types import SimpleNamespace

from saccade.devices import _label_cameras, _mac_camera_names

FOUND = [(0, "1920x1080"), (1, "1920x1080")]
NAMES = ["FaceTime HD Camera", "Desk iPhone Camera"]

# Shape system_profiler -json emits: nested item names come back as `_name`.
PAYLOAD = json.dumps(
    {
        "SPCameraDataType": [
            {"_name": "FaceTime HD Camera", "spcamera_model-id": "FaceTime HD Camera"},
            {"_name": "Desk iPhone Camera", "spcamera_model-id": "iPhone00,0"},
        ]
    }
)


def test_names_are_paired_with_indices_in_order():
    assert _label_cameras(FOUND, NAMES) == [
        (0, "FaceTime HD Camera (1920x1080)"),
        (1, "Desk iPhone Camera (1920x1080)"),
    ]


def test_mismatched_counts_fall_back_to_resolution():
    """If cv2 and the OS disagree on how many cameras exist, which name goes with
    which index is a guess, and a confidently wrong one sends you to the wrong
    camera. Say less instead."""
    assert _label_cameras(FOUND, ["Only One Camera"]) == FOUND


def test_no_names_falls_back_to_resolution():
    assert _label_cameras(FOUND, []) == FOUND


def test_mac_camera_names_parses_system_profiler(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=PAYLOAD))
    assert _mac_camera_names() == NAMES


def test_camera_names_are_empty_off_mac(monkeypatch):
    """No system_profiler outside macOS, and the fallback keeps the menu working.
    Pin the platform: unpinned, this passes on Linux and shells out for real on
    the macOS CI leg."""
    monkeypatch.setattr("sys.platform", "linux")
    assert _mac_camera_names() == []


def test_system_profiler_failure_is_not_fatal(monkeypatch):
    def boom(*a, **k):
        raise subprocess.SubprocessError("nope")

    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr(subprocess, "run", boom)
    assert _mac_camera_names() == []
