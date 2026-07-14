"""The setup wizard turns probed devices into .env lines. Probing needs
hardware, but the menu-building — which devices become which env vars — is the
contract worth pinning, with the device lists faked."""

from pathlib import Path

from saccade.setup import (
    _backend_choices,
    _missing_extras,
    _sensor_choices,
    _speaker_choices,
    _write_env,
)

CAMS = [(0, "1280x720"), (1, "1920x1080")]
SCREENS = [(1, "2560x1440 at (0,0)")]
MICS = [(0, "Built-in Microphone")]
OUTS = [(1, "Built-in Speakers")]

# Hint shapes devices.py actually emits.
IMPORT_HINT = "uv pip install -e '.[camera]'  # opencv-python-headless + pillow"
PORTAUDIO_HINT = "PortAudio not found — `sudo apt install libportaudio2` (Linux)"
DISPLAY_HINT = "screen probe failed: Cannot connect to display"


def _envs(choices):
    return [env for _, env in choices]


def test_each_camera_is_separately_pickable():
    """Pick the built-in *or* the external cam — so each camera is its own
    entry carrying its own index, not one 'webcam' bucket."""
    envs = _envs(_sensor_choices((CAMS, [], [])))
    assert {"SACCADE_SENSOR": "webcam", "SACCADE_WEBCAM_INDEX": "0"} in envs
    assert {"SACCADE_SENSOR": "webcam", "SACCADE_WEBCAM_INDEX": "1"} in envs


def test_camera_plus_mic_offers_the_av_sensor():
    envs = _envs(_sensor_choices((CAMS, [], MICS)))
    assert {
        "SACCADE_SENSOR": "av",
        "SACCADE_WEBCAM_INDEX": "0",
        "SACCADE_MIC_INDEX": "0",
    } in envs


def test_no_av_option_without_both_devices():
    for devs in ((CAMS, [], []), ([], [], MICS)):
        assert not any(e.get("SACCADE_SENSOR") == "av" for e in _envs(_sensor_choices(devs)))


def test_stub_is_always_offered_as_a_way_out():
    for devs in ((CAMS, SCREENS, MICS), ([], [], [])):
        assert _envs(_sensor_choices(devs))[-1] == {"SACCADE_SENSOR": "stub"}


def test_portaudio_hint_is_not_a_missing_extra():
    """Regression: sounddevice is installed, the C library isn't. Telling the
    user to `uv pip install -e '.[audio]'` fixes nothing."""
    assert _missing_extras(([], [], []), ("", "", PORTAUDIO_HINT)) == []


def test_display_failure_is_not_a_missing_extra():
    assert _missing_extras(([], [], []), ("", DISPLAY_HINT, "")) == []


def test_importerror_hint_is_a_missing_extra():
    assert _missing_extras(([], [], []), (IMPORT_HINT, "", "")) == ["camera"]


def test_found_devices_are_never_reported_missing():
    assert _missing_extras((CAMS, [], []), (IMPORT_HINT, "", "")) == []


def test_speaker_offers_text_first_then_each_output():
    choices = _speaker_choices(OUTS)
    assert choices[0][1] == {"SACCADE_SPEAKER": "print"}
    assert _envs(choices)[1]["SACCADE_AUDIO_OUT_INDEX"] == "1"


def test_speaker_is_text_only_with_no_outputs():
    assert _envs(_speaker_choices([])) == [{"SACCADE_SPEAKER": "print"}]


def test_every_backend_choice_sets_both_tiers():
    for env in _envs(_backend_choices()):
        assert env["SACCADE_GLANCE_BACKEND"] == env["SACCADE_FOCUS_BACKEND"]


def test_write_env_emits_loadable_lines(tmp_path: Path):
    path = tmp_path / ".env"
    assert _write_env(path, {"SACCADE_SENSOR": "webcam", "SACCADE_WEBCAM_INDEX": "1"})
    body = path.read_text()
    assert "SACCADE_SENSOR=webcam" in body
    assert "SACCADE_WEBCAM_INDEX=1" in body
