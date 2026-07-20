"""The setup wizard turns probed devices into .env lines. Probing needs
hardware, but the menu-building — which devices become which env vars — is the
contract worth pinning, with the device lists faked."""

from pathlib import Path

from saccade.setup import (
    _backend_choices,
    _device_choices,
    _missing_extras,
    _needed_keys,
    _notes,
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
    """The av entry names no indices — which camera and which mic are asked
    next. Defaulting to the first of each pairs a MacBook's webcam with the
    user's iPhone microphone, which is nobody's intent."""
    assert {"SACCADE_SENSOR": "av"} in _envs(_sensor_choices((CAMS, [], MICS)))


def test_device_choices_set_only_their_own_index():
    choices = _device_choices("Mic", "SACCADE_MIC_INDEX", MICS)
    assert choices == [("Mic 0 — Built-in Microphone", {"SACCADE_MIC_INDEX": "0"})]


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
    for env in _envs(_backend_choices((True, "ready"))):
        assert env["SACCADE_GLANCE_BACKEND"] == env["SACCADE_FOCUS_BACKEND"]


def test_ollama_leads_only_when_it_can_answer():
    """A reachable daemon earns the default slot; an installed-but-dead one
    doesn't — picking it is connection-refused on every tick forever."""
    ready = _backend_choices((True, "ready"))
    assert ready[0][1]["SACCADE_GLANCE_BACKEND"] == "ollama"
    dead = _backend_choices((False, "not running — start it: ollama serve"))
    assert dead[0][1]["SACCADE_GLANCE_BACKEND"] != "ollama"
    assert any(e["SACCADE_GLANCE_BACKEND"] == "ollama" for e in _envs(dead))


def test_backend_tag_is_shown_in_the_label():
    label = _backend_choices((False, "not running — start it: ollama serve"))[-1][0]
    assert "ollama serve" in label


def test_backup_is_env_bak_not_env_env_bak(tmp_path: Path, monkeypatch):
    """Regression: Path('.env').with_suffix('.env.bak') gives '.env.env.bak' —
    a dotfile is all stem, so there's no suffix to replace."""
    path = tmp_path / ".env"
    path.write_text("SACCADE_SENSOR=stub\n")
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert _write_env(path, {"SACCADE_SENSOR": "webcam"})
    assert (tmp_path / ".env.bak").read_text() == "SACCADE_SENSOR=stub\n"
    assert not (tmp_path / ".env.env.bak").exists()


def test_write_env_emits_loadable_lines(tmp_path: Path):
    path = tmp_path / ".env"
    assert _write_env(path, {"SACCADE_SENSOR": "webcam", "SACCADE_WEBCAM_INDEX": "1"})
    body = path.read_text()
    assert "SACCADE_SENSOR=webcam" in body
    assert "SACCADE_WEBCAM_INDEX=1" in body


def test_portaudio_hint_survives_the_note_filter():
    """It says "apt install libportaudio2", so a bare "install" match swallowed it
    and left a Linux box with no audio devices and nothing to act on."""
    assert PORTAUDIO_HINT in _notes(("", "", PORTAUDIO_HINT))


def test_extras_hints_are_not_repeated_as_notes():
    """The extras block already printed the `uv pip install` line."""
    assert _notes((IMPORT_HINT, "", "")) == []


def test_display_failure_is_still_a_note():
    assert DISPLAY_HINT in _notes(("", DISPLAY_HINT, ""))


def test_audio_sensor_leads_with_the_backend_that_hears():
    """Gemini is the only backend that forwards Frame.audio, so accepting the
    default with a mic selected must not hand you one that drops it."""
    heard = _backend_choices((True, "ready"), hears_audio=True)
    assert heard[0][1]["SACCADE_GLANCE_BACKEND"] == "gemini"


def test_video_only_still_leads_with_ollama():
    seen = _backend_choices((True, "ready"), hears_audio=False)
    assert seen[0][1]["SACCADE_GLANCE_BACKEND"] == "ollama"


def test_promoting_gemini_keeps_every_backend_on_the_menu():
    heard = _backend_choices((True, "ready"), hears_audio=True)
    assert len(heard) == len(_backend_choices((True, "ready")))
    assert len(_envs(heard)) == len({e["SACCADE_GLANCE_BACKEND"] for e in _envs(heard)})


def test_hosted_backend_plus_gemini_tts_asks_for_both_keys():
    """Thinking with OpenAI and speaking with Gemini needs two keys; asking only
    for the backend's wrote an .env whose speaker died on the first word."""
    env = {"SACCADE_GLANCE_BACKEND": "openai", "SACCADE_SPEAKER": "gemini_tts"}
    assert _needed_keys(env) == ["OPENAI_API_KEY", "GEMINI_API_KEY"]


def test_gemini_backend_and_gemini_tts_ask_once():
    env = {"SACCADE_GLANCE_BACKEND": "gemini", "SACCADE_SPEAKER": "gemini_tts"}
    assert _needed_keys(env) == ["GEMINI_API_KEY"]


def test_local_backend_printing_text_needs_no_key():
    assert _needed_keys({"SACCADE_GLANCE_BACKEND": "ollama", "SACCADE_SPEAKER": "print"}) == []


def test_local_backend_speaking_still_needs_the_tts_key():
    env = {"SACCADE_GLANCE_BACKEND": "ollama", "SACCADE_SPEAKER": "gemini_tts"}
    assert _needed_keys(env) == ["GEMINI_API_KEY"]
