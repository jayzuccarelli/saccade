"""The setup wizard turns probed devices into .env lines. Probing needs
hardware, but the menu-building (which devices become which env vars) is the
contract worth pinning, with the device lists faked."""

import sys
from pathlib import Path

import pytest

from saccade import setup as setuplib
from saccade.setup import (
    FOCUS_VAR,
    GLANCE_VAR,
    _device_choices,
    _focus_choices,
    _glance_choices,
    _merge_sensors,
    _missing_extras,
    _needed_keys,
    _notes,
    _one_sensor_choices,
    _pick_sensors,
    _piper_setup_commands,
    _sensor_choices,
    _sensor_kinds,
    _speaker_choices,
    _write_env,
)

CAMS = [(0, "1280x720"), (1, "1920x1080")]
SCREENS = [(1, "2560x1440 at (0,0)")]
MICS = [(0, "Built-in Microphone")]
OUTS = [(1, "Built-in Speakers")]

# Hint shapes devices.py actually emits.
IMPORT_HINT = "uv pip install -e '.[camera]'  # opencv-python-headless + pillow"
PORTAUDIO_HINT = "PortAudio not found: `sudo apt install libportaudio2` (Linux)"
DISPLAY_HINT = "screen probe failed: Cannot connect to display"


def _envs(choices):
    return [env for _, env in choices]


def test_each_camera_is_separately_pickable():
    """Pick the built-in *or* the external cam, so each camera is its own
    entry carrying its own index, not one 'webcam' bucket."""
    envs = _envs(_sensor_choices((CAMS, [], [])))
    assert {"SACCADE_SENSOR": "webcam", "SACCADE_WEBCAM_INDEX": "0"} in envs
    assert {"SACCADE_SENSOR": "webcam", "SACCADE_WEBCAM_INDEX": "1"} in envs


def test_a_camera_and_a_mic_fuse_without_being_asked():
    """Picking a camera and a mic means AVSensor, where a frame is what it saw
    and heard at the same instant. The menu used to ask this directly, as "a
    camera and a mic together" sitting next to "several at once": two phrases
    for the same intent, so the right answer depended on guessing our internal
    names. Combining is what picking several *means*."""
    cam, mic = _one_sensor_choices((CAMS, [], MICS))[0], _one_sensor_choices((CAMS, [], MICS))[-1]
    env, dropped, note = _pick_sensors([cam, mic])
    assert env["SACCADE_SENSOR"] == "av"
    assert env["SACCADE_WEBCAM_INDEX"] == "0" and env["SACCADE_MIC_INDEX"] == "0"
    assert not dropped
    assert "fused" in note


def test_any_other_combination_interleaves():
    """Three inputs, or two that can't fuse, run as independent streams."""
    env, _, note = _pick_sensors(_one_sensor_choices(([], SCREENS, MICS)))
    assert set(env["SACCADE_SENSOR"].split(",")) == {"screen", "mic"}
    assert env["SACCADE_SCREEN_INDEX"] == "1" and env["SACCADE_MIC_INDEX"] == "0"
    assert "own rate" in note


def test_a_single_pick_is_just_that_sensor():
    """One device shouldn't drag in MultiSensor or a note explaining itself."""
    env, _, note = _pick_sensors(_one_sensor_choices(([], SCREENS, [])))
    assert env["SACCADE_SENSOR"] == "screen"
    assert note == ""


def test_picking_the_demo_alongside_real_hardware_drops_the_demo():
    """"Nothing: scripted demo" is a way out, not an input to mix in."""
    picks = _one_sensor_choices((CAMS, [], [])) + [("Nothing", {"SACCADE_SENSOR": "stub"})]
    env, _, _ = _pick_sensors(picks)
    assert env["SACCADE_SENSOR"] == "webcam"


def test_picking_only_the_demo_gives_the_stub():
    env, _, _ = _pick_sensors([("Nothing", {"SACCADE_SENSOR": "stub"})])
    assert env["SACCADE_SENSOR"] == "stub"


def test_device_choices_set_only_their_own_index():
    choices = _device_choices("Mic", "SACCADE_MIC_INDEX", MICS)
    assert choices == [("Mic 0: Built-in Microphone", {"SACCADE_MIC_INDEX": "0"})]


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


PIPER_READY = (True, "local, free, no key")
PIPER_MISSING = (False, "not installed")


def test_piper_setup_commands_name_this_interpreter():
    """Regression, hit on a real Mac: the docs said `python -m piper.download_voices`.
    There is no `python` on PATH on macOS, so that ran Homebrew's 3.14 and reported
    'No module named piper' while piper sat happily in .venv."""
    cmds = _piper_setup_commands()
    assert f"{sys.executable} -m piper.download_voices" in cmds
    assert "\n    python -m" not in cmds


def test_install_line_matches_the_installer_this_venv_has(monkeypatch):
    """Second half of the same regression: a uv-made venv ships *without* pip, so
    telling that user `python -m pip install` swaps one confusing error for
    another. Ask what's here before giving an instruction."""
    monkeypatch.setattr(setuplib, "_importable", lambda mod: mod != "pip")
    monkeypatch.setattr(setuplib.shutil, "which", lambda name: "/usr/bin/uv")
    assert "uv pip install piper-tts" in _piper_setup_commands()

    monkeypatch.setattr(setuplib, "_importable", lambda mod: True)
    assert f"{sys.executable} -m pip install piper-tts" in _piper_setup_commands()


def test_speaking_out_loud_defaults_to_local_tts():
    """Text first, then Piper, then the hosted upgrade. An ambient agent that
    can't make a sound without a hosted API key is the wrong default: Gemini TTS
    should be the nicer voice you opt into, not the toll booth for any audio."""
    speakers = [e["SACCADE_SPEAKER"] for e in _envs(_speaker_choices(PIPER_READY))]
    assert speakers == ["print", "piper", "gemini_tts"]


def test_no_speaker_choice_names_a_device():
    """Which output is asked separately, so a second engine doesn't multiply the
    menu by every speaker on the machine."""
    for env in _envs(_speaker_choices(PIPER_READY)):
        assert "SACCADE_AUDIO_OUT_INDEX" not in env


def test_piper_state_is_shown_in_the_label():
    label = _speaker_choices(PIPER_MISSING)[1][0]
    assert "not installed" in label


def test_speech_is_offered_without_the_audio_extra():
    """Found by review. Listing output *devices* needs sounddevice; making a noise
    doesn't; SACCADE_PLAY_CMD hands the wav to afplay/aplay. Gating the menu on
    the device list denied a fresh install the one speaker that needs no key and
    no extra, which is exactly the speaker this PR added."""
    speakers = [e["SACCADE_SPEAKER"] for e in _envs(_speaker_choices(PIPER_READY))]
    assert speakers == ["print", "piper", "gemini_tts"]


def test_each_tier_sets_only_its_own_backend():
    """The two tiers are picked separately. One question that wrote both threw away
    the entire point of the split: you could not ask for cheap local eyes and an
    expensive hosted brain, which is the whole design."""
    for env in _envs(_glance_choices((True, "ready", ""))):
        assert list(env) == ["SACCADE_GLANCE_BACKEND"]
    for env in _envs(_focus_choices((True, "ready", ""))):
        assert list(env) == ["SACCADE_FOCUS_BACKEND"]


def test_the_recommended_pair_is_local_eyes_hosted_brain():
    """Accepting both defaults should land on the architecture: a cheap model
    watching continuously on this machine, a capable hosted one that only ever
    sees what already escalated."""
    assert _glance_choices((True, "ready", ""))[0][1]["SACCADE_GLANCE_BACKEND"] == "ollama"
    assert _focus_choices((True, "ready", ""))[0][1]["SACCADE_FOCUS_BACKEND"] == "gemini"


def test_glance_says_what_leaves_the_machine():
    """The privacy claim has to be legible at the point of choosing, since Glance
    is the tier that sees every frame all day."""
    labels = dict(
        (label, env["SACCADE_GLANCE_BACKEND"]) for label, env in _glance_choices((True, "ready", ""))
    )
    local = next(lbl for lbl, k in labels.items() if k == "ollama")
    hosted = next(lbl for lbl, k in labels.items() if k == "gemini")
    assert "never leave" in local
    assert "uploaded" in hosted


def test_a_stopped_ollama_still_leads_glance():
    """It used to lose the lead when the daemon was down, which steered the
    machine with nothing running yet (the one most in need of the local pick)
    hardest toward uploading every frame. `ollama serve` is a fixable state, not
    a reason to recommend a vendor."""
    dead = _glance_choices((False, "not running", "Start it with:  ollama serve"))
    assert dead[0][1]["SACCADE_GLANCE_BACKEND"] == "ollama"
    assert "recommended" in dead[0][0]


def test_the_label_says_the_state_and_leaves_the_shell_out_of_it():
    """Recommending a stopped daemon is only honest if the label says it's stopped.
    It used to append the fix too, so a menu row ended in `ollama serve`: a command
    offered at the one moment you can't run it, since you're inside the menu."""
    label = next(
        lbl
        for lbl, env in _glance_choices((False, "not running", "Start it with:  ollama serve"))
        if env["SACCADE_GLANCE_BACKEND"] == "ollama"
    )
    assert "not running" in label
    assert "ollama serve" not in label


def test_both_tiers_name_their_recommendation():
    """Ordering alone is invisible: a menu sorted by preference looks exactly like
    one sorted arbitrarily. Jay picked Gemini off a list whose first entry we
    intended as the recommendation and never said so."""
    assert "recommended" in _glance_choices((True, "ready", ""))[0][0]
    assert "recommended" in _focus_choices((True, "ready", ""))[0][0]
    # Exactly one, or it isn't a recommendation.
    assert sum("recommended" in lbl for lbl, _ in _glance_choices((True, "ready", ""))) == 1
    assert sum("recommended" in lbl for lbl, _ in _focus_choices((True, "ready", ""))) == 1


def test_the_recommendation_follows_the_audio_exception():
    """When the sensor hears, Gemini is genuinely the pick (it's the only backend
    that forwards audio), so the marker has to move with the ordering."""
    heard = _glance_choices((True, "ready", ""), hears_audio=True)
    assert heard[0][1][GLANCE_VAR] == "gemini"
    assert "recommended" in heard[0][0]


def test_no_model_option_does_not_say_stub():
    """ "Stub" is a test fixture's name. It meant nothing to the person reading the
    menu, who reasonably asked what it was."""
    labels = [label for label, env in _glance_choices((True, "ready", ""))]
    assert not any(lbl.lower().startswith("stub") for lbl in labels)
    assert any("scripted demo" in lbl for lbl in labels)


def test_backup_is_env_bak_not_env_env_bak(tmp_path: Path):
    """Regression: Path('.env').with_suffix('.env.bak') gives '.env.env.bak':
    a dotfile is all stem, so there's no suffix to replace."""
    path = tmp_path / ".env"
    path.write_text("SACCADE_SENSOR=stub\n")
    _write_env(path, {"SACCADE_SENSOR": "webcam"})
    assert (tmp_path / ".env.bak").read_text() == "SACCADE_SENSOR=stub\n"
    assert not (tmp_path / ".env.env.bak").exists()


def test_write_env_emits_loadable_lines(tmp_path: Path):
    path = tmp_path / ".env"
    _write_env(path, {"SACCADE_SENSOR": "webcam", "SACCADE_WEBCAM_INDEX": "1"})
    body = path.read_text()
    assert "SACCADE_SENSOR=webcam" in body
    assert "SACCADE_WEBCAM_INDEX=1" in body


def test_an_existing_env_is_merged_not_replaced(tmp_path: Path):
    """The wizard sets a handful of vars; a real .env holds keys, comments and
    tuning it never asked about. Overwriting the file to change three lines threw
    all of that away, which is why it had to stop and ask first."""
    path = tmp_path / ".env"
    path.write_text(
        "# my notes\n"
        "GEMINI_API_KEY=secret\n"
        "SACCADE_SENSOR=stub\n"
        "\n"
        "SACCADE_GLANCE_INTERVAL_S=3\n"
    )
    _write_env(path, {"SACCADE_SENSOR": "webcam", "SACCADE_SPEAKER": "piper"})
    assert path.read_text() == (
        "# my notes\n"
        "GEMINI_API_KEY=secret\n"
        "SACCADE_SENSOR=webcam\n"
        "\n"
        "SACCADE_GLANCE_INTERVAL_S=3\n"
        "\n"
        "SACCADE_SPEAKER=piper\n"
    )


def test_merging_asks_nothing(tmp_path: Path, monkeypatch):
    """The old prompt's kind answer ([N]) discarded the entire interview and told
    the user to paste their picks in by hand. Nothing here needs an answer."""
    path = tmp_path / ".env"
    path.write_text("SACCADE_SENSOR=stub\n")
    monkeypatch.setattr(
        "builtins.input", lambda *_: pytest.fail("_write_env asked something")
    )
    _write_env(path, {"SACCADE_SENSOR": "webcam"})
    assert "SACCADE_SENSOR=webcam" in path.read_text()


def test_a_stale_duplicate_does_not_survive_the_merge(tmp_path: Path):
    """_apply_dotenv keeps the first value it sees, so a second line for the same
    key was already inert. Leaving it would put a value in the file that
    contradicts the one saccade actually loads."""
    path = tmp_path / ".env"
    path.write_text("SACCADE_SENSOR=stub\nSACCADE_SENSOR=screen\n")
    _write_env(path, {"SACCADE_SENSOR": "webcam"})
    assert path.read_text() == "SACCADE_SENSOR=webcam\n"


def test_a_sourceable_env_stays_sourceable(tmp_path: Path):
    """`export FOO=1` is a line you can `source`. Rewriting the value shouldn't
    quietly cost the user that."""
    path = tmp_path / ".env"
    path.write_text("export SACCADE_SENSOR=stub\n")
    _write_env(path, {"SACCADE_SENSOR": "webcam"})
    assert path.read_text() == "export SACCADE_SENSOR=webcam\n"


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
    heard = _glance_choices((True, "ready", ""), hears_audio=True)
    assert heard[0][1]["SACCADE_GLANCE_BACKEND"] == "gemini"


def test_video_only_still_leads_with_ollama():
    seen = _glance_choices((True, "ready", ""), hears_audio=False)
    assert seen[0][1]["SACCADE_GLANCE_BACKEND"] == "ollama"


def test_promoting_gemini_keeps_every_backend_on_the_menu():
    heard = _glance_choices((True, "ready", ""), hears_audio=True)
    assert len(heard) == len(_glance_choices((True, "ready", "")))
    assert len(_envs(heard)) == len({e["SACCADE_GLANCE_BACKEND"] for e in _envs(heard)})


def test_hosted_backend_plus_gemini_tts_asks_for_both_keys():
    """Thinking with OpenAI and speaking with Gemini needs two keys; asking only
    for the backend's wrote an .env whose speaker died on the first word."""
    env = {"SACCADE_GLANCE_BACKEND": "openai", "SACCADE_SPEAKER": "gemini_tts"}
    assert _needed_keys(env) == ["OPENAI_API_KEY", "GEMINI_API_KEY"]


def test_speaking_locally_needs_no_key_at_all():
    """The whole point of the local speaker: a machine with no accounts on it can
    still talk. Ollama thinks, Piper speaks, nothing is asked for and nothing
    leaves the box."""
    assert _needed_keys({"SACCADE_GLANCE_BACKEND": "ollama", "SACCADE_SPEAKER": "piper"}) == []


def test_gemini_backend_and_gemini_tts_ask_once():
    env = {"SACCADE_GLANCE_BACKEND": "gemini", "SACCADE_SPEAKER": "gemini_tts"}
    assert _needed_keys(env) == ["GEMINI_API_KEY"]


def test_local_backend_printing_text_needs_no_key():
    assert _needed_keys({"SACCADE_GLANCE_BACKEND": "ollama", "SACCADE_SPEAKER": "print"}) == []


def test_local_backend_speaking_still_needs_the_tts_key():
    env = {"SACCADE_GLANCE_BACKEND": "ollama", "SACCADE_SPEAKER": "gemini_tts"}
    assert _needed_keys(env) == ["GEMINI_API_KEY"]


def test_the_menu_lists_devices_not_combinations():
    """Every entry is one real device plus the way out. A combination is
    expressed by picking several, not by a menu entry describing one."""
    envs = _envs(_sensor_choices((CAMS, SCREENS, MICS)))
    assert not any(e.get("SACCADE_SENSOR") in ("multi", "av") for e in envs)
    assert len(envs) == len(CAMS) + len(SCREENS) + len(MICS) + 1


def test_merging_keeps_each_kind_and_its_index():
    """screen + mic becomes one SACCADE_SENSOR plus both index vars."""
    picks = [c for c in _one_sensor_choices(([], SCREENS, MICS))]
    env, dropped = _merge_sensors(picks)
    assert env["SACCADE_SENSOR"] == "screen,mic"
    assert env["SACCADE_SCREEN_INDEX"] == "1" and env["SACCADE_MIC_INDEX"] == "0"
    assert dropped == []


def test_two_of_one_kind_are_reported_not_silently_dropped():
    """There's one SACCADE_WEBCAM_INDEX, so a second camera would overwrite the
    first. Watching a camera you didn't pick, with no message, is the bad outcome."""
    env, dropped = _merge_sensors(_one_sensor_choices((CAMS, [], [])))
    assert env["SACCADE_SENSOR"] == "webcam"
    assert env["SACCADE_WEBCAM_INDEX"] == "0"  # the first pick wins
    assert len(dropped) == 1 and "Camera 1" in dropped[0]


def test_a_multi_pick_that_includes_a_mic_still_leads_with_the_hearing_backend():
    assert _sensor_kinds({"SACCADE_SENSOR": "screen,mic"}) == {"screen", "mic"}
    assert _glance_choices((True, "ready", ""), hears_audio=True)[0][1][GLANCE_VAR] == "gemini"


def test_wizard_flags_a_backend_whose_sdk_is_missing(monkeypatch):
    """Picking Gemini without the extra wrote a valid .env and then failed on
    every tick with 'No module named google', which reads as saccade being broken
    rather than one install short."""
    monkeypatch.setattr(setuplib, "_importable", lambda module: False)
    env = {setuplib.GLANCE_VAR: "gemini", setuplib.FOCUS_VAR: "ollama"}
    assert setuplib._missing_sdks(env) == ["gemini"]


def test_wizard_is_quiet_when_the_sdk_is_there(monkeypatch):
    monkeypatch.setattr(setuplib, "_importable", lambda module: True)
    assert setuplib._missing_sdks({setuplib.GLANCE_VAR: "gemini"}) == []


def test_local_only_picks_need_no_sdk(monkeypatch):
    """Ollama talks over plain HTTP with the stdlib, so it must never be reported
    as missing an SDK however the probe answers."""
    monkeypatch.setattr(setuplib, "_importable", lambda module: False)
    assert setuplib._missing_sdks({setuplib.GLANCE_VAR: "ollama", setuplib.FOCUS_VAR: "stub"}) == []


def test_existing_key_is_offered_instead_of_demanded(monkeypatch, capsys):
    """Don't send someone to fetch a credential they already exported."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyEXAMPLEKEY0123456789")
    monkeypatch.setattr("builtins.input", lambda prompt="": print(prompt) or "")  # accept default
    assert setuplib._ask_key("GEMINI_API_KEY") == "AIzaSyEXAMPLEKEY0123456789"
    shown = capsys.readouterr().out
    assert "Found GEMINI_API_KEY" in shown
    assert "...6789" in shown  # recognizable
    assert "AIzaSyEXAMPLEKEY" not in shown  # but not the key itself


def test_declining_the_found_key_falls_back_to_typing_one(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyEXAMPLEKEY0123456789")
    answers = iter(["n", "AIzaSyTYPEDBYHAND9876543"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert setuplib._ask_key("GEMINI_API_KEY") == "AIzaSyTYPEDBYHAND9876543"


def test_no_key_in_the_environment_just_asks(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "typed-key")
    assert setuplib._ask_key("GEMINI_API_KEY") == "typed-key"


def test_install_prefers_uv_and_names_the_interpreter(monkeypatch):
    """An ambient `uv pip install` targets whatever venv the shell is in, which
    is not necessarily the one running saccade. Name the interpreter explicitly."""
    monkeypatch.setattr(setuplib.shutil, "which", lambda name: "/usr/bin/uv")
    cmd = setuplib._install_cmd(".[gemini]", editable=True)
    assert cmd == ["uv", "pip", "install", "--python", sys.executable, "-e", ".[gemini]"]


def test_install_falls_back_to_pip_when_uv_is_absent(monkeypatch):
    monkeypatch.setattr(setuplib.shutil, "which", lambda name: None)
    monkeypatch.setattr(setuplib, "_importable", lambda module: True)
    assert setuplib._install_cmd("piper-tts") == [sys.executable, "-m", "pip", "install", "piper-tts"]


def test_install_gives_up_cleanly_with_no_installer(monkeypatch):
    """A uv-made venv ships without pip, so 'neither' is a real state."""
    monkeypatch.setattr(setuplib.shutil, "which", lambda name: None)
    monkeypatch.setattr(setuplib, "_importable", lambda module: False)
    assert setuplib._install_cmd("piper-tts") is None


def test_offer_install_runs_the_command_on_yes(monkeypatch):
    """The whole point: it installs rather than assigning homework."""
    monkeypatch.setattr(setuplib.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr("builtins.input", lambda prompt="": "")  # default is yes
    ran = {}

    def fake_run(cmd, **kw):
        ran["cmd"] = cmd
        return _Ok()

    monkeypatch.setattr(setuplib.subprocess, "run", fake_run)
    assert setuplib._offer_install("The gemini backend", ".[gemini]", editable=True) is True
    assert ran["cmd"][:3] == ["uv", "pip", "install"] and ran["cmd"][-1] == ".[gemini]"


def test_running_the_install_does_not_also_print_it(monkeypatch, capsys):
    """Echoing the command we're about to run reads as homework: you can't tell
    whether it ran or whether you're being handed something to type. Jay had to
    ask which it was."""
    monkeypatch.setattr(setuplib.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    monkeypatch.setattr(setuplib.subprocess, "run", lambda cmd, **kw: _Ok())
    setuplib._offer_install("Speaking out loud", "piper-tts")
    shown = capsys.readouterr().out
    assert "uv pip install" not in shown
    assert "installing piper-tts" in shown


def test_declining_still_hands_over_the_command(monkeypatch, capsys):
    """The command is only useful when we're *not* running it, and then it has to
    be complete: a bare `uv pip install` targets whatever venv the shell is in."""
    monkeypatch.setattr(setuplib.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert setuplib._offer_install("Speaking out loud", "piper-tts") is False
    shown = capsys.readouterr().out
    assert "uv pip install" in shown and sys.executable in shown


def test_offer_install_respects_no(monkeypatch, capsys):
    monkeypatch.setattr(setuplib.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    def must_not_run(*a, **kw):
        raise AssertionError("declining must not install anything")

    monkeypatch.setattr(setuplib.subprocess, "run", must_not_run)
    assert setuplib._offer_install("The gemini backend", ".[gemini]") is False
    assert "uv pip install" in capsys.readouterr().out  # still tells you how


class _Ok:
    returncode = 0


def test_an_unusable_ollama_is_confirmed_not_assumed(monkeypatch, capsys):
    """Review's catch: Ollama keeps the recommendation in every unusable state,
    so 'not installed' rides in on a blind Enter exactly like 'stopped'. It stays
    recommended (that's a claim about where frames go), but it costs a keystroke."""
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "")
    env = {GLANCE_VAR: "ollama", FOCUS_VAR: "gemini"}
    setuplib._confirm_unusable_ollama(env, (False, "not installed", "Install it from:  https://ollama.com"), False)
    assert env[GLANCE_VAR] == "ollama"  # default keeps it
    assert [p.strip() for p in asked] == [">"]
    out = capsys.readouterr().out
    # Every pronoun resolves on screen: which tier picked it, what's wrong, the
    # command that fixes it, and what each answer does. The first cut read
    # "Keep it and fix that after? [Y/n]", which said none of those.
    assert "the watcher" in out
    assert "not installed" in out
    assert "https://ollama.com" in out
    assert "keep Ollama" in out
    assert "works right now" in out


def test_every_unusable_state_reads_as_a_sentence(monkeypatch, capsys):
    """The states are noun phrases as often as adjectives. The first version hung
    them off "is", which rendered "Ollama is no models pulled on this machine" for
    the one state a test never rendered."""
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    for state, fix in (
        ("not running", "Start it with:  ollama serve"),
        ("not installed", "Install it from:  https://ollama.com"),
        ("no models pulled", "Pull one with:  ollama pull gemma3:4b"),
    ):
        # Stands in for a start that didn't take, which is the only way the
        # "not running" copy still reaches a screen.
        monkeypatch.setattr(setuplib, "_start_ollama", lambda s=state, f=fix: (False, s, f))
        setuplib._confirm_unusable_ollama({GLANCE_VAR: "ollama"}, (False, state, fix), False)
        out = capsys.readouterr().out
        assert f"can't answer yet: {state}." in out
        assert fix in out


def test_the_prompt_names_every_tier_that_picked_it(monkeypatch, capsys):
    """Naming one tier when both are affected sends you to fix half of it."""
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    setuplib._confirm_unusable_ollama(
        {GLANCE_VAR: "ollama", FOCUS_VAR: "ollama"},
        (False, "no models pulled", "Pull one with:  ollama pull gemma3:4b"),
        False,
    )
    out = capsys.readouterr().out
    assert "the watcher and the thinker" in out
    assert "ollama pull gemma3:4b" in out


def test_a_stopped_ollama_is_started_not_assigned(monkeypatch, capsys):
    """The one failure the wizard can end by itself. Printing `ollama serve` sent
    the user to a second terminal to run a command we can run right here, which is
    the homework `_offer_install` already exists to stop handing out."""
    started = []
    monkeypatch.setattr(
        setuplib,
        "_start_ollama",
        lambda: started.append(True) or (True, "ready, 1 model(s) pulled", ""),
    )
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("nothing to ask, it's running"))
    env = {GLANCE_VAR: "ollama"}
    setuplib._confirm_unusable_ollama(env, (False, "not running", "Start it with:  ollama serve"), False)
    assert started == [True]
    assert env[GLANCE_VAR] == "ollama"


def test_nobody_else_gets_a_daemon_they_did_not_ask_for(monkeypatch):
    """Ollama being down is only the wizard's business once a tier has picked it.
    Starting a server for someone who chose Gemini is a side effect, not setup."""
    monkeypatch.setattr(setuplib, "_start_ollama", lambda: pytest.fail("started uninvited"))
    setuplib._confirm_unusable_ollama(
        {GLANCE_VAR: "gemini", FOCUS_VAR: "gemini"},
        (False, "not running", "Start it with:  ollama serve"),
        False,
    )


def test_a_start_that_did_not_take_says_so(monkeypatch, capsys):
    """Falling back to "Start it with: ollama serve" right after trying exactly
    that reads as a wizard that wasn't watching its own subprocess."""
    monkeypatch.setattr(setuplib.subprocess, "Popen", lambda *a, **k: None)
    monkeypatch.setattr(setuplib.time, "sleep", lambda _: None)
    monkeypatch.setattr(setuplib, "_ollama_state", lambda: (False, "not running", "Start it with:  ollama serve"))
    usable, state, fix = setuplib._start_ollama()
    assert not usable
    assert "didn't work" in fix
    assert "is up" not in capsys.readouterr().out


def test_declining_an_unusable_ollama_lands_somewhere_that_runs(monkeypatch):
    """Saying no has to go somewhere, or the confirmation is theater."""
    answers = iter(["n", "1"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    env = {GLANCE_VAR: "ollama", FOCUS_VAR: "gemini"}
    setuplib._confirm_unusable_ollama(env, (False, "not installed", "Install it from:  https://ollama.com"), False)
    assert env[GLANCE_VAR] != "ollama"


def test_a_usable_ollama_asks_nothing(monkeypatch):
    def boom(prompt=""):
        raise AssertionError("should not prompt when Ollama works")

    monkeypatch.setattr("builtins.input", boom)
    env = {GLANCE_VAR: "ollama", FOCUS_VAR: "gemini"}
    setuplib._confirm_unusable_ollama(env, (True, "ready, 2 model(s) pulled", ""), False)
    assert env[GLANCE_VAR] == "ollama"
