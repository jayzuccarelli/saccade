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
import time
from pathlib import Path
from urllib import request

from saccade.devices import _audio, _cameras, _screens

_OLLAMA_HOST = "http://localhost:11434"

# The one Ollama problem the wizard can fix by itself, named once so the check
# and the fix can't drift apart. Only ever set for a local host, which is what
# keeps `_start_ollama` from spawning a daemon on the wrong machine.
_NOT_RUNNING = "not running"

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]")


def _ollama_endpoint() -> tuple[str, bool]:
    """The host the backend will actually use, resolved the same way it resolves
    it, and whether that's this machine.

    The wizard probed localhost no matter what, so pointing `SACCADE_OLLAMA_HOST`
    at another box reported "not running" about a server that was answering fine.
    That was cosmetic until setup could start one; now it decides whether we spawn
    a daemon nobody asked for on the machine running the wizard."""
    host = os.environ.get("SACCADE_OLLAMA_HOST") or os.environ.get("OLLAMA_HOST") or _OLLAMA_HOST
    if "://" not in host:  # OLLAMA_HOST is conventionally bare: "127.0.0.1:11434"
        host = f"http://{host}"
    name = host.split("://", 1)[1].split("/")[0].rsplit(":", 1)[0]
    return host.rstrip("/"), name in _LOCAL_HOSTS

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
    """Every device, one entry each, plus the way out. Nothing here describes a
    *combination*: combining is what picking several means, not its own option."""
    return [*_one_sensor_choices(devs), ("Nothing: scripted demo, no hardware", {"SACCADE_SENSOR": "stub"})]


def _pick_sensors(picked: list[Choice]) -> tuple[dict[str, str], list[str], str]:
    """Fold any combination of picked devices into one sensor config, plus the
    entries that had to be dropped and a line explaining what we did.

    Which sensor *class* runs is an implementation detail, and it used to be a
    question: the menu offered "a camera and a mic together" next to "several at
    once", which are two descriptions of the same intent, so the answer depended
    on guessing our internal names. One camera and one mic fuse into AVSensor,
    where a frame carries what it saw and heard at the same instant. Anything
    else interleaves independent streams at their own rates. Both are just "I
    picked these inputs"."""
    env, dropped = _merge_sensors([c for c in picked if c[1].get("SACCADE_SENSOR") != "stub"])
    kinds = _sensor_kinds(env)
    if not kinds:
        return {"SACCADE_SENSOR": "stub"}, dropped, ""
    if kinds == {"webcam", "mic"}:
        # Fused, not interleaved: the sound and the image describe one moment,
        # which is the difference between "he flinched" and "he flinched at that".
        env["SACCADE_SENSOR"] = "av"
        return env, dropped, "camera and mic fused: each frame is one moment, seen and heard"
    if len(kinds) > 1:
        return env, dropped, f"{len(kinds)} inputs, each at its own rate"
    return env, dropped, ""


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


def _ollama_state() -> tuple[bool, str, str]:
    """Whether Ollama can actually answer, what's wrong in a few words, and the one
    line that fixes it. Ask the daemon, don't ask `which ollama`: a Mac with the
    binary installed and the server down is the common case, and it fails as
    connection-refused on every tick rather than at setup time, which is where
    you'd want to hear it.

    Two fields, not one, because the two readers want different lengths. A menu
    label wants "not running"; the prompt that asks you to accept that state wants
    the command that ends it. Concatenated, the menu carried a shell command no
    one could run from inside a menu."""
    host, local = _ollama_endpoint()
    try:
        with request.urlopen(f"{host}/api/tags", timeout=0.5) as resp:
            models = json.loads(resp.read()).get("models", [])
    except (OSError, ValueError):  # URLError subclasses OSError
        if not local:
            # Nothing here to install or start: that machine's problem, and
            # saying "not installed" about it would send them to fix this one.
            return False, f"not answering at {host}", "Check that it's running and reachable there."
        if shutil.which("ollama"):
            return False, _NOT_RUNNING, "Start it with:  ollama serve"
        return False, "not installed", "Install it from:  https://ollama.com"
    if not models:
        return False, "no models pulled", "Pull one with:  ollama pull gemma3:4b"
    return True, f"ready, {len(models)} model(s) pulled", ""


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


def _recommend(choices: list[Choice]) -> list[Choice]:
    """Mark [1] as the one we'd pick.

    Ordering on its own says nothing: a menu sorted by our preference looks
    exactly like a menu sorted arbitrarily, so on the tier that decides whether
    a camera streams all day, the pick goes to whatever happens to read first."""
    (label, env), *rest = choices
    return [(f"{label}  [recommended]", env), *rest]


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
        # Names what saccade does, not what a vendor does. "only Gemini accepts
        # audio" read as a plug and wasn't even the claim: Anthropic takes no
        # audio at all, OpenAI takes it only through a dedicated audio model, and
        # neither of those backends is wired for it here. What the user needs to
        # know is that picking this narrows their model choice to one.
        (
            "Send the recording to the model: raw audio, and only the Gemini "
            "backend forwards it (the others drop it)",
            {STT_VAR: ""},
        ),
    ]


def _glance_choices(ollama: tuple[bool, str, str], hears_audio: bool = False) -> list[Choice]:
    """Glance is the tier that runs about once a second and looks at *every* frame,
    so this pick decides whether a camera pointed at your kitchen streams to a
    vendor all day. Local leads whenever it can actually run.

    Gemini takes the lead instead when the sensor captures audio, because it's the
    only backend that forwards it; anything else accepts the mic pick and then
    silently drops the audio half.

    A stopped Ollama does *not* cost it the lead. It used to, and the effect was
    that the machine most in need of the local pick (nothing running yet) was the
    one steered hardest toward uploading every frame. Being one `ollama serve`
    away is a fixable state, and the label says so; the wizard warns again at the
    end if the pick is still down."""
    _usable, state, _fix = ollama
    labels = {
        "ollama": f"Ollama: runs on this machine, frames never leave it ({state})",
        "gemini": "Gemini: hosted; every frame it looks at is uploaded (only one that hears audio)",
        "openai": "OpenAI: hosted; every frame it looks at is uploaded",
        "anthropic": "Anthropic: hosted; every frame it looks at is uploaded",
        "stub": _NO_MODEL,
    }
    order = _ordered(
        ["ollama", "gemini", "openai", "anthropic", "stub"],
        first="gemini" if hears_audio else "",
    )
    return _recommend([(labels[k], {GLANCE_VAR: k}) for k in order])


def _focus_choices(ollama: tuple[bool, str, str]) -> list[Choice]:
    """Focus only runs when Glance escalates, which is rare by design. That's why
    the expensive, capable model belongs here and not on the 1 Hz tier: you pay
    for it a few times a day, and the only thing it ever sees is a moment that
    already cleared the bar."""
    usable, state, _fix = ollama
    sees = "hosted; sees only the moments Glance escalates"
    labels = {
        "gemini": f"Gemini: {sees}",
        "anthropic": f"Anthropic: {sees}",
        "openai": f"OpenAI: {sees}",
        "ollama": f"Ollama: runs on this machine, nothing leaves it ({state})",
        "stub": _NO_MODEL,
    }
    order = _ordered(
        ["gemini", "anthropic", "openai", "ollama", "stub"], last="" if usable else "ollama"
    )
    return _recommend([(labels[k], {FOCUS_VAR: k}) for k in order])


def _start_ollama() -> tuple[bool, str, str]:
    """Start the daemon instead of telling someone to, and return what it does next.

    A stopped Ollama is the only one of the three failures the wizard can end by
    itself: the binary is already there, and nothing is answering on the port, so
    there's no server to collide with (a Mac running Ollama.app would already be
    replying). Handing back `ollama serve` made the user open a second terminal to
    run a command we could run here, which is the same homework `_offer_install`
    exists to stop assigning.

    Detached, because it has to outlive the wizard: the whole point is that saccade
    can reach it afterwards. Polled rather than assumed, because "the process
    spawned" and "the daemon answers" are different claims, and only the second one
    is the one worth printing."""
    # Whatever comes back, not the advice `_ollama_state` gives a machine that
    # hasn't tried yet: "Start it with: ollama serve", printed immediately after
    # doing exactly that and failing, reads as a wizard that isn't watching.
    failed = (False, _NOT_RUNNING, "Starting it here didn't work; try  ollama serve  yourself.")
    print("\n  Ollama is installed but not running. Starting it...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return failed
    stop = "taskkill /IM ollama.exe" if sys.platform == "win32" else "pkill ollama"
    for _ in range(20):
        time.sleep(0.25)
        usable, state, fix = _ollama_state()
        if state == _NOT_RUNNING:
            continue
        # Announced whenever it came up, not only when it came up ready: a server
        # with no models pulled is still a server we started, and the "Starting
        # it..." above needs an ending either way. What's left to do is the next
        # prompt's job.
        print(f"  Ollama is up. It stays up after setup; `{stop}` stops it.")
        return usable, state, fix
    return failed


def _confirm_unusable_ollama(
    env: dict[str, str], ollama: tuple[bool, str, str], hears_audio: bool
) -> None:
    """If a tier picked an Ollama that can't answer, make that an explicit choice.

    Ollama keeps the recommendation in every unusable state, because the
    recommendation is a claim about where frames go, not about what happens to be
    running right now. Review pushed back that this lets "not installed" ride in
    on a blind Enter exactly like "stopped", which is fair: those need more than
    `ollama serve`. So it costs one keystroke instead of the lead, and declining
    lands on a backend that actually runs rather than leaving the loop to find
    out.

    Both answers spell out what they do. This used to read "Keep it and fix that
    after? [Y/n]", where "it" and "that" had no antecedent on screen, the tier that
    picked Ollama went unnamed, and pressing n announced nothing at all. A prompt
    that can't be answered without guessing is worse than no prompt."""
    usable, state, fix = ollama
    tiers = [
        name
        for var, name in ((GLANCE_VAR, "the watcher"), (FOCUS_VAR, "the thinker"))
        if env.get(var) == "ollama"
    ]
    if usable or not tiers:
        return
    if state == _NOT_RUNNING:
        # Only now, once a tier has actually picked it: starting a daemon nobody
        # asked for is a side effect, starting the one they just chose is setup.
        ollama = _start_ollama()
        usable, state, fix = ollama
        if usable:
            return
    # "can't answer yet: {state}", not "Ollama is {state}". The states are noun
    # phrases as often as adjectives, and hanging them off "is" produced
    # "Ollama is no models pulled on this machine".
    print(
        f"\n  You picked Ollama for {' and '.join(tiers)}, "
        f"but it can't answer yet: {state}.\n"
        f"  {fix}\n\n"
        "  [Y]  keep Ollama, and do that before you run saccade\n"
        "  [n]  pick a model that works right now\n"
    )
    if input("  > ").strip().lower() in ("", "y", "yes"):
        return
    for var, choices in (
        (GLANCE_VAR, _glance_choices(ollama, hears_audio)),
        (FOCUS_VAR, _focus_choices(ollama)),
    ):
        if env.get(var) == "ollama":
            env.update(_ask("What instead?", [c for c in choices if c[1][var] != "ollama"]))


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


def _install_cmd(spec: str, editable: bool = False) -> list[str] | None:
    """The command that installs `spec` into *this* interpreter, or None if this
    environment has no installer we can drive.

    uv first, and `--python sys.executable` explicitly: a uv-made venv ships
    without pip, and an ambient `uv pip install` targets whatever venv the shell
    is in, which is not necessarily the one running saccade."""
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install", "--python", sys.executable]
    elif _importable("pip"):
        cmd = [sys.executable, "-m", "pip", "install"]
    else:
        return None
    if editable:
        cmd.append("-e")
    cmd.append(spec)
    return cmd


MODEL_VARS = {"glance": "SACCADE_GLANCE_MODEL", "focus": "SACCADE_FOCUS_MODEL"}


def _pulled_models() -> dict[str, set[str]]:
    """Every model already on this machine, mapped to what it can do.

    Capabilities come from /api/show rather than a list of model names we'd have
    to keep current: "vision" is the one that matters, because a text-only model
    picked for a camera fails on every frame instead of at setup."""
    host, _ = _ollama_endpoint()
    try:
        with request.urlopen(f"{host}/api/tags", timeout=0.5) as resp:
            names = [m["name"] for m in json.loads(resp.read()).get("models", [])]
    except (OSError, ValueError, KeyError):
        return {}
    out: dict[str, set[str]] = {}
    for name in names:
        req = request.Request(
            f"{host}/api/show",
            data=json.dumps({"model": name}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=2.0) as resp:
                out[name] = set(json.loads(resp.read()).get("capabilities", []))
        except (OSError, ValueError):
            # An older Ollama doesn't report capabilities. Keep the model listed
            # rather than hiding what someone already has; it just can't be
            # offered for a tier that needs to see.
            out[name] = set()
    return out


def _default_model(role: str) -> str:
    from saccade.__main__ import DEFAULT_MODELS  # module scope would cycle

    return DEFAULT_MODELS[("ollama", role)]


def _model_choices(role: str, pulled: dict[str, set[str]], needs_vision: bool) -> list[Choice]:
    """What this tier could run: the models already here first, the download last.

    Downloading a fourth model onto a machine that already has three that would
    do the job is the wrong default, and it's the wizard that knows both halves.
    A tier watching a camera only gets models that report vision; the rest would
    take the job and then fail on every frame."""
    var = MODEL_VARS[role]
    usable = [n for n, caps in sorted(pulled.items()) if "vision" in caps or not needs_vision]
    out: list[Choice] = [(f"{n}: already pulled, no download", {var: n}) for n in usable]
    default = _default_model(role)
    if default in pulled:
        return [(f"{default}: already pulled, no download", {var: default})]
    out.append((f"{default}: the default, a few GB to download", {var: default}))
    return out


def _resolve_models(env: dict[str, str], needs_vision: bool) -> list[str]:
    """Settle which model each Ollama tier runs, and return the ones to download.

    Always writes the model var for an Ollama tier, even when it picks the
    default: leaving it unset lets an override from a previous run quietly win
    over the answer just given."""
    to_pull = []
    pulled = _pulled_models()
    for var, role, label in (
        (GLANCE_VAR, "glance", "the watcher"),
        (FOCUS_VAR, "focus", "the thinker"),
    ):
        if env.get(var) != "ollama":
            continue
        choices = _model_choices(role, pulled, needs_vision)
        picked = choices[0][1] if len(choices) == 1 else _ask(f"Which model for {label}?", choices)
        env.update(picked)
        model = picked[MODEL_VARS[role]]
        if model not in pulled:
            to_pull.append(model)
    return to_pull


def _offer_missing_models(env: dict[str, str], needs_vision: bool = True) -> None:
    """Settle the models, then download only what isn't here.

    Checked by name, not by count: `ready, 3 model(s) pulled` was standing in for
    "has the model this run needs", and the two came apart the moment someone had
    three unrelated models, at which point setup said ready and every tick died on
    `no model 'gemma3:4b'`.

    The download is offered, not automatic, and never at run time. Everything else
    the wizard runs for you takes seconds; this is gigabytes over someone's
    network, and a download that size starting on its own (or worse, inside the
    loop, where it would look like a hang) is the one case where asking earns its
    keystroke."""
    missing = _resolve_models(env, needs_vision)
    if not missing:
        return
    if not shutil.which("ollama"):
        # Keeping Ollama after being told it isn't installed is allowed, and then
        # `ollama pull` is a FileNotFoundError that takes the whole wizard down
        # before .env is written. Offering to run a command that can't run is
        # worse than printing it.
        print("\n  Once Ollama is installed:\n")
        for model in missing:
            print(f"    ollama pull {model}")
        print()
        return
    names = " and ".join(missing)
    size = "a few GB each" if len(missing) > 1 else "a few GB"
    print(f"\n  {names} still {'need' if len(missing) > 1 else 'needs'} downloading: {size}.")
    if input("  Download now? [Y/n] ").strip().lower() not in ("", "y", "yes"):
        print(f"\n  When you want {'them' if len(missing) > 1 else 'it'}:\n")
        for model in missing:
            print(f"    ollama pull {model}")
        print("\n  Until then every tick fails with 'Ollama has no model'.\n")
        return
    for model in missing:
        print(f"\n  pulling {model}...\n")
        subprocess.run(["ollama", "pull", model])


def _offer_install(what: str, spec: str, editable: bool = False) -> bool:
    """Offer to run the install rather than assigning it as homework.

    The wizard already knows the exact command; printing it and quitting makes
    the user re-derive a working shell invocation, which is where a Mac loses
    them (no `python` on PATH, wrong `pip`, right package in the wrong venv)."""
    cmd = _install_cmd(spec, editable)
    if cmd is None:
        print(f"\n  {what} needs `{spec}`, but neither uv nor pip is available here.\n")
        return False
    if input(f"\n{what} needs `{spec}`. Install it now? [Y/n] ").strip().lower() not in (
        "",
        "y",
        "yes",
    ):
        print(f"\n  When you want it:\n\n    {' '.join(cmd)}\n")
        return False
    # Says what's happening, not what to type. Echoing the command we're about to
    # run reads as homework: you can't tell whether it ran or whether you're being
    # handed it, and the installer's own output is already the progress report.
    print(f"\n  installing {spec}...\n")
    return subprocess.run(cmd).returncode == 0


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


# Every var the wizard writes as an answer to one of its questions. Merging keeps
# what it doesn't own, so it has to know what it does: a var in here that this run
# didn't set is a stale answer to a question just asked again, and leaving it in
# lets the old answer beat the new one. Deliberately excludes the API keys, which
# are stored secrets rather than answers, and dropping one because this run picked
# a different backend would make you dig it out again.
_WIZARD_VARS = frozenset(
    {
        "SACCADE_SENSOR",
        "SACCADE_WEBCAM_INDEX",
        "SACCADE_SCREEN_INDEX",
        "SACCADE_MIC_INDEX",
        GLANCE_VAR,
        FOCUS_VAR,
        STT_VAR,
        "SACCADE_SPEAKER",
        "SACCADE_PLAY_CMD",
        _OUT_VAR,
        # The model vars only get written for an Ollama tier, so they have to be
        # cleared for one that isn't: moving Glance from Ollama to Gemini used to
        # leave SACCADE_GLANCE_MODEL=gemma3:4b behind, and the runtime would then
        # ask Gemini, in earnest, for gemma3:4b.
        *MODEL_VARS.values(),
    }
)


def _write_env(path: Path, env: dict[str, str]) -> None:
    """Write the picks into `path`, rewriting only the vars the wizard just set.

    Merge, don't ask. This used to prompt "Overwrite? [y/N]", and both answers
    threw away something the user wanted: y dropped every hand-added line in the
    file, N dropped the whole interview and printed the picks back to paste in by
    hand. The safe-looking default was the destructive one, discarding the dozen
    questions they had just answered. There's nothing to ask about: the wizard
    owns the vars it set and nothing else, so those lines change in place and
    comments, blanks and anything else survive untouched."""
    lines = path.read_text().splitlines() if path.exists() else []
    if lines:
        # .env has no suffix to replace (it's all stem), so with_suffix would
        # make ".env.env.bak"; append instead.
        backup = path.with_name(path.name + ".bak")
        shutil.copyfile(path, backup)
        print(f"\n  (kept the rest of your {path}; the version before this is at {backup})")
    out = [] if lines else ["# Written by `python -m saccade setup`. Edit freely."]
    unwritten = dict(env)
    for line in lines:
        body = line.strip()
        prefix = "export " if body.startswith("export ") else ""
        key = body[len(prefix) :].partition("=")[0].strip()
        if key in _WIZARD_VARS and key not in env:
            # Review's catch: an old SACCADE_AUDIO_OUT_INDEX outlived a run that
            # chose SACCADE_PLAY_CMD instead, and the index wins at playback, so
            # speech died on the stale device the wizard had just replaced.
            continue
        if body.startswith("#") or "=" not in body or key not in env:
            out.append(line)
        elif key in unwritten:
            out.append(f"{prefix}{key}={unwritten.pop(key)}")
        # else: a second line for a key already rewritten above. _apply_dotenv
        # takes the first one it sees, so this was dead weight before we got
        # here; keeping it would just disagree with the live value in writing.
    if unwritten:
        if out and out[-1].strip():
            out.append("")
        out.extend(f"{k}={v}" for k, v in unwritten.items())
    path.write_text("\n".join(out) + "\n")


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
    picked = _ask_many(
        "What should saccade watch and hear?  (any combination: a camera, a screen, a mic, or several)",
        _sensor_choices(devs),
    )
    merged, dropped, note = _pick_sensors(picked)
    for label in dropped:
        print(f"  note: skipped {label} (only one of each kind for now)")
    if note:
        print(f"  {note}")
    env.update(merged)
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

    _confirm_unusable_ollama(env, ollama, hears_audio)
    # A mic-only run never sends an image, so a text-only model someone already
    # has is a perfectly good watcher. Anything with a camera or a screen in it
    # needs one that can see.
    _offer_missing_models(env, needs_vision=bool(_sensor_kinds(env) - {"mic"}))
    if env.get(STT_VAR) == "whisper" and not stt[0]:
        # Offered, like every other extra. Printing the command and moving on left
        # the run to die on its first audio frame with a traceback, which is the
        # homework `_offer_install` exists to stop assigning.
        if not _offer_install("Transcribing on this machine", ".[stt]", editable=True):
            print("  Until then the first audio frame ends the run.\n")
    if env.get("SACCADE_SPEAKER") == "piper" and not piper[0]:
        if _offer_install("Speaking out loud", "piper-tts"):
            voice = env.get("SACCADE_PIPER_VOICE", "en_US-lessac-medium")
            print(f"\n  downloading the {voice} voice...\n")
            subprocess.run([sys.executable, "-m", "piper.download_voices", voice])
        else:
            print(f"\n  The two commands, when you want them:\n\n{_piper_setup_commands()}")

    for extra in _missing_sdks(env):
        if not _offer_install(f"The {extra} backend", f".[{extra}]", editable=True):
            print("  Until then every tick fails with 'No module named ...'.\n")

    for var in _needed_keys(env):
        key = _ask_key(var)
        if key:
            env[var] = key

    _write_env(Path(".env"), env)

    # sys.executable, not a bare `python`: the shell that just ran setup may have
    # no `python` on it at all (macOS), and the one it does have is often not the
    # venv holding saccade's deps.
    print(f"\nWrote .env. Start it with:\n\n    {sys.executable} -m saccade\n")
    if {"webcam", "av"} & _sensor_kinds(env) and sys.platform == "darwin":
        print("macOS: approve the Camera prompt on first run, then rerun.\n")
