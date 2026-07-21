# CLAUDE.md: notes for contributors (human or agent)

Orientation for working in this repo. User-facing setup lives in `README.md`.

## What it is

An ambient-agent harness with a two-tier attention loop:

- **Sensor** streams `Frame`s (camera / screen / mic / RTSP / replay).
- **Glance**: a cheap, fast model runs ~1 Hz over recent frames and decides
  whether anything is worth a closer look (`escalate`, `salience`). Biased
  toward *quiet*: precision over recall. It also emits `next_glance_s` so the
  loop can slow down when nothing's happening (adaptive cadence).
- **Focus**: an expensive model reasons only on escalation and decides whether
  to actually speak. It runs concurrently, so Glance keeps observing while Focus
  thinks.
- **Speaker** emits the action (print / Piper / Gemini TTS / Home Assistant).

Everything is swappable through `config.py` (env vars) and the factories in
`__main__.py`. Models sit behind the `Backend` protocol (Gemini / OpenAI /
Anthropic / Ollama / stub); sensors and speakers behind their own protocols.

The intended split is a local model on Glance (it sees every frame, so it decides
whether a camera pointed at your kitchen streams to a vendor all day) and a
hosted model on Focus (it runs a few times a day, only on moments that already
cleared the bar). `SACCADE_GLANCE_BACKEND` and `SACCADE_FOCUS_BACKEND` are
independent for exactly this reason.

## Rules

Four things the design depends on. Everything else is negotiable.

- **No hand-coded decision rules.** The only judgments are the two model calls
  (`glance.py`, `focus.py`). Salience, urgency, tone: context fed to the model,
  never `if x and y: speak`.
- **Vendor SDKs only in `backends/`** and the TTS speakers. Importing
  `google.genai` or `openai` anywhere else is the wrong layer.
- **Don't break the Protocols** in `sensors/base.py`, `backends/base.py`,
  `speakers/base.py`. Every concrete class is interchangeable; that's the point.
- **Structured output goes through the schemas** in `schema.py`
  (`PERCEPT_SCHEMA`, `DECISION_SCHEMA`). Don't parse free text, and don't add a
  fourth schema unless you're adding a fourth role.

**Keep Focus quiet.** Speaking on every low-stakes event is the failure mode.
When in doubt, don't speak.

## Where things go

One file per new piece; nothing else changes.

- New camera/mic/screen → `sensors/yours.py` implementing `Sensor.stream()`.
- New model provider → `backends/yours.py` implementing `Backend.complete()`,
  translating `schema` to that provider's structured-output mechanism.
- New output (a speaker, a TV, a phone) → `speakers/yours.py` implementing
  `Speaker.say()`.
- Register it in `__main__.py`'s dispatch.

**Config and secrets.** `config.py` auto-loads `.env` at import time (stdlib
only, no `python-dotenv`); the real environment wins over `.env`. Add new vars as
dataclass fields with `os.environ.get(...)` defaults rather than reading env
scattered across modules.

## The gate

`make check` is the self-grading verify loop and must be green before anything
ships:

    make check      # ruff + mypy --strict + pytest

CI runs the same thing across 3.10/3.12 on Linux and macOS, plus a
`uv sync --locked --all-extras` that proves the committed lockfile still
installs (`make check` goes through `uv run`, which re-resolves and would not
notice a broken lock).

Green tests are necessary, not sufficient. Before calling something done:

1. `uv run python -m saccade` with no env still works end-to-end.
2. If you touched a real path (camera, model, speaker), run it. Tests verify
   code; running verifies the feature.

## Conventions

- **uv** for everything: deps, lockfile, running (`uv run ...`). Commit
  `uv.lock`. Lint + format with **ruff**. Never document a bare `python` or
  `pip` command: macOS has neither on PATH, and the wrong `pip` installs where
  the running saccade can't import from.
- **Typed strict.** `mypy --strict` is clean; keep it that way. Untyped
  third-party libs are allowed via `ignore_missing_imports`.
- **Optional deps go behind extras** (`camera`, `screen`, `audio`, `stt`,
  `gemini`, …) and are imported lazily inside the function that needs them, so
  the base harness has no hard dependency on cv2 / PortAudio / a provider SDK.
  A GPL dependency is run as a subprocess, never imported (see `speakers/piper.py`).
- **Evals measure the hard part.** `uv run python -m saccade.evals` scores
  Glance's salience judgment (precision/recall) against labeled scenes. Tune a
  prompt, re-run, watch the numbers; don't tune by vibe.
- **Docstrings earn their keep.** Say why, not what the code already says.

## Avoid

- A fallback that tries the next backend on error. A silent provider switch is a
  debugging trap.
- A "smart" cache that compares frames. An image diff is hand-coded salience,
  which is the first rule above.
- Refactoring what wasn't asked for.

## Try it fast

    uv run python -m saccade                   # scripted stub: no key, no camera
    uv run python -m saccade devices           # list cameras / screens / mics / outputs
    uv run python -m saccade snapshot pic.jpg  # one frame through Glance (then Focus if salient)
