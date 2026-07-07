# CLAUDE.md — notes for contributors (human or agent)

Orientation for working in this repo. User-facing setup lives in `README.md`.

## What it is

An ambient-agent harness with a two-tier attention loop:

- **Sensor** streams `Frame`s (camera / screen / mic / RTSP / replay).
- **Glance** — a cheap, fast model runs ~1 Hz over recent frames and decides
  whether anything is worth a closer look (`escalate`, `salience`). Biased
  toward *quiet*: precision over recall. It also emits `next_glance_s` so the
  loop can slow down when nothing's happening (adaptive cadence).
- **Focus** — an expensive model reasons only on escalation and decides whether
  to actually speak. It runs concurrently, so Glance keeps observing while Focus
  thinks.
- **Speaker** emits the action (print / Gemini TTS / Home Assistant).

Everything is swappable through `config.py` (env vars) and the factories in
`__main__.py`. Models sit behind the `Backend` protocol (Gemini / OpenAI /
Anthropic / Ollama / stub); sensors and speakers behind their own protocols.

## The gate

`make check` is the self-grading verify loop and must be green before anything
ships:

    make check      # ruff + mypy --strict + pytest

CI runs the same thing across 3.10/3.12 on Linux and macOS. Keep it green.

## Conventions

- **uv** for everything — deps, lockfile, running (`uv run ...`). Commit
  `uv.lock`. Lint + format with **ruff**.
- **Typed strict.** `mypy --strict` is clean; keep it that way. Untyped
  third-party libs are allowed via `ignore_missing_imports`.
- **Optional deps go behind extras** (`camera`, `screen`, `audio`, `gemini`,
  …) and are imported lazily inside the function that needs them, so the base
  harness has no hard dependency on cv2 / PortAudio / a provider SDK.
- **Evals measure the hard part.** `python -m saccade.evals` scores Glance's
  salience judgment (precision/recall) against labeled scenes. Tune a prompt,
  re-run, watch the numbers — don't tune by vibe.
- **Keep Focus quiet.** Speaking on every low-stakes event is the failure mode.
  When in doubt, don't speak.

## Try it fast

    python -m saccade                 # scripted stub — no key, no camera
    python -m saccade devices         # list cameras / screens / mics / outputs
    python -m saccade snapshot pic.jpg  # one frame through Glance (then Focus if salient)
