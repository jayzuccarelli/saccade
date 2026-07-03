# Changelog

## 0.2.0

Audio, and an attention loop that paces itself.

**Hearing**
- `mic` sensor — records a glance-interval clip per tick and feeds it to the model.
- `av` sensor — webcam + mic fused into one Frame, so a single glance both sees
  and hears the same instant.
- `Frame` now carries `audio` alongside `image`; a frame can be sound, a picture,
  or both.
- `saccade snapshot clip.wav` — test hearing on a file, no hardware.
- Audio input needs an audio-capable backend — Gemini today (Anthropic/Ollama are
  vision-only; OpenAI needs a dedicated audio model).

**Voice out**
- `SACCADE_AUDIO_OUT_INDEX` — play speech to a specific output device (the twin
  of picking a mic), overriding the OS-default `play_cmd`.

**Attention**
- Adaptive cadence — Glance emits `next_glance_s` and paces itself: watch every
  tick on action, rest on a calm scene. Only ever slower than `glance_fps`.
- Concurrent Focus — a salient frame spawns Focus in the background so Glance
  never goes blind while the big model reasons (single-slot).

Both on by default (`SACCADE_ADAPTIVE_CADENCE=0` / `SACCADE_CONCURRENT_FOCUS=0`
for the old behavior).

## 0.1.1

- Fixed 29 first-run blockers found by an adversarial fresh-clone audit: the
  Pillow-missing crash, the broken console script, `saccade devices` crashing
  without PortAudio, `.env` trailing-comment parsing, and more.
- CI now runs the real `make check` gate (ruff + full pytest with pillow) on
  ubuntu 3.10/3.12 and macOS.

## 0.1.0

- First release: two-tier ambient loop (Glance → Focus → Speaker), swappable
  sensors/backends/speakers, structured output, episodic memory, evals, tests.
