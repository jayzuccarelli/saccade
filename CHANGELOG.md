# Changelog

## Unreleased

- `setup` starts Ollama when a tier picks it and the daemon is down, instead of
  printing `ollama serve` and leaving. It only does this for a stopped server,
  and only once you've chosen it: installing Ollama and pulling a model are still
  yours. If the start doesn't take, it says so rather than repeating the command.
- `setup` merges into an existing `.env` instead of asking whether to overwrite
  it. The prompt's two answers both lost something: yes dropped every hand-added
  key and comment, no dropped the whole interview and printed the picks back to
  paste in by hand. Only the vars the wizard set are rewritten; the previous file
  is still copied to `.env.bak`.
- `setup` probes the Ollama endpoint the backend will actually use
  (`SACCADE_OLLAMA_HOST`, then `OLLAMA_HOST`, then localhost) instead of always
  probing localhost, so an Ollama on another machine no longer reads as "not
  running" here, and setup never starts a local daemon to fix a remote one.

- The `home_assistant` speaker now synthesizes with Piper by default instead of
  Gemini TTS, so playing on a media_player no longer forces an API key. Set
  `SACCADE_HA_TTS=gemini_tts` to keep the hosted voices. **Behavior change:** an
  existing HA setup that relied on Gemini needs that variable, or Piper installed.
- CI resolves `uv.lock` against every extra on each matrix Python. `make check`
  goes through `uv run`, which re-resolves and ignores the lock, so nothing was
  checking that the committed lock could actually be installed.

## 0.2.0

Audio, and an attention loop that paces itself.

**Hearing**
- `mic` sensor: records a glance-interval clip per tick and feeds it to the model.
- `av` sensor: webcam + mic fused into one Frame, so a single glance both sees
  and hears the same instant.
- `Frame` now carries `audio` alongside `image`; a frame can be sound, a picture,
  or both.
- `saccade snapshot clip.wav`: test hearing on a file, no hardware.
- Audio input needs an audio-capable backend: Gemini today (Anthropic/Ollama are
  vision-only; OpenAI needs a dedicated audio model).

**Voice out**
- `SACCADE_AUDIO_OUT_INDEX`: play speech to a specific output device (the twin
  of picking a mic), overriding the OS-default `play_cmd`.

**Attention**
- Adaptive cadence. Glance emits `next_glance_s` and paces itself: watch every
  tick on action, rest on a calm scene. Only ever slower than `glance_fps`.
- Concurrent Focus: a salient frame spawns Focus in the background so Glance
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
