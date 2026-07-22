# Changelog

## Unreleased

- `setup` offers to install the `stt` extra instead of printing the command.
  Picking local transcription without it left the run to die on its first audio
  frame with a 40-line traceback ending in `No module named 'faster_whisper'`.
  That module now maps to its extra too, so if it does happen you get the one
  line that fixes it.

- A glance sees one frame from *each* input, not just the newest frame. With a
  camera and a mic both running, the camera wins nearly every tick and the room
  is effectively never heard, so `SACCADE_SENSOR=webcam,screen,mic` looked like a
  broken mic. Frames now carry which input they came from. A single sensor is
  unchanged.
- Ctrl-C exits immediately instead of waiting on an in-flight model call. A
  blocking request in a worker thread can't be cancelled and the interpreter
  joins it at exit, so a 120s client timeout meant two minutes of a dead-looking
  terminal swallowing further Ctrl-Cs.
- The audio question describes what saccade does rather than naming a vendor:
  "only Gemini accepts audio" read as a plug and wasn't the claim either.

- Playback falls back to the OS player whenever the chosen device refuses the
  clip, not just when it can't do the channel count. A device that won't open at
  Piper's 22050 Hz answers `PaMacCore err='-50'`, and that was still eating every
  utterance after the channel fix.

- Glance's `salience` has anchors instead of a vibe. A real run scored a man
  sitting at a desk 0.7-0.8 on every tick, which means the number carries no
  information and Focus has nothing to go on. Someone working is 0.1, 0.9+ is
  something wrong, and not being able to tell what's happening is 0.1 with a
  summary that says so.
- A quiet glance overwrites the last quiet one instead of scrolling. An hour of
  an empty room was 3,600 near-identical lines, which buries the few that meant
  something and, when a screen is one of the sensors, feeds back in as input.
  Escalations, Focus verdicts and errors still scroll, and piping to a file still
  keeps every tick.

- `setup` offers the Ollama models you already have before offering a download,
  and checks them by name rather than by count. Having some other model pulled
  read as "ready", and then every tick died on `Ollama has no model 'gemma3:4b'`.
  Models are filtered by what they can do (`/api/show` capabilities), so a
  text-only model is never offered to a tier watching a camera. Downloading is
  offered rather than automatic: it's gigabytes, unlike everything else the
  wizard runs for you.
- Playback matches the output device instead of assuming it takes the clip as-is.
  Piper writes mono and plenty of CoreAudio outputs will only open a stereo
  stream, so every utterance died with `PortAudioError: Invalid number of
  channels [-9998]`: the agent watched all day and never made a sound. An index
  that can't output at all now warns and falls back to the OS default rather than
  costing you every spoken line.
- Glance is told that a screen may be showing saccade's own log, and that reading
  it back is not an event. Watching a screen means watching the terminal saccade
  prints to: it read its own `man` summaries twenty times over and reported
  `man`, and Focus escalated on the user typing about it.
- Ctrl-C exits instead of printing a threading traceback, and says the pause is a
  model call still in flight rather than leaving it looking like a hang.
- Glance's prompt asks for a sentence, not a label. "one short line: what you
  notice now" got `man`, once a second, which is indistinguishable from nothing
  happening and is all Focus had to decide on.
- The loop starts a stopped local Ollama instead of failing every tick with
  `start it: ollama serve`. Setup already did this; the tick that actually needs
  the daemon didn't. Local host only, once per process, and it says so in the log.
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
