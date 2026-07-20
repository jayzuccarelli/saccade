# saccade

An open harness for **proactive ambient agents** — things that watch and listen,
and speak up only when it's actually useful.

It copies how attention works in the brain. Cheap **peripheral** awareness scans
constantly; when something is **salient**, it snaps **focus** onto it to reason
and act. Named after the *saccade*: the eye's flick from periphery to focus.

```
Sensors ──▶ Glance ──▶ Percept ──▶ salient? ──▶ Focus ──▶ Decision ──▶ Act
(inputs)   (cheap      (what it    (model's      (smart    (speak /
            model,1Hz)  saw)        own call)     model)     do)
              │                                     │
              └───────────────▶ Memory ◀────────────┘
                        working / episodic / semantic
```

## How it works

Two model judgments, no rule engine: **Glance** decides *worth a closer look?*,
**Focus** decides *speak, and what?*. Preferences and recent history are context
fed to the model, not `if` branches. Everything else swaps behind a Protocol:

- **Models** live only in `backends/` (Gemini, OpenAI, Anthropic, Ollama, stub).
  The intended setup is a cheap *local* model for the always-on Glance tier —
  private, free, no rate limits — and a bigger API model when something
  escalates. Run any mix while you build toward that.
- **Inputs** are `Sensor`s (camera, screen, mic, RTSP, replay); **outputs** are
  `Speaker`s (print, TTS, webhook, hardware). One file each, drop-in.
- **Cost is the cascade, not tricks** — the cheap tier gates the expensive one,
  Glance downscales its input, and cadence adapts. No heuristic pre-filters.

## Quickstart — for agents

Extending this repo — as a human or an AI coding agent — comes down to one file
per new piece. The rules below keep the design intact.

**Hard rules.** Breaking these breaks the design.
- **No hand-coded decision rules.** The only judgments are the two model calls
  (`glance.py`, `focus.py`). Salience, urgency, tone — context fed to the model,
  never `if x and y: speak`.
- **Vendor SDKs only in `backends/`** and `speakers/gemini_tts.py`. If you find
  yourself importing `google.genai` or `openai` anywhere else, that's the wrong
  layer.
- **Don't break the Protocols** in `sensors/base.py`, `backends/base.py`,
  `speakers/base.py`. Every concrete class is interchangeable; that's the point.
- **Structured output goes through the schemas** in `schema.py`
  (`PERCEPT_SCHEMA`, `DECISION_SCHEMA`). Don't parse free text and don't add a
  fourth schema unless you're adding a fourth role.

**Where to add things.** One file each — nothing else changes.
- New camera/mic/screen → `sensors/yours.py` implementing `Sensor.stream()`.
- New model provider → `backends/yours.py` implementing `Backend.complete()`.
  Translate `schema` to the provider's native structured-output mechanism.
- New voice output (a speaker, a TV, a phone) → `speakers/yours.py` implementing
  `Speaker.say()`.
- Register the new class in `__main__.py`'s small dispatch.

**Before claiming done.**
1. `python -m pytest -q` — all green.
2. `python -m saccade` with no env — the scripted stub run still works end-to-end.
3. If you touched a real path (camera, model, speaker), actually run it. Tests
   verify code; running verifies the feature.

**Config + secrets.** `saccade/config.py` auto-loads `.env` at import time
(stdlib only, no `python-dotenv`). Real env wins over `.env`. Add new vars as
dataclass fields with `os.environ.get(...)` defaults; don't read env scattered
across modules.

**Avoid.** A fallback that "tries the next backend on error" (a silent provider
switch is a debugging trap). A "smart" cache that compares frames (image diff =
hand-coded salience, see hard rule #1). Refactoring what wasn't asked. Docstring
blocks — one short line max.

## Quickstart — for humans

**Install.**

```bash
git clone https://github.com/jayzuccarelli/saccade && cd saccade
uv venv
source .venv/bin/activate
uv pip install -e '.[camera,screen,audio]'
```

The extras are what let saccade see and hear — `uv pip install -e .` on its own
installs no camera, screen, or mic support. No uv? `python3 -m venv .venv &&
source .venv/bin/activate && pip install -e '.[camera,screen,audio]'` does the same.

**Pick your devices.**

```bash
python -m saccade setup
```

Probes this machine and asks three questions: what to watch or hear (your
built-in camera, an external one, a screen, a mic, or camera + mic together),
which model thinks, and whether it answers as text or out loud. Writes a `.env`
you can edit by hand afterwards. Then start it:

```bash
python -m saccade
```

**Or skip setup entirely.** No key, no camera — runs a scripted scene on the
stdlib alone, so you can see the loop before wiring anything up:

```bash
python -m saccade
```

You'll see Glance/Percept/Focus output in the terminal. That's the whole loop,
just with a stub model and a scripted sensor.

The sections below swap in real models — pair them with a real sensor
(`SACCADE_SENSOR=webcam` / `screen` / `reolink`, further down) or sanity-check
one image with `python -m saccade snapshot pic.jpg`. With no sensor set, the
scripted stub feeds a real model no images (saccade warns if you try).

**Go local, private, free with Ollama** (recommended for the always-on Glance
tier — no API bill, no rate limits, frames never leave your machine):

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:4b
ollama pull gemma3:12b

SACCADE_GLANCE_BACKEND=ollama SACCADE_FOCUS_BACKEND=ollama python -m saccade
```

`gemma3:4b` (~3GB, multimodal) is the Glance default; `gemma3:12b` is smarter and
the Focus default.

Any Ollama vision model works — swap with `SACCADE_GLANCE_MODEL=qwen2.5vl:7b` etc.

**Or use a hosted model.** Pick a provider, add a key, mix and match per tier:

```bash
# Gemini (default: Glance=3.1 Flash-Lite, Focus=3.5 Flash)
uv pip install google-genai
SACCADE_GLANCE_BACKEND=gemini SACCADE_FOCUS_BACKEND=gemini \
  GEMINI_API_KEY=your_key python -m saccade

# Or OpenAI, or Claude — same harness (validated path is Gemini + Ollama;
# OpenAI/Anthropic backends follow each provider's spec but aren't live-tested):
SACCADE_GLANCE_BACKEND=openai SACCADE_FOCUS_BACKEND=anthropic \
  OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python -m saccade
```

Structured output is enforced provider-agnostically: each role declares a JSON
Schema and each backend translates it natively (Ollama `format`, Gemini
`response_json_schema`, OpenAI `response_format`, Claude forced tool-use). Cheap
and smart tiers are independent — Glance on Ollama + Focus on Gemini is a
common combo (private always-on, paid SOTA only when something escalates).

**Point it at one image** (fastest way to sanity-check a key):

```bash
python -m saccade snapshot photo.jpg
```

**See what's plugged into this machine.** Enumerates cameras, screens, mics,
and audio outputs — the `.env lines:` under each section paste verbatim into `.env`:

```bash
python -m saccade devices
```

**Point it at your laptop webcam** (Mac / Linux / Windows — cv2 picks the OS backend):

```bash
uv pip install -e '.[camera]'
SACCADE_SENSOR=webcam python -m saccade
```

Set `SACCADE_WEBCAM_INDEX=1` if you have more than one cam.

macOS: grant Camera access to your terminal app (System Settings > Privacy &
Security > Camera) — approve the prompt on first run, then rerun.

**Point it at your screen** (watch what you're doing — useful for coding-agent
side-cars, meeting notes, "did I actually close that tab"):

```bash
uv pip install -e '.[screen]'
SACCADE_SENSOR=screen SACCADE_SCREEN_INDEX=1 python -m saccade
```

Index 1 is the primary monitor; `python -m saccade devices` lists them all.

macOS: grant Screen Recording to your terminal app (System Settings > Privacy &
Security > Screen Recording) and restart the terminal — without it, mss silently
captures wallpaper-only frames and the agent watches nothing.

**Let it hear** (local microphone). Each tick records a short clip and sends it
to the model, so the agent reacts to sound, not just sight:

```bash
uv pip install -e '.[audio]'
SACCADE_SENSOR=mic SACCADE_GLANCE_BACKEND=gemini SACCADE_FOCUS_BACKEND=gemini \
  GEMINI_API_KEY=... python -m saccade
```

`SACCADE_MIC_INDEX=1` picks a device; `-1` (default) uses the system mic. No
hardware? Test hearing on a file: `python -m saccade snapshot clip.wav`.

Hearing needs an audio-capable backend — **gemini** today. Anthropic and Ollama
(gemma) are vision-only; OpenAI needs a dedicated audio model. Linux also needs
PortAudio (`sudo apt install libportaudio2`); it's bundled in the Mac/Windows
wheels.

**See and hear at once** (webcam + mic fused). Each glance carries both a camera
frame and an audio clip of the same instant, so the agent reasons over sight and
sound together:

```bash
uv pip install -e '.[camera,audio]'
SACCADE_SENSOR=av SACCADE_GLANCE_BACKEND=gemini SACCADE_FOCUS_BACKEND=gemini \
  GEMINI_API_KEY=... python -m saccade
```

**Point it at an RTSP camera** (Reolink or anything that speaks RTSP). Give the
parts and saccade assembles + URL-encodes the URL — a password with `@ : / #`
won't break it:

```bash
uv pip install -e '.[camera]'
SACCADE_SENSOR=reolink \
  SACCADE_RTSP_USER=admin SACCADE_RTSP_PASSWORD='your-pass' \
  SACCADE_RTSP_HOST=192.168.1.100:554 SACCADE_RTSP_PATH=/h264Preview_01_sub \
  python -m saccade
```

Or pass a full `SACCADE_RTSP_URL='rtsp://...'` yourself.

**Stop typing env vars every run.** Copy `.env.example` → `.env` (gitignored)
and fill it in. saccade auto-loads it, so `python -m saccade` just works.

## Layout

| Path | What |
|---|---|
| `schema.py` | the contracts: Frame, Window, Percept, Decision |
| `sensors/` | input streams — `stub`, `webcam`, `screen`, `mic`, `av` (webcam+mic), `reolink`, `replay` (Protocol in `base.py`) |
| `devices.py` | `python -m saccade devices` — enumerate cameras/screens/mics/audio-outs |
| `backends/` | swappable models — `stub`, `ollama` (local, stdlib), `gemini`, `openai`, `anthropic` (the only files that touch a model SDK) |
| `speakers/` | swappable output — `print` (default), `piper` (local TTS, no key), `gemini_tts` (hosted TTS), `home_assistant` (example of a remote-output speaker) |
| `glance.py` | cheap peripheral perceiver → Percept |
| `focus.py` | on-demand deep reasoner → Decision |
| `memory.py` | working / episodic / semantic |
| `loop.py` | the orchestrator |

## Outputs

When Focus decides to act, the message goes to a `Speaker`. "Speaker" is the
generic name for any output — print, audio, an HTTP webhook, a phone push, a
hardware actuator. Default is `print` (no audio, no key, works out of the box).

To actually talk, use Piper: local, offline, no API key, same on Linux, macOS
and Windows. Install it yourself — it's GPL and saccade is MIT, so saccade runs
it as a subprocess rather than depending on it:

```bash
pip install piper-tts
python -m piper.download_voices en_US-lessac-medium
SACCADE_SPEAKER=piper python -m saccade
```

`SACCADE_PIPER_VOICE` picks the voice (`python -m piper.download_voices` with no
argument lists them). Voices download to the working directory; if you run
saccade from elsewhere, point `SACCADE_PIPER_DATA_DIR` at them.

Gemini TTS is the hosted upgrade — better voices, at the cost of a key and a
network round trip per utterance:

```bash
SACCADE_SPEAKER=gemini_tts GEMINI_API_KEY=... python -m saccade
```

The box that *watches* often has no audio out, so by default the speaker writes
the clip (`SACCADE_TTS_DIR`, default `utterances/`) and prints where. Point
`SACCADE_PLAY_CMD` at anything that takes a file path to actually play it —
`aplay`, `afplay`, or a wrapper that pushes audio elsewhere. `SACCADE_TTS_VOICE`
picks the voice (default `Kore`).

To play out a **specific** speaker rather than the OS default, install the audio
extra and set `SACCADE_AUDIO_OUT_INDEX` to a device index from `saccade devices`
— the output twin of picking a mic. It overrides `SACCADE_PLAY_CMD`.

Adding a new output — a phone notification, a webhook, an MQTT message, a
hardware actuator, a smart-home media player — is one `Speaker` class in
`speakers/`. Nothing else changes. See `speakers/home_assistant.py` for a
worked example of a remote-output speaker.

## Cost & cadence

Glance runs constantly, so cost = cadence × price. Reality check:

- **Gemini free tier is tiny** — ~10 requests/min *and* ~20/day for Flash-Lite.
  An always-on agent blows through that in seconds. For real use, enable billing
  (paid tier is thousands/min; a day of 1Hz Flash-Lite watching is a few dollars).
- **Two clocks, running in parallel.** Capture fills the buffer in its own
  coroutine and never pauses while the model thinks. `SACCADE_CAPTURE_FPS` = how
  fast frames stream in (cheap, no API); `SACCADE_GLANCE_FPS` = how often we call
  the model. If `glance_fps >= capture_fps`, every frame gets glanced (lockstep —
  use this for replay). If lower (e.g. `0.14` ≈ every 7s to fit a rate limit),
  the loop samples the latest while the buffer keeps a dense clip for Focus.
  `SACCADE_GLANCE_FPS=0` glances as fast as it can.
- **Glance downscales its input (`SACCADE_GLANCE_MAX_DIM`, default 768)** —
  peripheral vision is low-acuity, so the cheap always-on tier doesn't need full
  resolution to spot that *something* changed. Focus always gets the full-res
  frame, since it reasons carefully and runs rarely. Fewer tokens where it's
  frequent, full detail where it matters.
- **Adaptive cadence** (`SACCADE_ADAPTIVE_CADENCE`, on by default). Instead of a
  fixed rate, Glance emits how soon it should look again — quiet scene → check
  back in seconds; something happening → every tick. The *model* sets the
  interval, so total calls (and cost) fall with no hand-coded rules. It only ever
  *slows* (never faster than `SACCADE_GLANCE_FPS`) and clamps to
  `SACCADE_GLANCE_MAX_INTERVAL` (default 15s). See `loop.py`.

## Develop without a camera or API

`ReplaySensor` plays back a folder of images, so you can iterate on behavior
deterministically — no live feed, no quota:

```bash
SACCADE_SENSOR=replay SACCADE_REPLAY_DIR=frames/ python -m saccade
```

## Evals — tune the hard part with a number, not vibes

The genuinely hard part is *when to speak up* (Glance's salience judgment). Evals
make it measurable: a set of labeled scenes (`evals/scenes.json`) and a scorer.

```bash
SACCADE_GLANCE_BACKEND=gemini GEMINI_API_KEY=... python -m saccade.evals
```

Prints per-case hits/misses and precision / recall / accuracy. Tweak the Glance
prompt, re-run, watch the numbers move. Cases use `scene` (text, for the stub)
or `image` (a file path, for real vision models) — drop labeled frames in and
score a real model the same way.

## Tests

```bash
uv pip install pytest && python -m pytest -q
```

## Status

v0.1: end-to-end loop validated live. RTSP camera → cheap-tier judge (1Hz) →
escalate → bigger model → audio out. Local + hosted backends, swappable sensors
+ speakers, structured output, episodic memory, evals, tests.

v0.2: hearing (mic sensor), sight+sound fused in one glance (`av` sensor),
per-device speaker output, adaptive cadence (Glance paces its own attention),
concurrent Focus (Glance keeps watching while the big model reasons).

Next:
- **More audio backends.** Only Gemini hears today; add others as they ship
  native audio input.
- **Full duplex / barge-in.** Interruptible speech — belongs with a realtime
  streaming path, not the turn-based loop.

License: MIT.
