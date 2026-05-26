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

## Design rules (non-negotiable)

- **No hand-coded decision rules.** Exactly two model judgments: Glance decides
  *"worth a closer look?"*, Focus decides *"speak, and what?"*. Recency and
  preferences are **context** fed to the model, never `if` branches.
- **Models are swappable.** Vendor SDKs live only in `backends/`. Swap a model =
  swap a Backend. Both tiers are independent.
- **Inputs are swappable.** Any `Sensor` (camera, mic, screen) is a drop-in.
- **Cost is managed by architecture, not hacks** — the cheap-gates-expensive
  cascade, image size, and cadence. No heuristic pre-filters.
- **The brain parallel guides design, it isn't dogma.** Use it where it makes
  things clearer; drop it the moment it's just decoration.

## Run it

No key, no camera, no installs — runs the scripted scene on the stdlib alone:

```bash
cd saccade
python -m saccade
```

With real models (add your key) — any provider, mix and match per tier:

```bash
# Gemini (default models: Glance=2.5 Flash-Lite, Focus=3.5 Flash)
pip install google-genai
SACCADE_GLANCE_BACKEND=gemini SACCADE_FOCUS_BACKEND=gemini \
  GEMINI_API_KEY=your_key python -m saccade

# or OpenAI, or Claude — same harness, no code change
SACCADE_GLANCE_BACKEND=anthropic SACCADE_FOCUS_BACKEND=anthropic \
  ANTHROPIC_API_KEY=your_key python -m saccade
```

Structured output is enforced provider-agnostically: the role declares a JSON
Schema, and each backend translates it natively — Gemini `response_json_schema`,
OpenAI `response_format`, Claude forced tool-use. The cheap and smart tiers are
independent, so you can even run Glance on one provider and Focus on another.

Point it at a single image (fastest way to see a real read once a key is set):

```bash
python -m saccade snapshot photo.jpg
```

With a Reolink (or any RTSP camera):

```bash
pip install opencv-python-headless
SACCADE_SENSOR=reolink \
  SACCADE_RTSP_URL='rtsp://user:pass@camera-ip:554/h264Preview_01_main' \
  python -m saccade
```

## Layout

| Path | What |
|---|---|
| `schema.py` | the contracts: Frame, Window, Percept, Decision |
| `sensors/` | input streams — `stub`, `reolink` (Protocol in `base.py`) |
| `backends/` | swappable models — `stub`, `gemini`, `openai`, `anthropic` (the only files that touch a model SDK) |
| `glance.py` | cheap peripheral perceiver → Percept |
| `focus.py` | on-demand deep reasoner → Decision |
| `memory.py` | working / episodic / semantic |
| `loop.py` | the orchestrator |

## Tests

```bash
pip install pytest && python -m pytest -q
```

## Status

v0: scripted stub end-to-end, vision-only, working memory, prints suggestions.
Multi-provider (Gemini / OpenAI / Claude), resilient loop, snapshot mode, tests.
Next: wire a key, point it at the Reolink, add voice out.
