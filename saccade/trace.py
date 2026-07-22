"""Save what the model was actually shown, so a bad tick is evidence, not a debate.

    SACCADE_TRACE_DIR=trace python -m saccade

Every glance writes the exact image bytes it sent (post-downscale, what the model
saw, not what the camera saw), the transcript it attached, and the raw reply.
This exists because "cover the lens and it still says a man is at a desk" has at
least three unrelated causes (a stale frame, a model parroting its context, audio
never arriving), and without the artifacts every one of them looks identical from
the terminal.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

from saccade.schema import Frame

# Exactly the names this module mints, and nothing else: the sweep below deletes
# whatever matches, so a loose pattern would eat a user's own files for the crime
# of living in the trace dir.
_RUN_DIR = re.compile(r"run-(\d+)-\d+(-\d+)?")


class Trace:
    def __init__(self, root: str | Path, keep: int = 300, keep_runs: int = 3):
        # A fresh subdir per run: numbering restarts at 1 each process, so writing
        # into a shared dir would interleave two runs' files and the pruner would
        # eat the wrong ones. Pid and a collision suffix on top of the timestamp,
        # because Ctrl-C-and-rerun lands inside the same second, and losing the
        # previous run's evidence is the one failure a trace must not have.
        parent = Path(root)
        base = parent / f"run-{int(time.time())}-{os.getpid()}"
        path, i = base, 1
        while path.exists():
            path = Path(f"{base}-{i}")
            i += 1
        self.root = path
        self.root.mkdir(parents=True)
        self.keep = keep
        self.n = 0
        # The per-run cap bounds one run; this bounds the habit. A debugging
        # afternoon is a dozen Ctrl-C-and-reruns, each leaving up to `keep` ticks
        # behind, and nobody returns to sweep them.
        runs = sorted(
            (d for d in parent.iterdir() if d.is_dir() and _RUN_DIR.fullmatch(d.name)),
            key=lambda d: int(_RUN_DIR.fullmatch(d.name).group(1)),  # type: ignore[union-attr]
        )
        for stale in runs[:-keep_runs]:
            shutil.rmtree(stale, ignore_errors=True)

    def record(self, role: str, frames: list[Frame], reply: str) -> None:
        """One tick's evidence: the frames as sent, and the answer as received."""
        self.n += 1
        stem = f"{self.n:06d}_{role}"
        images = 0
        for i, f in enumerate(frames):
            if f.image:
                ext = ".png" if f.mime == "image/png" else ".jpg"
                (self.root / f"{stem}_{i}{ext}").write_bytes(f.image)
                images += 1
        meta = {
            "ts": frames[-1].ts if frames else 0.0,
            "images": images,
            "heard": [f.text for f in frames if f.text.strip()],
            "audio_bytes": sum(len(f.audio) for f in frames if f.audio),
            "reply": reply,
        }
        (self.root / f"{stem}.json").write_text(json.dumps(meta, indent=2))
        self._prune()

    def _prune(self) -> None:
        """Cap disk use: a 768px jpeg a second is ~300MB/hour left unchecked."""
        cutoff = self.n - self.keep
        if cutoff <= 0:
            return
        for path in self.root.iterdir():
            head = path.name.split("_", 1)[0]
            if head.isdigit() and int(head) <= cutoff:
                path.unlink(missing_ok=True)
