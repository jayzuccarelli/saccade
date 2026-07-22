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
_RUN_DIR = re.compile(r"run-(\d+)-\d+(?:-(\d+))?")


def _run_order(d: Path) -> tuple[int, int]:
    """Timestamp, then collision suffix: same-second dirs otherwise share a key
    and the sweep would pick its victim by directory-listing order."""
    m = _RUN_DIR.fullmatch(d.name)
    assert m is not None  # callers filter on the same pattern
    return int(m.group(1)), int(m.group(2) or 0)


class Trace:
    def __init__(self, root: str | Path, keep: int = 300, keep_runs: int = 3):
        # A fresh subdir per run: numbering restarts at 1 each process, so writing
        # into a shared dir would interleave two runs' files and the pruner would
        # eat the wrong ones. Pid and a collision suffix on top of the timestamp,
        # because Ctrl-C-and-rerun lands inside the same second, and losing the
        # previous run's evidence is the one failure a trace must not have.
        parent = Path(root)
        base = parent / f"run-{int(time.time())}-{os.getpid()}"
        # Max existing suffix plus one, not the first free name: the sweep frees
        # old names, and reusing one makes the suffix lie about creation order and
        # lets two Trace objects alias one path.
        taken = [0] if base.exists() else []
        taken += [
            int(m.group(2))
            for d in parent.glob(f"{base.name}-*")
            if (m := _RUN_DIR.fullmatch(d.name)) and m.group(2)
        ]
        self.root = Path(f"{base}-{max(taken) + 1}") if taken else base
        self.root.mkdir(parents=True)
        self.keep = keep
        self.n = 0
        # The per-run cap bounds one run; this bounds the habit. A debugging
        # afternoon is a dozen Ctrl-C-and-reruns, each leaving up to `keep` ticks
        # behind, and nobody returns to sweep them.
        #
        # The run just created is exempt rather than sorted: several starts in the
        # same second share a timestamp, and review caught the sweep picking its
        # victim among them by directory-listing order, sometimes the fresh one.
        # Being the run that is about to write is what makes it the newest, and no
        # tie-break on names expresses that (nor survives a clock stepping back).
        others = sorted(
            (
                d
                for d in parent.iterdir()
                if d != self.root and d.is_dir() and _RUN_DIR.fullmatch(d.name)
            ),
            key=_run_order,
        )
        cut = max(0, len(others) - (keep_runs - 1))
        for stale in others[:cut]:
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
