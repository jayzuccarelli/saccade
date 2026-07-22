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
import time
from pathlib import Path

from saccade.schema import Frame


class Trace:
    def __init__(self, root: str | Path, keep: int = 300):
        # A fresh subdir per run: numbering restarts at 1 each process, so writing
        # into a shared dir would interleave two runs' files and the pruner would
        # eat the wrong ones.
        self.root = Path(root) / f"run-{int(time.time())}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self.n = 0

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
