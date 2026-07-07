"""Replay a folder of images as a Sensor — deterministic, API-free tuning.

Lets you record real scenes once, then iterate on prompts/behavior against the
same footage without a live camera or burning API quota. Reads raw file bytes;
no opencv needed.
"""

from __future__ import annotations

import asyncio
import glob
import itertools
import os
import time
from collections.abc import AsyncIterator

from saccade.schema import Frame


class ReplaySensor:
    def __init__(self, folder: str, fps: float = 1.0, loop: bool = False):
        self.folder = folder
        self.interval = 1.0 / fps
        self.loop = loop

    def _paths(self) -> list[str]:
        paths: list[str] = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths += glob.glob(os.path.join(self.folder, ext))
        return sorted(paths)

    async def stream(self) -> AsyncIterator[Frame]:
        paths = self._paths()
        if not paths:
            raise RuntimeError(f"no images (.jpg/.jpeg/.png) in {self.folder}")
        seq = itertools.cycle(paths) if self.loop else iter(paths)
        for p in seq:
            with open(p, "rb") as f:
                data = f.read()
            mime = "image/png" if p.lower().endswith(".png") else "image/jpeg"
            yield Frame(ts=time.time(), image=data, mime=mime, meta={"path": p})
            await asyncio.sleep(self.interval)
