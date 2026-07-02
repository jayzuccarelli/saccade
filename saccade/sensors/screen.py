"""Screen capture as a Sensor. mss is fast, tiny, cross-platform.

    SACCADE_SENSOR=screen SACCADE_SCREEN_INDEX=1 python -m saccade

Index 1 = primary monitor, 2 = second monitor, etc. (0 = all combined).
mss is imported lazily; install with `uv pip install -e '.[screen]'`.

macOS: grant Screen Recording to your terminal app (System Settings >
Privacy & Security) — without it mss silently captures wallpaper-only frames.

"""

from __future__ import annotations

import asyncio
import time

from saccade.schema import Frame


class ScreenSensor:
    def __init__(self, monitor: int = 1, fps: float = 1.0):
        self.monitor = monitor
        self.interval = 1.0 / fps

    async def stream(self):
        import mss
        import mss.tools

        with mss.mss() as sct:
            monitors = sct.monitors
            if not 0 <= self.monitor < len(monitors):
                raise RuntimeError(
                    f"screen index {self.monitor} out of range; found {len(monitors) - 1} monitors"
                )
            target = monitors[self.monitor]
            while True:
                await asyncio.sleep(self.interval)
                shot = sct.grab(target)
                png = mss.tools.to_png(shot.rgb, shot.size)
                yield Frame(ts=time.time(), image=png, mime="image/png")
