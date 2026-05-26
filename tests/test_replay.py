"""ReplaySensor yields a Frame per image file, in order."""

import asyncio

from saccade.sensors.replay import ReplaySensor


def test_replay_yields_frames_in_order(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff one")
    (tmp_path / "b.png").write_bytes(b"\x89PNG two")

    async def collect():
        out = []
        async for frame in ReplaySensor(str(tmp_path), fps=1000).stream():
            out.append(frame)
        return out

    frames = asyncio.run(collect())
    assert len(frames) == 2
    assert frames[0].image == b"\xff\xd8\xff one"  # a.jpg first (sorted)
    assert frames[0].mime == "image/jpeg"
    assert frames[1].mime == "image/png"
