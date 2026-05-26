"""Downscaling shrinks big frames and leaves small ones / off-mode alone."""

import io

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from saccade.imageutil import downscale_jpeg  # noqa: E402


def _jpeg(w, h) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (w, h), (120, 120, 120)).save(out, "JPEG")
    return out.getvalue()


def test_downscale_shrinks_to_max_dim():
    big = _jpeg(2000, 1000)
    small = downscale_jpeg(big, 768)
    assert max(Image.open(io.BytesIO(small)).size) == 768
    assert len(small) < len(big)


def test_downscale_is_noop_when_already_small():
    data = _jpeg(400, 300)
    assert downscale_jpeg(data, 768) is data


def test_downscale_off_returns_original():
    data = _jpeg(2000, 1000)
    assert downscale_jpeg(data, 0) is data
