"""Downscaling shrinks big frames of any raster format and leaves small ones /
off-mode / missing-Pillow alone (returns None = pass the frame through)."""

import io

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from saccade.imageutil import downscale  # noqa: E402


def _img(w, h, fmt) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (w, h), (120, 120, 120)).save(out, fmt)
    return out.getvalue()


def test_downscale_shrinks_jpeg_to_max_dim():
    big = _img(2000, 1000, "JPEG")
    small, mime = downscale(big, 768)
    assert mime == "image/jpeg"
    assert max(Image.open(io.BytesIO(small)).size) == 768
    assert len(small) < len(big)


def test_downscale_shrinks_png_and_reencodes_jpeg():
    # The screen sensor emits PNG: Retina grabs are multi-MB and must not
    # bypass the cheap tier's downscale.
    big = _img(2880, 1800, "PNG")
    small, mime = downscale(big, 768)
    assert mime == "image/jpeg"
    assert max(Image.open(io.BytesIO(small)).size) == 768
    assert len(small) < len(big)


def test_downscale_is_noop_when_already_small():
    assert downscale(_img(400, 300, "JPEG"), 768) is None


def test_downscale_off_returns_none():
    assert downscale(_img(2000, 1000, "JPEG"), 0) is None
