"""Image helpers. Downscaling is a cost lever: smaller frames = fewer vision
tokens per Glance, which at 1Hz is most of the bill. Not a rule, just a knob."""

from __future__ import annotations

import io

_warned_no_pil = False


def downscale(data: bytes, max_dim: int) -> tuple[bytes, str] | None:
    """Shrink any raster image (JPEG, PNG, ...) so its longest side is <= max_dim,
    re-encoded as JPEG (the cheap tier doesn't need lossless). Returns
    (bytes, "image/jpeg"), or None when the frame should pass through unchanged:
    max_dim is 0, the image already fits, or Pillow isn't installed — then warn
    once and keep running; full-size frames still work, they just cost more."""
    global _warned_no_pil
    if not max_dim:
        return None
    try:
        from PIL import Image
    except ImportError:
        if not _warned_no_pil:
            print("[saccade] pillow not installed — sending full-size frames (uv pip install pillow)")
            _warned_no_pil = True
        return None

    img = Image.open(io.BytesIO(data))
    w, h = img.size
    if max(w, h) <= max_dim:
        return None
    scale = max_dim / max(w, h)
    img = img.convert("RGB").resize((max(1, int(w * scale)), max(1, int(h * scale))))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=85)
    return out.getvalue(), "image/jpeg"
