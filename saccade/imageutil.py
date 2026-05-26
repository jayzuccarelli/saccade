"""Image helpers. Downscaling is a cost lever: smaller frames = fewer vision
tokens per Glance, which at 1Hz is most of the bill. Not a rule, just a knob."""

from __future__ import annotations

import io


def downscale_jpeg(data: bytes, max_dim: int) -> bytes:
    """Shrink a JPEG so its longest side is <= max_dim. No-op if max_dim is 0
    or the image already fits. Returns JPEG bytes."""
    if not max_dim:
        return data
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    w, h = img.size
    if max(w, h) <= max_dim:
        return data
    scale = max_dim / max(w, h)
    img = img.convert("RGB").resize((max(1, int(w * scale)), max(1, int(h * scale))))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=85)
    return out.getvalue()
