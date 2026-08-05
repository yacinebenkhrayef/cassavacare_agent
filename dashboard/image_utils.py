from __future__ import annotations
import io
from PIL import Image


def blend_gradcam_overlay(original_bytes: bytes, gradcam_bytes: bytes, alpha: float) -> bytes:
    """
    Alpha-blends the Grad-CAM PNG returned by the API on top of the original uploaded
    photo, at the given opacity (0.0 = original only, 1.0 = full Grad-CAM image).

    Resizes the Grad-CAM image to match the original's dimensions first — the model
    likely runs inference at a fixed input resolution, so the two are not guaranteed
    to already match pixel-for-pixel.
    """
    original = Image.open(io.BytesIO(original_bytes)).convert("RGBA")
    gradcam = Image.open(io.BytesIO(gradcam_bytes)).convert("RGBA")

    if gradcam.size != original.size:
        gradcam = gradcam.resize(original.size, Image.LANCZOS)

    blended = Image.blend(original, gradcam, alpha)

    buf = io.BytesIO()
    blended.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()