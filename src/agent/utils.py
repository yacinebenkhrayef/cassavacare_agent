"""Small helpers shared across agent nodes."""
import os
from pathlib import Path
import numpy as np
from PIL import Image


def save_gradcam_overlay(overlay: np.ndarray, image_path: str, output_dir: str) -> str:
    """Save a Grad-CAM overlay (HWC uint8 RGB array from GradCAMWrapper) to disk
    and return the saved path.
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem
    out_path = os.path.join(output_dir, f"{stem}_gradcam.png")
    Image.fromarray(overlay.astype("uint8")).save(out_path)
    return out_path


def disease_query_text(pred_disease: str) -> str:
    """Map a snake_case disease label to a natural-language RAG query.
    Your Phase 3 corpus (FAO/PubMed/Wikipedia/IITA) is in English, so the
    query is built in English regardless of the report's French text.
    """
    mapping = {
        "bacterial_blight": "cassava bacterial blight (CBB) treatment and management practices",
        "brown_streak_disease": "cassava brown streak disease (CBSD) treatment and management practices",
        "green_mottle": "cassava green mottle disease (CGM) treatment and management practices",
        "mosaic_disease": "cassava mosaic disease (CMD) treatment and management practices",
    }
    return mapping.get(pred_disease, f"cassava {pred_disease} treatment")