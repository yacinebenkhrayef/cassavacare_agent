from __future__ import annotations
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

from src.preprocess import get_inference_transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def predict(model, img_path: str | Path, device: torch.device,
            img_size: int = 224, label_map: dict = None) -> dict:
    """Single-image inference (no TTA).

    Returns: {pred_class (int), pred_disease (str), confidence (float),
              probabilities (list[float])}
    """
    tf = get_inference_transforms(img_size)
    img = Image.open(img_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred = probs.argmax().item()
    conf = probs[pred].item()

    disease = label_map[str(pred)] if label_map else str(pred)
    return {
        "pred_class"    : pred,
        "pred_disease"  : disease,
        "confidence"    : round(conf, 4),
        "probabilities" : [round(p, 4) for p in probs.tolist()],
    }


def predict_with_tta(model, img_path: str | Path, device: torch.device,
                      n_augs: int = 5, img_size: int = 224,
                      label_map: dict = None) -> dict:
    """Single-image inference with Test-Time Augmentation.

    Averages probabilities from original + (n_augs-1) augmented versions.
    """
    tf = get_inference_transforms(img_size)
    img = Image.open(img_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)

    tta_transforms = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=5),
    ]

    model.eval()
    all_probs = []

    with torch.inference_mode():
        # Original
        probs_orig = torch.softmax(model(tensor), dim=1)
        all_probs.append(probs_orig)

        # Augmented versions
        for _ in range(n_augs - 1):
            aug = tensor.clone()
            for t in tta_transforms:
                aug = t(aug)
            all_probs.append(torch.softmax(model(aug), dim=1))

    avg_probs = torch.stack(all_probs).mean(dim=0)[0]
    pred = avg_probs.argmax().item()
    conf = avg_probs[pred].item()

    disease = label_map[str(pred)] if label_map else str(pred)
    return {
        "pred_class"    : pred,
        "pred_disease"  : disease,
        "confidence"    : round(conf, 4),
        "probabilities" : [round(p, 4) for p in avg_probs.tolist()],
        "tta_augs"      : n_augs,
    }
