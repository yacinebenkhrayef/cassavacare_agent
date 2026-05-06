import sys, os
sys.path.insert(0, os.path.abspath("."))

import json
import numpy as np
import torch
from PIL import Image

from src.config import CFG, LABEL_MAP
from src.model import build_model, load_model
from src.inference import predict, predict_with_tta
from src.utils import get_device, set_seed


def test_model_architecture():
    """Test model instantiation and forward pass."""
    model = build_model(CFG, register_hooks=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (2, 5), f"Expected (2,5), got {out.shape}"
    print("✅ test_model_architecture passed")


def test_checkpoint_loads():
    """Test that checkpoint file loads correctly."""
    device = get_device()
    model = load_model(CFG["checkpoint_path"], CFG, device)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.inference_mode():
        out = model(dummy)
    assert out.shape == (1, 5)
    print("✅ test_checkpoint_loads passed")


def test_inference_dummy():
    """Test predict() on a synthetic dummy image."""
    device = get_device()
    model = load_model(CFG["checkpoint_path"], CFG, device)

    # Create dummy RGB image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224,224,3), dtype=np.uint8))
    dummy_img.save("/tmp/dummy_leaf.jpg")

    result = predict(model, "/tmp/dummy_leaf.jpg", device,
                     label_map=LABEL_MAP)
    assert "pred_disease" in result
    assert 0.0 <= result["confidence"] <= 1.0
    print(f"✅ test_inference_dummy passed: {result}")


def test_tta_inference():
    """Test predict_with_tta() on dummy image."""
    device = get_device()
    model = load_model(CFG["checkpoint_path"], CFG, device)
    result = predict_with_tta(model, "/tmp/dummy_leaf.jpg", device,
                               n_augs=3, label_map=LABEL_MAP)
    assert result["tta_augs"] == 3
    print(f"✅ test_tta_inference passed: {result}")


def test_artifacts_exist():
    """Verify all expected artifacts are present."""
    required = [
        CFG["checkpoint_path"],
        "data/processed/label_num_to_disease_map.json",
        "data/processed/class_weights.json",
        "data/processed/train_split.csv",
        "data/processed/val_split.csv",
        "data/processed/test_split.csv",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing files: {missing}")
    else:
        print("✅ test_artifacts_exist passed — all required files present")
    assert not missing


if __name__ == "__main__":
    set_seed(42)
    test_model_architecture()
    test_checkpoint_loads()
    test_inference_dummy()
    test_tta_inference()
    test_artifacts_exist()
    print("\n🏆 All tests passed!")
