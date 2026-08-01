"""
Phase 4, Part 4 — samples 30 test images stratified across the 5 cahier-des-
charges classes (CBB, CBSD, CGM, CMD, Healthy) from your local Kaggle Cassava
Leaf Disease dataset, for the "démonstration end-to-end sur 30 images de
test" deliverable required by §7 (P5).

No train.csv is used here — this reads directly from a per-class folder
layout:

    data/raw/train_images/
        CBB/       *.jpg
        CBSD/      *.jpg
        CGM/       *.jpg
        CMD/       *.jpg
        Healthy/   *.jpg

Adjust IMAGES_DIR below if your folder names differ (e.g. full disease names
instead of the short codes) — CLASS_FOLDERS must match your actual
subfolder names exactly, since there's no CSV to cross-check spelling against.
"""
import random
import shutil
from pathlib import Path

IMAGES_DIR = Path("data/raw/train_images")
OUTPUT_DIR = Path("data/e2e_test_set")
IMAGES_PER_CLASS = 6                          # 6 x 5 classes = 30 images
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Must match your actual subfolder names under IMAGES_DIR exactly (case-
# sensitive on Linux/macOS). Confirm this also matches src.configs.SHORT_NAMES
# in your Phase 2 code before trusting "true_label" for anything you report.
CLASS_FOLDERS = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]


def main():
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for class_name in CLASS_FOLDERS:
        class_dir = IMAGES_DIR / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Expected folder not found: {class_dir}. "
                f"Check CLASS_FOLDERS matches your actual subfolder names."
            )

        candidates = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(candidates) < IMAGES_PER_CLASS:
            raise ValueError(
                f"{class_dir} has only {len(candidates)} images, "
                f"need {IMAGES_PER_CLASS}."
            )

        chosen = random.sample(candidates, IMAGES_PER_CLASS)
        for src_path in chosen:
            # Prefix with class name so identically-named files from
            # different class folders (e.g. every folder having "1.jpg")
            # can't collide once copied into one flat output folder.
            dst_name = f"{class_name}_{src_path.name}"
            dst_path = OUTPUT_DIR / dst_name
            shutil.copy(src_path, dst_path)
            manifest_rows.append({"image_path": str(dst_path), "true_label": class_name})

    manifest_lines = ["image_path,true_label"] + [
        f"{r['image_path']},{r['true_label']}" for r in manifest_rows
    ]
    (OUTPUT_DIR / "manifest.csv").write_text("\n".join(manifest_lines) + "\n")

    print(f"Sampled {len(manifest_rows)} images into {OUTPUT_DIR}")
    counts = {c: sum(1 for r in manifest_rows if r["true_label"] == c) for c in CLASS_FOLDERS}
    for class_name, count in counts.items():
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()