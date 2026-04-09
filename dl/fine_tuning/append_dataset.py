"""
append_dataset.py — Append new dataset to existing merged/ folder without re-merging everything.

Run from dl/ after placing new dataset in fine_tuning/dataset/:
    python fine_tuning/append_dataset.py

This adds new images/labels to the existing merged/train and merged/valid splits.
"""

import os
import shutil
import random
from pathlib import Path

BASE        = Path(__file__).parent
DATASET_DIR = BASE / "dataset"
OUT_DIR     = BASE / "merged"
SPLIT_RATIO = 0.85

# ---------------------------------------------------------------------------
# Dataset to append + class mapping to unified IDs
# 0: car, 1: bus, 2: truck, 3: motorcycle, 4: auto_rickshaw
# ---------------------------------------------------------------------------
APPEND_DATASET = "Vehicle Detection.yolov8"
CLASS_MAP = {
    # ['bus', 'car', 'motorbike', 'truck']
    0: 1,   # bus
    1: 0,   # car
    2: 3,   # motorbike → motorcycle
    3: 2,   # truck
}


def remap_yolo_label(label_path: Path, class_map: dict) -> list[str]:
    out = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            new_cls = class_map.get(int(parts[0]))
            if new_cls is None:
                continue
            out.append(f"{new_cls} " + " ".join(parts[1:]))
    return out


def unique_stem(prefix: str, orig_stem: str) -> str:
    return f"{prefix}_{orig_stem}"


def add_sample(img_path: Path, label_lines: list[str], split: str, stem: str):
    if not label_lines:
        return
    suffix = img_path.suffix
    shutil.copy2(img_path, OUT_DIR / split / "images" / f"{stem}{suffix}")
    with open(OUT_DIR / split / "labels" / f"{stem}.txt", "w") as f:
        f.write("\n".join(label_lines))


if __name__ == "__main__":
    random.seed(42)

    ds_dir  = DATASET_DIR / APPEND_DATASET
    img_dir = ds_dir / "train" / "images"
    lbl_dir = ds_dir / "train" / "labels"
    prefix  = APPEND_DATASET.replace(" ", "_").replace(".", "_")

    if not img_dir.exists():
        print(f"ERROR: {img_dir} not found.")
        raise SystemExit(1)

    images = sorted(img_dir.glob("*.*"))
    random.shuffle(images)
    split_idx = int(len(images) * SPLIT_RATIO)

    added_train = added_valid = 0

    for i, img in enumerate(images):
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue
        lines = remap_yolo_label(lbl, CLASS_MAP)
        stem  = unique_stem(prefix, img.stem)
        split = "train" if i < split_idx else "valid"
        add_sample(img, lines, split, stem)
        if split == "train":
            added_train += 1
        else:
            added_valid += 1

    print(f"Done. Added {added_train} train + {added_valid} valid samples from '{APPEND_DATASET}'.")
    print(f"New totals — train: {len(list((OUT_DIR / 'train' / 'images').glob('*.*')))}, "
          f"valid: {len(list((OUT_DIR / 'valid' / 'images').glob('*.*')))}")
