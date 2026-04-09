"""
append_dataset.py — Append new datasets to existing merged/ folder without re-merging everything.

Unified classes (8 total after emergency vehicles added):
    0: car
    1: bus
    2: truck
    3: motorcycle
    4: auto_rickshaw
    5: ambulance
    6: fire_truck
    7: police

Run from dl/:
    python fine_tuning/append_dataset.py

This adds new images/labels to the existing merged/train and merged/valid splits.
"""

import shutil
import random
from pathlib import Path

BASE        = Path(__file__).parent
DATASET_DIR = BASE / "dataset"
OUT_DIR     = BASE / "merged"
SPLIT_RATIO = 0.85

# ---------------------------------------------------------------------------
# Datasets to append — add new entries here as needed
# Key: dataset folder name (relative to DATASET_DIR, or use absolute path)
# Value: class mapping {original_id: unified_id}  (None = skip)
# ---------------------------------------------------------------------------
DATASETS = {
    # Vehicle Detection dataset — train only
    "Vehicle Detection.yolov8": {
        "base": DATASET_DIR,
        "map": {
            # ['bus', 'car', 'motorbike', 'truck']
            0: 1,   # bus
            1: 0,   # car
            2: 3,   # motorbike → motorcycle
            3: 2,   # truck
        }
    },
    # Indian Emergency Vehicles
    "Indian emergency vehicles.yolov8": {
        "base": DATASET_DIR,
        "map": {
            # 24 classes — only keep vehicles we care about
            # ['Army','Vehicle','ambulance','ambulance_108','ambulance_SOL',
            #  'ambulance_lamp','ambulance_text','auto','bike','bus','car',
            #  'fire_truck','fireladder','firelamp','firesymbol','firewriting',
            #  'hose','police','police_lamp','police_lamp_ON','road_sign',
            #  'tempo traveller','truck','writing']
            0:  None,  # Army → skip
            1:  None,  # Vehicle (generic) → skip
            2:  5,     # ambulance → ambulance
            3:  5,     # ambulance_108 → ambulance
            4:  5,     # ambulance_SOL → ambulance
            5:  None,  # ambulance_lamp → skip
            6:  None,  # ambulance_text → skip
            7:  4,     # auto → auto_rickshaw
            8:  3,     # bike → motorcycle
            9:  1,     # bus → bus
            10: 0,     # car → car
            11: 6,     # fire_truck → fire_truck
            12: None,  # fireladder → skip
            13: None,  # firelamp → skip
            14: None,  # firesymbol → skip
            15: None,  # firewriting → skip
            16: None,  # hose → skip
            17: 7,     # police → police
            18: None,  # police_lamp → skip
            19: None,  # police_lamp_ON → skip
            20: None,  # road_sign → skip
            21: None,  # tempo traveller → skip
            22: 2,     # truck → truck
            23: None,  # writing → skip
        }
    },
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


def add_sample(img_path: Path, label_lines: list[str], split: str, stem: str):
    if not label_lines:
        return
    shutil.copy2(img_path, OUT_DIR / split / "images" / f"{stem}{img_path.suffix}")
    with open(OUT_DIR / split / "labels" / f"{stem}.txt", "w") as f:
        f.write("\n".join(label_lines))


def process_dataset(name: str, base: Path, class_map: dict):
    ds_dir = base / name
    prefix = name.replace(" ", "_").replace(".", "_")

    img_dir = ds_dir / "train" / "images"
    lbl_dir = ds_dir / "train" / "labels"

    if not img_dir.exists():
        print(f"  SKIP {name}: {img_dir} not found")
        return 0, 0

    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(f for f in img_dir.glob("*.*") if f.suffix.lower() in VALID_EXTS)
    random.shuffle(images)
    split_idx = int(len(images) * SPLIT_RATIO)

    added_train = added_valid = 0
    for i, img in enumerate(images):
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue
        lines = remap_yolo_label(lbl, class_map)
        stem  = f"{prefix}_{img.stem}"
        split = "train" if i < split_idx else "valid"
        add_sample(img, lines, split, stem)
        if split == "train":
            added_train += 1
        else:
            added_valid += 1

    print(f"  {name}: +{added_train} train, +{added_valid} valid")
    return added_train, added_valid


if __name__ == "__main__":
    random.seed(42)

    print("Appending datasets to merged/...")
    total_train = total_valid = 0

    for name, cfg in DATASETS.items():
        t, v = process_dataset(name, cfg["base"], cfg["map"])
        total_train += t
        total_valid += v

    print(f"\nAdded: {total_train} train + {total_valid} valid")
    print(f"Merged totals — train: {len(list((OUT_DIR / 'train' / 'images').glob('*.*')))}, "
          f"valid: {len(list((OUT_DIR / 'valid' / 'images').glob('*.*')))}")
