"""
merge_datasets.py — Merge all traffic datasets into a unified YOLOv8 training set.

Unified classes:
    0: car
    1: bus
    2: truck        (truck + heavy truck + light truck + van)
    3: motorcycle   (bike + motorbike + motor)
    4: auto_rickshaw

Run from dl/:
    python fine_tuning/merge_datasets.py
Output: fine_tuning/merged/  (train/ and valid/ splits)
"""

import os
import sys
import shutil
import random
import yaml
from pathlib import Path

BASE = Path(__file__).parent
DATASET_DIR = BASE / "dataset"
OUT_DIR     = BASE / "merged"
SPLIT_RATIO = 0.85   # train fraction when no valid split exists

UNIFIED_CLASSES = ["car", "bus", "truck", "motorcycle", "auto_rickshaw"]

# ---------------------------------------------------------------------------
# Per-dataset class → unified class ID  (None = skip / discard)
# ---------------------------------------------------------------------------
ROBOFLOW_MAPS = {
    "CCTVv2.yolov8": {
        # ['bus', 'car', 'heavy truck', 'light truck', 'motorbike']
        0: 1,   # bus
        1: 0,   # car
        2: 2,   # heavy truck → truck
        3: 2,   # light truck → truck
        4: 3,   # motorbike   → motorcycle
    },
    "UA-DETRAC-DATASET-10K.yolov8": {
        # ['bus', 'car', 'truck', 'van']
        0: 1,   # bus
        1: 0,   # car
        2: 2,   # truck
        3: 2,   # van → truck
    },
    "indian traffic.yolov8": {
        # ['autorickshaw', 'bike', 'car', 'cycle', 'tractor', 'truck']
        0: 4,       # autorickshaw
        1: 3,       # bike → motorcycle
        2: 0,       # car
        3: None,    # cycle → skip
        4: None,    # tractor → skip
        5: 2,       # truck
    },
    # Original Roboflow dataset sitting at dataset root (train/valid/test)
    "_root": {
        # ['bus', 'car', 'cng', 'truck']
        0: 1,   # bus
        1: 0,   # car
        2: 4,   # cng (CNG auto) → auto_rickshaw
        3: 2,   # truck
    },
}

# VisDrone category IDs (1-indexed in annotations)
# 0=ignored,1=pedestrian,2=people,3=bicycle,4=car,5=van,
# 6=truck,7=tricycle,8=awning-tricycle,9=bus,10=motor,11=others
VISDRONE_MAP = {
    4:  0,   # car
    5:  2,   # van → truck
    6:  2,   # truck
    9:  1,   # bus
    10: 3,   # motor → motorcycle
    # all others → skip
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_dirs():
    for split in ("train", "valid"):
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_image(src: Path, split: str, stem: str, suffix: str):
    dst = OUT_DIR / split / "images" / f"{stem}{suffix}"
    shutil.copy2(src, dst)


def write_label(lines: list[str], split: str, stem: str):
    dst = OUT_DIR / split / "labels" / f"{stem}.txt"
    with open(dst, "w") as f:
        f.write("\n".join(lines))


def remap_yolo_label(label_path: Path, class_map: dict) -> list[str]:
    """Read a YOLO label file, remap class IDs, return filtered lines."""
    out = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            orig_cls = int(parts[0])
            new_cls = class_map.get(orig_cls)
            if new_cls is None:
                continue
            out.append(f"{new_cls} " + " ".join(parts[1:]))
    return out


def unique_stem(prefix: str, orig_stem: str) -> str:
    return f"{prefix}_{orig_stem}"


counter = {"train": 0, "valid": 0}


def add_sample(img_path: Path, label_lines: list[str], split: str, stem: str):
    if not label_lines:
        return  # skip images with no relevant vehicles
    suffix = img_path.suffix
    copy_image(img_path, split, stem, suffix)
    write_label(label_lines, split, stem)
    counter[split] += 1


# ---------------------------------------------------------------------------
# Process Roboflow-format datasets
# ---------------------------------------------------------------------------
def process_roboflow(folder_name: str, class_map: dict, split_override: str | None = None):
    """
    split_override: force all images into this split (for root dataset valid/)
    """
    if folder_name == "_root":
        ds_dir = DATASET_DIR
    else:
        ds_dir = DATASET_DIR / folder_name

    prefix = folder_name.replace(" ", "_").replace(".", "_")

    for split_name, out_split in [("train", "train"), ("valid", "valid"), ("test", None)]:
        if out_split is None:
            continue  # skip test sets
        split_dir = ds_dir / split_name
        if not split_dir.exists():
            continue
        img_dir   = split_dir / "images"
        label_dir = split_dir / "labels"
        if not img_dir.exists():
            continue

        images = sorted(img_dir.glob("*.*"))
        for img in images:
            lbl = label_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            lines = remap_yolo_label(lbl, class_map)
            stem  = unique_stem(prefix, img.stem)
            add_sample(img, lines, out_split, stem)

    print(f"  {folder_name}: done")


def process_roboflow_train_only(folder_name: str, class_map: dict):
    """For datasets that only have train/ — do an 80/20 local split."""
    if folder_name == "_root":
        ds_dir = DATASET_DIR
    else:
        ds_dir = DATASET_DIR / folder_name

    prefix  = folder_name.replace(" ", "_").replace(".", "_")
    img_dir = ds_dir / "train" / "images"
    lbl_dir = ds_dir / "train" / "labels"

    if not img_dir.exists():
        return

    images = sorted(img_dir.glob("*.*"))
    random.shuffle(images)
    split_idx = int(len(images) * SPLIT_RATIO)

    for i, img in enumerate(images):
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue
        lines = remap_yolo_label(lbl, class_map)
        stem  = unique_stem(prefix, img.stem)
        out_split = "train" if i < split_idx else "valid"
        add_sample(img, lines, out_split, stem)

    print(f"  {folder_name} (train-only split): done")


# ---------------------------------------------------------------------------
# Process VisDrone  (custom CSV annotation format)
# ---------------------------------------------------------------------------
def visdrone_annotation_to_yolo(ann_path: Path, img_w: int, img_h: int) -> list[str]:
    """
    VisDrone format per line: x,y,w,h,score,category,truncation,occlusion
    category is 1-indexed (see VISDRONE_MAP).
    Converts to YOLO: class cx cy w h (normalised).
    """
    lines = []
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            category = int(parts[5])
            new_cls = VISDRONE_MAP.get(category)
            if new_cls is None:
                continue
            if w <= 0 or h <= 0:
                continue
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            lines.append(f"{new_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def process_visdrone():
    import cv2

    vds_dir = DATASET_DIR / "visdrone"
    if not vds_dir.exists():
        print("  visdrone: folder not found, skipping")
        return

    split_map = {"train": "train", "val": "valid", "testdev": None}

    for src_split, out_split in split_map.items():
        if out_split is None:
            continue
        img_dir = vds_dir / src_split / "images"
        ann_dir = vds_dir / src_split / "annotations"
        if not img_dir.exists():
            continue

        images = sorted(img_dir.glob("*.*"))
        for img_path in images:
            ann_path = ann_dir / (img_path.stem + ".txt")
            if not ann_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            lines = visdrone_annotation_to_yolo(ann_path, img_w, img_h)
            stem  = unique_stem("visdrone", img_path.stem)
            add_sample(img_path, lines, out_split, stem)

    print("  visdrone: done")


# ---------------------------------------------------------------------------
# Write merged data.yaml
# ---------------------------------------------------------------------------
def write_yaml():
    merged_yaml = {
        "train": str((OUT_DIR / "train" / "images").resolve()),
        "val":   str((OUT_DIR / "valid" / "images").resolve()),
        "nc":    len(UNIFIED_CLASSES),
        "names": UNIFIED_CLASSES,
    }
    with open(OUT_DIR / "data.yaml", "w") as f:
        yaml.dump(merged_yaml, f, default_flow_style=False)
    print(f"\n  Wrote: {OUT_DIR / 'data.yaml'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    make_dirs()

    print("Merging datasets...")

    # Roboflow datasets (have train-only, no valid split)
    process_roboflow_train_only("CCTVv2.yolov8",                  ROBOFLOW_MAPS["CCTVv2.yolov8"])
    process_roboflow_train_only("UA-DETRAC-DATASET-10K.yolov8",   ROBOFLOW_MAPS["UA-DETRAC-DATASET-10K.yolov8"])
    process_roboflow_train_only("indian traffic.yolov8",          ROBOFLOW_MAPS["indian traffic.yolov8"])

    # Original Roboflow dataset (has train/valid/test)
    process_roboflow("_root", ROBOFLOW_MAPS["_root"])

    # VisDrone (custom format, has train/val)
    process_visdrone()

    write_yaml()

    print(f"\nDone.")
    print(f"  Train samples : {counter['train']}")
    print(f"  Valid samples : {counter['valid']}")
    print(f"  Output        : {OUT_DIR}")
