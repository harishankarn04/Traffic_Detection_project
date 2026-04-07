# -*- coding: utf-8 -*-
"""
prepare_dataset.py — YOLO Fine-Tuning Dataset Preparation

Extracts frames from traffic videos, auto-labels them using pretrained YOLOv8s,
visualises preprocessing steps, and splits into train/val.

Run from inside dl/:
    python fine_tuning/prepare_dataset.py

Reads:  data/raw/video_1.mp4
Writes: data/processed/raw_frames/
        data/processed/auto_labels/
        data/annotated/  (train/val split)
        data/processed/preprocessing_steps.png
        data/processed/class_distribution.png

Original Colab: https://colab.research.google.com/drive/1_btm4YonItZsR385HpUVi9i-kRT5vWih
"""

import cv2
import os
import numpy as np
import shutil
import random
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# COCO class ID → dataset class ID mapping
# car=2, bus=5, truck=7, motorcycle=3, bicycle=1
COCO_TO_YOUR_ID = {2: 0, 5: 1, 7: 2, 3: 3, 1: 4}
CLASS_NAMES = ['Car', 'Bus', 'Truck', 'Motorcycle', 'Bicycle']


def extract_frames(video_path, output_dir, every_n_frames=10):
    os.makedirs(output_dir, exist_ok=True)
    cap   = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = int(cap.get(cv2.CAP_PROP_FPS))
    duration     = total_frames // fps
    print(f"Video Info:")
    print(f"  Total Frames : {total_frames}")
    print(f"  FPS          : {fps}")
    print(f"  Duration     : {duration} seconds")
    print(f"  Sampling     : every {every_n_frames} frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n_frames == 0:
            cv2.imwrite(f"{output_dir}/frame_{saved:04d}.jpg", frame)
            saved += 1
        count += 1

    cap.release()
    print(f"\nExtracted : {saved} frames saved to '{output_dir}'")
    return saved


def auto_label_frames(frames_dir, labels_dir, conf=0.4):
    os.makedirs(labels_dir, exist_ok=True)
    pretrained = YOLO('yolov8s.pt')
    frames     = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    labeled    = 0
    empty      = 0

    for fname in frames:
        results = pretrained(f"{frames_dir}/{fname}", verbose=False)[0]
        lines   = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in COCO_TO_YOUR_ID and float(box.conf[0]) >= conf:
                yid          = COCO_TO_YOUR_ID[cls_id]
                cx, cy, w, h = box.xywhn[0].tolist()
                lines.append(f"{yid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(f"{labels_dir}/{fname.replace('.jpg','.txt')}", 'w') as f:
            f.write('\n'.join(lines))

        if lines: labeled += 1
        else:     empty   += 1

    print(f"Labeled : {labeled} frames with vehicles")
    print(f"Empty   : {empty} frames (no vehicles detected)")


def show_preprocessing_steps(frame_path, label_path, output_path):
    """Visualise preprocessing pipeline — useful for demo/presentation."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Dataset Preprocessing Pipeline', fontsize=16, fontweight='bold')

    original     = cv2.imread(frame_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    axes[0,0].imshow(original_rgb)
    axes[0,0].set_title(f'1. Original Frame\n{original.shape[1]}x{original.shape[0]}')
    axes[0,0].axis('off')

    resized = cv2.resize(original_rgb, (640, 640))
    axes[0,1].imshow(resized)
    axes[0,1].set_title('2. Resized\n640x640')
    axes[0,1].axis('off')

    normalized = resized / 255.0
    axes[0,2].imshow(normalized)
    axes[0,2].set_title('3. Normalized\nPixels: 0.0 - 1.0')
    axes[0,2].axis('off')

    labeled_frame = resized.copy()
    CLASS_COLORS  = {0: (0,255,0), 1: (255,0,0), 2: (0,0,255), 3: (255,255,0), 4: (0,255,255)}
    h, w          = labeled_frame.shape[:2]

    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f.readlines():
                cls, cx, cy, bw, bh = map(float, line.strip().split())
                cls = int(cls)
                x1  = int((cx - bw/2) * w)
                y1  = int((cy - bh/2) * h)
                x2  = int((cx + bw/2) * w)
                y2  = int((cy + bh/2) * h)
                cv2.rectangle(labeled_frame, (x1,y1), (x2,y2), CLASS_COLORS[cls], 2)
                cv2.putText(labeled_frame, CLASS_NAMES[cls], (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[cls], 2)

    axes[0,3].imshow(labeled_frame)
    axes[0,3].set_title('4. Annotated\n(Bounding Boxes)')
    axes[0,3].axis('off')

    aug_configs = [
        ('5. Rotation',   lambda img: np.array(Image.fromarray(img).rotate(10))),
        ('6. Flip',       lambda img: cv2.flip(img, 1)),
        ('7. Brightness', lambda img: np.clip(img * 1.4, 0, 255).astype(np.uint8)),
        ('8. Zoom',       lambda img: cv2.resize(img[32:608, 32:608], (640,640))
                          if img.shape[0] > 64 else img),
    ]

    for i, (title, transform) in enumerate(aug_configs):
        try:
            aug = transform(resized.copy())
            axes[1,i].imshow(aug)
        except:
            axes[1,i].imshow(resized)
        axes[1,i].set_title(title)
        axes[1,i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def split_and_summarize(frames_dir, labels_dir, output_dir, val_split=0.2, dist_output_path=None):
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    random.seed(42)
    random.shuffle(frames)

    split_idx   = int(len(frames) * (1 - val_split))
    train_files = frames[:split_idx]
    val_files   = frames[split_idx:]

    for split, files in [('train', train_files), ('val', val_files)]:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)
        for fname in files:
            shutil.copy(f"{frames_dir}/{fname}", f"{output_dir}/images/{split}/{fname}")
            lname = fname.replace('.jpg', '.txt')
            if os.path.exists(f"{labels_dir}/{lname}"):
                shutil.copy(f"{labels_dir}/{lname}", f"{output_dir}/labels/{split}/{lname}")

    class_counts = {name: 0 for name in CLASS_NAMES}
    for fname in frames:
        lpath = f"{labels_dir}/{fname.replace('.jpg','.txt')}"
        if os.path.exists(lpath):
            with open(lpath) as f:
                for line in f:
                    cls = int(line.split()[0])
                    class_counts[CLASS_NAMES[cls]] += 1

    print("=" * 40)
    print("      DATASET SUMMARY")
    print("=" * 40)
    print(f"Total Images  : {len(frames)}")
    print(f"Train Images  : {len(train_files)} ({100*(1-val_split):.0f}%)")
    print(f"Val Images    : {len(val_files)}   ({100*val_split:.0f}%)")
    print(f"Image Size    : 640 x 640")
    print(f"Classes       : {len(CLASS_NAMES)}")
    print("-" * 40)
    print("Vehicle Distribution:")
    for name, count in class_counts.items():
        bar = '█' * (count // 10)
        print(f"  {name:12s}: {count:5d}  {bar}")
    print("=" * 40)

    plt.figure(figsize=(8, 4))
    plt.bar(class_counts.keys(), class_counts.values(),
            color=['#2196F3','#FF5722','#4CAF50','#FF9800','#9C27B0'])
    plt.title('Vehicle Class Distribution in Dataset')
    plt.xlabel('Vehicle Type')
    plt.ylabel('Count')
    plt.tight_layout()
    if dist_output_path:
        plt.savefig(dist_output_path, dpi=150)
        print(f"Saved: {dist_output_path}")
    plt.close()


if __name__ == "__main__":
    VIDEO_PATH   = "data/raw/video_1.mp4"
    FRAMES_DIR   = "data/processed/raw_frames"
    LABELS_DIR   = "data/processed/auto_labels"
    DATASET_DIR  = "data/annotated"
    PREP_PNG     = "data/processed/preprocessing_steps.png"
    DIST_PNG     = "data/processed/class_distribution.png"

    print("=== Step 1: Extract Frames ===")
    extract_frames(VIDEO_PATH, FRAMES_DIR, every_n_frames=10)

    print("\n=== Step 2: Auto-Label Frames ===")
    auto_label_frames(FRAMES_DIR, LABELS_DIR)

    print("\n=== Step 3: Visualise Preprocessing ===")
    show_preprocessing_steps(
        f"{FRAMES_DIR}/frame_0000.jpg",
        f"{LABELS_DIR}/frame_0000.txt",
        PREP_PNG
    )

    print("\n=== Step 4: Train/Val Split ===")
    split_and_summarize(FRAMES_DIR, LABELS_DIR, DATASET_DIR, dist_output_path=DIST_PNG)
