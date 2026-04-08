"""
train.py — Fine-tune YOLOv8n on merged traffic dataset.

Run from dl/:
    python fine_tuning/train.py

Output: fine_tuning/runs/yolov8n_traffic/weights/best.pt
"""

from ultralytics import YOLO
from pathlib import Path

DATA_YAML = Path(__file__).parent / "merged" / "data.yaml"
RUNS_DIR  = Path(__file__).parent / "runs"

model = YOLO("yolov8n.pt")  # COCO pretrained — auto-downloads if not present

model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=8,        # safe for 6GB VRAM (RTX 4050)
    patience=10,    # early stopping
    augment=True,   # mosaic, flip, HSV — helps with CCTV angle variation
    device="cuda",  # NVIDIA GPU (change to "mps" for M1, "cpu" as fallback)
    cache=False,
    workers=4,
    project=str(RUNS_DIR),
    name="yolov8n_traffic",
    exist_ok=True,
)
