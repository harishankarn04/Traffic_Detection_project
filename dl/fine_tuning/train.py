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
BEST_PT   = RUNS_DIR / "yolov8n_traffic" / "weights" / "best.pt"

if __name__ == "__main__":
    # NOTE: If number of classes changed, must use yolov8n.pt (not best.pt)
    # best.pt can only be reused when class count stays the same
    model = YOLO("yolov8n.pt")
    print("Starting from yolov8n.pt (class count changed to 8)")

    model.train(
        data=str(DATA_YAML),
        epochs=40,
        imgsz=640,
        batch=8,
        patience=10,    # early stopping
        augment=True,   # mosaic, flip, HSV — helps with CCTV angle variation
        device="cuda",  # NVIDIA GPU (change to "mps" for M1, "cpu" as fallback)
        cache=False,
        workers=4,
        project=str(RUNS_DIR),
        name="yolov8n_traffic",
        exist_ok=True,
    )
