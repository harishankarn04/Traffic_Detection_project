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
    # Resume from previous best if available, else start from COCO pretrained
    weights = str(BEST_PT) if BEST_PT.exists() else "yolov8n.pt"
    print(f"Loading weights: {weights}")
    model = YOLO(weights)

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
