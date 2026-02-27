"""
export_onnx.py — Export YOLOv8 .pt model to ONNX format.

Run once from inside dl/:
    python detection/export_onnx.py

Output: <model_name>.onnx (in the current directory)
Copy to RPi for use with detect_video_onnx.py / detect_live_onnx.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from config.settings import MODEL_NAME

model = YOLO(MODEL_NAME)
model.export(format="onnx", imgsz=640, simplify=True)

onnx_name = MODEL_NAME.replace(".pt", ".onnx")
print(f"Export complete: {onnx_name}")
