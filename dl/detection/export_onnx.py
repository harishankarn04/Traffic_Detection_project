"""
export_onnx.py — Export YOLOv8 .pt model to ONNX format.

Run once from inside dl/:
    python detection/export_onnx.py

Output: yolov8n.onnx (in the current directory)
Copy this file to RPi for use with detect_video_onnx.py / detect_live_onnx.py
"""

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, simplify=True)

print("Export complete: yolov8n.onnx")
