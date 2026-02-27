"""
detect_live_onnx.py — Live camera feed vehicle detection using ONNX Runtime.

For Raspberry Pi with Pi Camera Module (or any USB/CSI camera).

Usage (run from inside dl/):
    python detection/detect_live_onnx.py --model yolov8n.onnx
    python detection/detect_live_onnx.py --model yolov8n.onnx --camera 0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time

import cv2
import numpy as np
import onnxruntime as ort

from config.settings import VEHICLE_CLASSES, CONFIDENCE_THRESHOLD
from density.count_vehicles import count_vehicles
from density.density_logic import build_output
from utils.visualization import draw_detections, draw_hud

INPUT_SIZE = 640
NMS_THRESHOLD = 0.45


def preprocess(frame):
    h, w = frame.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    pad_x = (INPUT_SIZE - new_w) // 2
    pad_y = (INPUT_SIZE - new_h) // 2

    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    img = canvas[:, :, ::-1].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img, scale, pad_x, pad_y


def postprocess(output, scale, pad_x, pad_y):
    preds = output[0].T  # [8400, 84]

    boxes_cxcywh = preds[:, :4]
    class_scores  = preds[:, 4:]

    cls_ids = class_scores.argmax(axis=1)
    confs   = class_scores.max(axis=1)

    mask = (confs >= CONFIDENCE_THRESHOLD) & np.isin(cls_ids, list(VEHICLE_CLASSES.keys()))
    boxes_cxcywh = boxes_cxcywh[mask]
    confs         = confs[mask]
    cls_ids       = cls_ids[mask]

    if len(boxes_cxcywh) == 0:
        return []

    cx, cy, bw, bh = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = ((cx - bw / 2 - pad_x) / scale).astype(int)
    y1 = ((cy - bh / 2 - pad_y) / scale).astype(int)
    x2 = ((cx + bw / 2 - pad_x) / scale).astype(int)
    y2 = ((cy + bh / 2 - pad_y) / scale).astype(int)

    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(boxes_xywh, confs.tolist(), CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    return [
        {"cls": int(cls_ids[int(i)]), "conf": float(confs[int(i)]),
         "box": [int(x1[int(i)]), int(y1[int(i)]), int(x2[int(i)]), int(y2[int(i)])]}
        for i in indices
    ]


def run(model_path, camera_index):
    ort.set_default_logger_severity(3)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: cannot open camera index {camera_index}")
        sys.exit(1)

    print("Press 'q' to quit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        tensor, scale, pad_x, pad_y = preprocess(frame)
        raw_output = session.run(None, {input_name: tensor})[0]
        detections = postprocess(raw_output, scale, pad_x, pad_y)

        count = count_vehicles(detections)
        output = build_output(count)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        frame = draw_detections(frame, detections)
        frame = draw_hud(frame, count, output["current_density"], fps)

        print(json.dumps(output))

        cv2.imshow("Traffic Detection — Live (ONNX)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live vehicle detection (ONNX Runtime)")
    parser.add_argument("--model",  required=True, help="Path to ONNX model (yolov8n.onnx)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    run(args.model, args.camera)
