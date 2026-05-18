"""
extract_video_data.py — Extract real traffic timeseries from the MAIN_GATE video.

This script processes the video at 1 FPS, runs the YOLOv8 ONNX model,
and saves the true vehicle counts to data/processed/vehicle_counts.csv
for LSTM training.
"""

import cv2
import numpy as np
import onnxruntime as ort
import csv
from datetime import datetime, timedelta
import os
from pathlib import Path

# Config
VIDEO_PATH = Path(__file__).parent.parent.parent / "results_graphs" / "test_video" / "MAIN_GATE_02_20260330075917_20260330090110_3.mp4"
MODEL_PATH = Path(__file__).parent.parent / "yolov8n_traffic.onnx"
OUTPUT_CSV = Path(__file__).parent.parent / "data" / "processed" / "vehicle_counts.csv"
INPUT_SIZE = (640, 640)
CONF_THRESH = 0.4
DENSITY_THRESHOLDS = {"LOW": (0, 10), "MEDIUM": (11, 25), "HIGH": (26, 40), "CONGESTED": (41, 999)}

def density_label(count):
    for label, (lo, hi) in DENSITY_THRESHOLDS.items():
        if lo <= count <= hi:
            return label
    return "CONGESTED"

def run_yolo(sess, frame):
    h, w = frame.shape[:2]
    img = cv2.resize(frame, INPUT_SIZE)
    blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = blob[np.newaxis]

    inp = sess.get_inputs()[0].name
    out = sess.run(None, {inp: blob})[0]
    
    if len(out.shape) == 3:
        out = out[0].T

    boxes = []
    scores = []

    for row in out:
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        score = classes_scores[class_id]

        if score > CONF_THRESH:
            # We only count valid vehicle classes (0=car, 1=bus, 2=truck, 3=motorcycle, 4=auto)
            if class_id in [0, 1, 2, 3, 4, 5, 7]:
                boxes.append(row[:4])
                scores.append(float(score))

    count = 0
    if len(boxes) > 0:
        nms_boxes = []
        for b in boxes:
            cx, cy, bw, bh = b
            nms_boxes.append([int(cx - bw/2), int(cy - bh/2), int(bw), int(bh)])
            
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores, CONF_THRESH, 0.4)
        if len(indices) > 0:
            count = len(indices)
            
    return count

def main():
    print(f"Loading YOLO model: {MODEL_PATH}")
    sess = ort.InferenceSession(str(MODEL_PATH))
    
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 12.0 # fallback

    # We will sample 8 frames per second (e.g. if 24fps, skip every 3 frames)
    # This gives us 8x the data points!
    frame_skip = max(1, int(fps / 8.0))
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime(2026, 3, 30, 8, 0, 0)
    rows = []
    
    frame_idx = 0
    sample_idx = 0
    
    print("Extracting data (1 frame per second)...")
    while True:
        # Read the exact frame we want
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        count = run_yolo(sess, frame)
        # Advance by 1/8th of a second
        ts = start_time + timedelta(seconds=(sample_idx * 0.125))
        
        rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "vehicle_count": count,
            "density_level": density_label(count)
        })
        
        if sample_idx % 400 == 0:
            print(f"Processed {sample_idx} samples... (current count: {count})")
            
        sample_idx += 1
        frame_idx += frame_skip
        
    cap.release()
    
    # Save to CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "vehicle_count", "density_level"])
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Successfully extracted {len(rows)} data points -> {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
