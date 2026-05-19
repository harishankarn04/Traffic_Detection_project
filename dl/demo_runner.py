#!/usr/bin/env python3
"""
demo_runner.py — ATCS Full Demo
=================================
Runs detection on a video/YouTube stream and sends data to BOTH ESP32s.

  Master ESP32 (city)    <- density JSON (vehicle_count, current_density, ...)
  Slave  ESP32 (outskirt) <- city sync JSON (city_density, city_emergency)

Usage:
    python dl/demo_runner.py \\
        --master /dev/cu.usbserial-0001 \\
        --slave  /dev/cu.usbserial-4 \\
        --model  dl/yolov8n_traffic_v2.onnx \\
        --lstm   dl/lstm.onnx \\
        --video  "results_graphs/test_video/hosa_road_1.mp4"

Requirements:
    pip install pyserial onnxruntime ultralytics yt-dlp opencv-python numpy
"""

import argparse
import json
import sys
import time
import threading
from collections import deque

import serial
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


# ─── Config ──────────────────────────────────────────────────────────────────
CONF_THRESH   = 0.20   # Lowered to catch rear-view autos and distant bikes
WINDOW_SEC    = 30     # Rolling window for LSTM density prediction
SEND_INTERVAL = 3.0    # Seconds between ESP32 updates

# Colour for each class label on screen (BGR format)
CLASS_COLORS = {
    "car":           (0,   255, 0),
    "bus":           (255, 165, 0),
    "truck":         (0,   165, 255),
    "van":           (255, 100, 180),   # pink — distinct from truck
    "motorcycle":    (255, 255, 0),
    "auto_rickshaw": (0,   255, 255),
    "ambulance":     (0,   0,   255),
    "fire_truck":    (128, 0,   128),
}
EMERGENCY_CLASSES = {"ambulance", "fire_truck"}


# ─── Helpers ─────────────────────────────────────────────────────────────────
def get_stream_url(youtube_url: str) -> str:
    if yt_dlp is None:
        raise ImportError("yt-dlp not installed. Run: pip install yt-dlp")
    opts = {"format": "best[ext=mp4]/best", "quiet": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]


def density_label(count: int) -> str:
    if count <= 5:   return "LOW"
    if count <= 15:  return "MEDIUM"
    if count <= 30:  return "HIGH"
    return "CONGESTED"


def predict_density(lstm_sess, history: list) -> str:
    """Run LSTM on rolling count history to predict next-window density."""
    if len(history) < 30:
        return density_label(history[-1] if history else 0)

    arr = np.array(history[-30:], dtype=np.float32)
    arr = np.clip(arr / 50.0, 0.0, 1.0)
    arr = arr.reshape(1, 30, 1)

    inp = lstm_sess.get_inputs()[0].name
    out = lstm_sess.run(None, {inp: arr})[0]

    predicted_class = np.argmax(out[0])
    labels = ["LOW", "MEDIUM", "HIGH", "CONGESTED"]
    return labels[predicted_class]


def run_yolo(yolo_model, frame):
    """
    Run Ultralytics YOLO on a frame.
    Uses the full Ultralytics engine for reliable ONNX inference
    (fixes class remapping, NMS, and scaling bugs in raw ONNX parsing).
    Returns: (vehicle_count, emergency_detected, annotated_frame)
    """
    results = yolo_model(frame, conf=CONF_THRESH, verbose=False)[0]

    count = 0
    emergency = False
    annotated = frame.copy()

    for box in results.boxes:
        cls_id   = int(box.cls[0])
        conf     = float(box.conf[0])
        cls_name = yolo_model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        count += 1

        if cls_name in EMERGENCY_CLASSES:
            emergency = True

        color = CLASS_COLORS.get(cls_name, (0, 255, 0))

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label + confidence
        label   = f"{cls_name.replace('_', ' ').title()} {conf*100:.1f}%"
        label_y = max(y1 - 8, 15)
        cv2.putText(annotated, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Extra highlight for emergency vehicles
        if cls_name in EMERGENCY_CLASSES:
            cv2.rectangle(annotated, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 4)
            cv2.putText(annotated, "! EMERGENCY !", (x1, label_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return count, emergency, annotated


# ─── Main ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--master", required=False, default=None, help="Master ESP32 serial port (optional)")
    p.add_argument("--slave",  required=False, default=None, help="Slave ESP32 serial port (optional)")
    p.add_argument("--model",  required=True,  help="Path to YOLOv8 ONNX model")
    p.add_argument("--lstm",   required=True,  help="Path to LSTM ONNX model")
    p.add_argument("--video",  required=True,  help="YouTube URL or video file path")
    p.add_argument("--baud",   type=int, default=115200)
    return p.parse_args()


def main():
    args = parse_args()

    # Load models — Ultralytics engine handles ONNX correctly
    print("[DEMO] Loading YOLO model via Ultralytics engine...")
    yolo_model = YOLO(args.model, task="detect")

    print(f"[DEMO] Loading LSTM model: {args.lstm}")
    lstm_sess = ort.InferenceSession(args.lstm)

    # Serial ports
    master_ser = None
    slave_ser  = None
    if args.master and args.slave:
        print(f"[DEMO] Opening Master serial: {args.master}")
        master_ser = serial.Serial(args.master, args.baud, timeout=1)
        print(f"[DEMO] Opening Slave  serial: {args.slave}")
        slave_ser  = serial.Serial(args.slave, args.baud, timeout=1)
        time.sleep(2)
    else:
        print("[DEMO] Running in SOFTWARE-ONLY mode (No ESP32s connected)")

    # Open video
    print(f"[DEMO] Opening video: {args.video}")
    stream_url = get_stream_url(args.video) if args.video.startswith("http") else args.video
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("[ERROR] Cannot open video stream.")
        sys.exit(1)

    print("[DEMO] Running. Press Q to quit.\n")

    # Forward Master output to Slave
    if master_ser and slave_ser:
        def master_reader():
            while True:
                try:
                    line = master_ser.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith('{"city_density"'):
                        slave_ser.write((line + "\n").encode())
                        print(f"[-> SLAVE] {line}")
                    elif line:
                        print(f"[MASTER LOG] {line}")
                except Exception:
                    break
        threading.Thread(target=master_reader, daemon=True).start()

    last_send = 0.0
    history   = deque(maxlen=30)
    current_density = "LOW"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[DEMO] Stream ended or stalled. Retrying...")
            time.sleep(1)
            continue

        count, emergency, frame = run_yolo(yolo_model, frame)

        now = time.time()
        if now - last_send >= 1.0:
            history.append(count)
            current_density = predict_density(lstm_sess, list(history))

        # ── HUD overlay ──────────────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (460, 175), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, f"VEHICLES DETECTED: {count}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        if len(history) < 30:
            cv2.putText(frame, f"BUFFERING LSTM: {len(history)}/30s", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            hist_str = str(list(history)[-10:])
            cv2.putText(frame, "LSTM Window (last 10s):", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, hist_str, (15, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        color_map = {"LOW": (0, 255, 0), "MEDIUM": (0, 255, 255),
                     "HIGH": (0, 165, 255), "CONGESTED": (0, 0, 255)}
        dens_color = color_map.get(current_density, (0, 255, 0))
        cv2.putText(frame, f"PREDICTED DENSITY: {current_density}", (15, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, dens_color, 2)

        if emergency:
            cv2.putText(frame, "!!! EMERGENCY VEHICLE DETECTED !!!", (15, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.imshow("ATCS Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if now - last_send >= SEND_INTERVAL:
            last_send = now
            payload = {
                "vehicle_count":   count,
                "current_density": current_density,
                "emergency":       emergency,
            }
            msg = json.dumps(payload) + "\n"
            if master_ser:
                master_ser.write(msg.encode())
                print(f"[-> MASTER] {msg.strip()}")
            else:
                print(f"[PREDICTION LOG] {msg.strip()}")

    cap.release()
    cv2.destroyAllWindows()
    if master_ser: master_ser.close()
    if slave_ser:  slave_ser.close()


if __name__ == "__main__":
    main()
