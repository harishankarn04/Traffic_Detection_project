#!/usr/bin/env python3
"""
demo_runner.py — ATCS Full Demo
=================================
Runs detection on a YouTube live stream and sends data to BOTH ESP32s.

  Master ESP32 (city)    ← density JSON (vehicle_count, current_density, ...)
  Slave  ESP32 (outskirt) ← city sync JSON (city_density, city_emergency)

Both boards only need their USB cables. NRF modules are physically present
but not used for data — all sync happens via laptop.

Usage:
    python dl/demo_runner.py \
        --master /dev/cu.usbserial-0001 \
        --slave  /dev/cu.usbserial-4 \
        --model  dl/yolov8n_traffic.onnx \
        --lstm   dl/lstm.onnx \
        --video  "https://www.youtube.com/watch?v=1EiC9bvVGnk"

Requirements:
    pip install pyserial onnxruntime yt-dlp opencv-python numpy
"""

import argparse
import json
import sys
import time
import threading
from pathlib import Path
from collections import deque

import serial
import cv2
import numpy as np
import onnxruntime as ort

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


# ─── Config ──────────────────────────────────────────────────────────────────
CLASSES       = ["car", "truck", "bus", "motorcycle"]
CONF_THRESH   = 0.4
INPUT_SIZE    = (640, 640)
WINDOW_SEC    = 30       # Rolling window for LSTM density prediction
SEND_INTERVAL = 3.0      # Seconds between ESP32 updates


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
    # We need exactly 30 timesteps for the LSTM input
    if len(history) < 30:
        # Fallback to current instantaneous density until buffer is full
        return density_label(history[-1] if history else 0)
        
    arr = np.array(history[-30:], dtype=np.float32)
    # Normalize the counts to [0,1] using the same MAX_COUNT=50.0 as training
    arr = np.clip(arr / 50.0, 0.0, 1.0)
    arr = arr.reshape(1, 30, 1)
    
    inp = lstm_sess.get_inputs()[0].name
    out = lstm_sess.run(None, {inp: arr})[0]
    
    # out shape is [1, 4] for the 4 density classes
    predicted_class = np.argmax(out[0])
    
    # 0=LOW, 1=MEDIUM, 2=HIGH, 3=CONGESTED
    labels = ["LOW", "MEDIUM", "HIGH", "CONGESTED"]
    return labels[predicted_class]


def run_yolo(sess, frame):
    """Returns vehicle count and whether emergency vehicle detected."""
    h, w = frame.shape[:2]
    img  = cv2.resize(frame, INPUT_SIZE)
    blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = blob[np.newaxis]

    inp  = sess.get_inputs()[0].name
    out  = sess.run(None, {inp: blob})[0]
    
    # YOLOv8 ONNX raw output: [1, 4+classes, 8400]
    if len(out.shape) == 3:
        out = out[0].T  # Shape becomes (8400, 4+classes)

    count = 0
    emergency = False
    
    boxes = []
    scores = []
    class_ids_list = []

    for row in out:
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        score = classes_scores[class_id]

        if score > CONF_THRESH:
            # COCO vehicle classes: 2=car, 3=motorcycle, 5=bus, 7=truck
            # (adjust if your custom model has different class IDs)
            if class_id in [0, 1, 2, 3, 5, 7]: # added 0,1 just in case custom model is 0=car
                boxes.append(row[:4])
                scores.append(float(score))
                class_ids_list.append(class_id)

    if len(boxes) > 0:
        # Convert cx, cy, w, h -> x1, y1, w, h
        nms_boxes = []
        for b in boxes:
            cx, cy, bw, bh = b
            nms_boxes.append([int(cx - bw/2), int(cy - bh/2), int(bw), int(bh)])
            
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores, CONF_THRESH, 0.4)
        if len(indices) > 0:
            count = len(indices)
            
            x_scale = w / 640.0
            y_scale = h / 640.0
            
            for i in indices.flatten():
                bx, by, bw, bh = nms_boxes[i]
                
                real_x = int(bx * x_scale)
                real_y = int(by * y_scale)
                real_w = int(bw * x_scale)
                real_h = int(bh * y_scale)
                
                cv2.rectangle(frame, (real_x, real_y), (real_x + real_w, real_y + real_h), (0, 255, 0), 2)
                
                # Map class ID to name
                cid = class_ids_list[i]
                # Fallback to COCO names if it's the base model, or Custom names if it's the fine-tuned model
                class_names = {0: "Car", 1: "Bus", 2: "Truck", 3: "Motorcycle", 4: "Auto-Rickshaw", 5: "Bus", 7: "Truck"}
                label = class_names.get(cid, "Vehicle")
                
                # Draw the Accuracy/Confidence value + Label
                accuracy = scores[i] * 100
                cv2.putText(frame, f"{label} {accuracy:.1f}%", (real_x, max(real_y-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Check for "emergency" based on size (since emergency classes aren't in this ONNX file)
                if bw * bh > 0.3 * INPUT_SIZE[0] * INPUT_SIZE[1]:
                    emergency = True
                    cv2.rectangle(frame, (real_x, real_y), (real_x + real_w, real_y + real_h), (0, 0, 255), 4)
                    cv2.putText(frame, f"LARGE VEHICLE {accuracy:.1f}%", (real_x, max(real_y-20, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return count, emergency


# ─── Main ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--master",  required=False, default=None, help="Master ESP32 serial port (optional)")
    p.add_argument("--slave",   required=False, default=None, help="Slave ESP32 serial port (optional)")
    p.add_argument("--model",   required=True, help="Path to YOLOv8 ONNX model")
    p.add_argument("--lstm",    required=True, help="Path to LSTM ONNX model")
    p.add_argument("--video",   required=True, help="YouTube URL or video file path")
    p.add_argument("--baud",    type=int, default=115200)
    return p.parse_args()


def main():
    args = parse_args()

    # Load models
    print("[DEMO] Loading YOLO model...")
    yolo_sess = ort.InferenceSession(args.model)
    
    print(f"[DEMO] Loading LSTM model: {args.lstm}")
    lstm_sess = ort.InferenceSession(args.lstm)

    # Open serial ports if provided
    master_ser = None
    slave_ser = None
    if args.master and args.slave:
        print(f"[DEMO] Opening Master serial: {args.master}")
        master_ser = serial.Serial(args.master, args.baud, timeout=1)
        print(f"[DEMO] Opening Slave  serial: {args.slave}")
        slave_ser  = serial.Serial(args.slave,  args.baud, timeout=1)
        time.sleep(2)  # Wait for ESP32s to boot
    else:
        print("[DEMO] Running in SOFTWARE-ONLY mode (No ESP32s connected)")

    # Open video stream
    print(f"[DEMO] Opening video: {args.video}")
    if args.video.startswith("http"):
        stream_url = get_stream_url(args.video)
    else:
        stream_url = args.video

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("[ERROR] Cannot open video stream.")
        sys.exit(1)

    print("[DEMO] Running. Press Ctrl+C to stop.\n")

    # Start a thread to read from Master and forward to Slave (if connected)
    if master_ser and slave_ser:
        def master_reader():
            while True:
                try:
                    line = master_ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    if line.startswith('{"city_density"'):
                        slave_ser.write((line + "\n").encode())
                        print(f"[→ SLAVE] forwarded from master: {line}")
                    else:
                        print(f"[MASTER LOG] {line}")
                except Exception as e:
                    break
                    
        reader_thread = threading.Thread(target=master_reader, daemon=True)
        reader_thread.start()

    last_send = 0.0
    history = deque(maxlen=30)
    current_density = "LOW"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[DEMO] Stream ended or stalled. Retrying...")
            time.sleep(1)
            continue

        count, emergency = run_yolo(yolo_sess, frame)

        # Update History once per second to simulate real-time sampling rate
        now = time.time()
        if now - last_send >= 1.0:
            history.append(count)
            # We predict density based on the rolling window
            current_density = predict_density(lstm_sess, list(history))

        # Build LSTM Visual UI Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (450, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Base Stats
        cv2.putText(frame, f"CURRENT VEHICLES: {count}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # LSTM UI
        hist_str = str(list(history)[-10:]) if len(history) > 0 else "[]"
        if len(history) < 30:
            cv2.putText(frame, f"BUFFERING LSTM: {len(history)}/30s", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.putText(frame, f"LSTM Window (Last 10s preview):", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        cv2.putText(frame, hist_str, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Color code density
        color = (0, 255, 0)
        if current_density == "MEDIUM": color = (0, 255, 255)
        if current_density == "HIGH": color = (0, 165, 255)
        if current_density == "CONGESTED": color = (0, 0, 255)
            
        cv2.putText(frame, f"LSTM PRED. FUTURE: {current_density}", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("ATCS Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Send to ESP32s at fixed interval
        if now - last_send >= SEND_INTERVAL:
            last_send = now

            # Full density packet → Master
            master_payload = {
                "vehicle_count":   count,
                "current_density": current_density,
                "emergency":       emergency,
            }
            master_json = json.dumps(master_payload) + "\n"
            if master_ser:
                master_ser.write(master_json.encode())
                print(f"[→ MASTER] {master_json.strip()}")
            else:
                # Software-only log
                print(f"[PREDICTION LOG] {master_json.strip()}")
            
            # Note: Slave update is now handled by the master_reader thread
            # which forwards the Master's state (including its BOOT button).

    cap.release()
    cv2.destroyAllWindows()
    if master_ser:
        master_ser.close()
    if slave_ser:
        slave_ser.close()


if __name__ == "__main__":
    main()
