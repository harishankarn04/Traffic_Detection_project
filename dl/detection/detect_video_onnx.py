"""
detect_video_onnx.py — Run YOLOv8 vehicle detection using ONNX Runtime (no ultralytics).

This is the inference path used on Raspberry Pi.
Test it on your laptop first to validate results match detect_video.py.

Usage (run from inside dl/):
    python detection/detect_video_onnx.py --video data/raw/sample.mp4 --model yolov8n.onnx
    python detection/detect_video_onnx.py --video data/raw/sample.mp4 --model yolov8n.onnx --save
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
import subprocess

from config.settings import VEHICLE_CLASSES, CONFIDENCE_THRESHOLD
from density.count_vehicles import count_vehicles
from density.density_logic import build_output
from utils.visualization import draw_detections, draw_hud

# LSTM predictor — loaded only if --lstm flag is passed
_lstm_predictor = None

INPUT_SIZE = 640
NMS_THRESHOLD = 0.45


def preprocess(frame):
    """
    Prepare a BGR frame for YOLOv8 ONNX inference.

    Returns:
        tensor (np.ndarray): shape [1, 3, 640, 640], float32, values in [0, 1]
        scale (float): scale factor used to resize (needed to map boxes back)
        pad_x (int): horizontal padding in pixels
        pad_y (int): vertical padding in pixels
    """
    h, w = frame.shape[:2]

    # Letterbox resize: keep aspect ratio, pad to square
    scale = INPUT_SIZE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    pad_x = (INPUT_SIZE - new_w) // 2
    pad_y = (INPUT_SIZE - new_h) // 2

    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # BGR → RGB, normalize, NHWC → NCHW, add batch dim
    img = canvas[:, :, ::-1].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img, scale, pad_x, pad_y


def postprocess(output, scale, pad_x, pad_y):
    """
    Convert raw YOLOv8 ONNX output to the common detection format.

    YOLOv8 output shape: [1, 12, 8400]
      12 = 4 (cx, cy, w, h) + 8 class scores (car, bus, truck, motorcycle, auto_rickshaw, ambulance, fire_truck, police)
      8400 = number of anchor candidates

    Returns:
        list[dict]: Filtered, NMS-applied detections.
                    Each dict: {"cls": int, "conf": float, "box": [x1, y1, x2, y2]}
    """
    preds = output[0].T  # [8400, 84]

    boxes_cxcywh = preds[:, :4]
    class_scores  = preds[:, 4:]

    cls_ids = class_scores.argmax(axis=1)
    confs   = class_scores.max(axis=1)

    # Filter by confidence and vehicle class
    mask = (confs >= CONFIDENCE_THRESHOLD) & np.isin(cls_ids, list(VEHICLE_CLASSES.keys()))
    boxes_cxcywh = boxes_cxcywh[mask]
    confs         = confs[mask]
    cls_ids       = cls_ids[mask]

    if len(boxes_cxcywh) == 0:
        return []

    # cx,cy,w,h → x1,y1,x2,y2 (still in 640×640 letterbox space)
    cx, cy, bw, bh = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # Map back to original frame coordinates
    x1 = ((x1 - pad_x) / scale).astype(int)
    y1 = ((y1 - pad_y) / scale).astype(int)
    x2 = ((x2 - pad_x) / scale).astype(int)
    y2 = ((y2 - pad_y) / scale).astype(int)

    # NMS per class
    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(boxes_xywh, confs.tolist(), CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    detections = []
    for i in indices:
        idx = int(i)
        detections.append({
            "cls":  int(cls_ids[idx]),
            "conf": float(confs[idx]),
            "box":  [int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx])],
        })

    return detections

def get_youtube_stream_url(youtube_url):
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "best[ext=mp4]", "-g", youtube_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception as e:
        print("Error extracting YouTube stream:", e)
        return None

def run(video_path, model_path, save_output=False, lstm_path=None, serial_port=None):
    global _lstm_predictor
    ort.set_default_logger_severity(3)

    # Load LSTM predictor if model path provided
    if lstm_path:
        from prediction.lstm_predict import LSTMPredictor
        _lstm_predictor = LSTMPredictor(lstm_path)
        print(f"LSTM predictor loaded: {lstm_path}")

    # Open serial port to ESP32 if specified
    ser = None
    if serial_port:
        import serial
        ser = serial.Serial(serial_port, 115200, timeout=1)
        print(f"Serial port opened: {serial_port}")

    # session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Enable for GPU or NPU
    session = ort.InferenceSession(model_path, providers=["CoreMLExecutionProvider"])
    
    input_name = session.get_inputs()[0].name

    # --- Video source selection (File OR YouTube) ---
    if "youtube.com" in video_path or "youtu.be" in video_path:
        print("YouTube link detected. Extracting stream...")
        stream_url = get_youtube_stream_url(video_path)
        
        if stream_url is None:
            print("Failed to extract YouTube stream.")
            sys.exit(1)
        
        cap = cv2.VideoCapture(stream_url)
    else:
        cap = cv2.VideoCapture(video_path)

    # cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video '{video_path}'")
        sys.exit(1)

    writer = None
    if save_output:
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        out_path = os.path.splitext(video_path)[0] + "_onnx_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))
        print(f"Saving annotated video to: {out_path}")

    prev_time = time.time()
    last_serial_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tensor, scale, pad_x, pad_y = preprocess(frame)
        raw_output = session.run(None, {input_name: tensor})[0]
        detections = postprocess(raw_output, scale, pad_x, pad_y)

        count = count_vehicles(detections)

        predicted = None
        if _lstm_predictor:
            _lstm_predictor.update(count)
            predicted = _lstm_predictor.predict()

        output = build_output(count, predicted)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        frame = draw_detections(frame, detections)
        frame = draw_hud(frame, count, output["current_density"], fps)

        print(json.dumps(output))

        # Send to ESP32 via UART every 2 seconds
        if ser and (curr_time - last_serial_time >= 2.0):
            payload = json.dumps({
                "vehicle_count":      output["vehicle_count"],
                "current_density":    output["current_density"],
                "predicted_density":  output["predicted_density"] or "LOW",
            }) + "\n"
            ser.write(payload.encode())
            last_serial_time = curr_time

        cv2.imshow("Traffic Detection (ONNX)", frame)

        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    if ser:
        ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 vehicle detection on video (ONNX Runtime)")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", required=True, help="Path to ONNX model file (yolov8n.onnx)")
    parser.add_argument("--save", action="store_true", help="Save annotated output video")
    parser.add_argument("--lstm", default=None, help="Path to LSTM ONNX model (optional)")
    parser.add_argument("--serial", default=None, help="Serial port to ESP32 (e.g. /dev/serial0)")
    args = parser.parse_args()

    run(args.video, args.model, save_output=args.save, lstm_path=args.lstm, serial_port=args.serial)
