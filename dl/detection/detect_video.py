"""
detect_video.py — Run YOLOv8 vehicle detection on a video file (PyTorch path).

Usage (run from inside dl/):
    python detection/detect_video.py --video data/raw/sample.mp4
    python detection/detect_video.py --video data/raw/sample.mp4 --save
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time

import cv2
from ultralytics import YOLO

from config.settings import MODEL_NAME, VEHICLE_CLASSES, CONFIDENCE_THRESHOLD
from density.count_vehicles import count_vehicles
from density.density_logic import build_output
from utils.visualization import draw_detections, draw_hud


def ultralytics_to_detections(result):
    """
    Convert a single ultralytics result object to the common detection format.

    Returns:
        list[dict]: Each dict has "cls" (int), "conf" (float), "box" ([x1,y1,x2,y2]).
                    Only vehicle classes above confidence threshold are included.
    """
    detections = []
    boxes = result.boxes

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf   = float(boxes.conf[i].item())

        if cls_id in VEHICLE_CLASSES and conf >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            detections.append({"cls": cls_id, "conf": conf, "box": [x1, y1, x2, y2]})

    return detections


def run(video_path, save_output=False):
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video '{video_path}'")
        sys.exit(1)

    writer = None
    if save_output:
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        out_path = os.path.splitext(video_path)[0] + "_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))
        print(f"Saving annotated video to: {out_path}")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)
        detections = ultralytics_to_detections(results[0])

        count = count_vehicles(detections)
        output = build_output(count)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        frame = draw_detections(frame, detections)
        frame = draw_hud(frame, count, output["current_density"], fps)

        print(json.dumps(output))

        cv2.imshow("Traffic Detection (PyTorch)", frame)

        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 vehicle detection on video (PyTorch)")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--save", action="store_true", help="Save annotated output video")
    args = parser.parse_args()

    run(args.video, save_output=args.save)
