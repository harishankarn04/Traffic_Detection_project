"""
draw_roi.py — Interactive ROI zone drawing tool for lane-wise vehicle detection.

Run once per camera to define approach zones. Saves zones to a JSON file
that detect_video_onnx.py reads for per-zone counting.

Usage:
    python tools/draw_roi.py --source data/raw/video.mp4
    python tools/draw_roi.py --source 0                        # webcam
    python tools/draw_roi.py --source data/raw/video.mp4 --output zones.json

Controls:
    - Click points to draw a polygon (min 3 points)
    - Press 'n' to finish current zone and name it
    - Press 'r' to reset current zone
    - Press 'd' to delete last saved zone
    - Press 's' to save all zones to JSON and exit
    - Press 'q' to quit without saving
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path

points = []
zones = {}
frame_display = None
frame_clean = None

COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
]


def mouse_callback(event, x, y, flags, param):
    global points, frame_display

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw()


def redraw():
    global frame_display
    frame_display = frame_clean.copy()

    # Draw existing saved zones
    for i, (name, pts) in enumerate(zones.items()):
        color = COLORS[i % len(COLORS)]
        poly = np.array(pts, np.int32)
        cv2.polylines(frame_display, [poly], True, color, 2)
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(frame_display, name, (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw current in-progress zone
    if points:
        for pt in points:
            cv2.circle(frame_display, pt, 5, (255, 255, 255), -1)
        if len(points) > 1:
            cv2.polylines(frame_display, [np.array(points, np.int32)], False, (255, 255, 255), 2)

    # Instructions
    y = 25
    instructions = [
        "Click: add point | n: finish zone | r: reset",
        "d: delete last | s: save & exit | q: quit",
        f"Zones: {len(zones)} | Current points: {len(points)}",
    ]
    for text in instructions:
        cv2.putText(frame_display, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20


def main():
    global points, frame_display, frame_clean

    parser = argparse.ArgumentParser(description="Draw ROI zones for lane-wise detection")
    parser.add_argument("--source", required=True, help="Video file path or camera index")
    parser.add_argument("--output", default="zones.json", help="Output JSON path (default: zones.json)")
    args = parser.parse_args()

    # Get first frame
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: cannot read from '{args.source}'")
        return

    frame_clean = frame.copy()
    frame_display = frame.copy()

    cv2.namedWindow("Draw ROI Zones")
    cv2.setMouseCallback("Draw ROI Zones", mouse_callback)

    redraw()

    while True:
        cv2.imshow("Draw ROI Zones", frame_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("n"):  # finish current zone
            if len(points) < 3:
                print("Need at least 3 points for a zone")
                continue
            name = input("Zone name (e.g. north, south, east, west): ").strip()
            if name:
                zones[name] = list(points)
                print(f"  Saved zone '{name}' with {len(points)} points")
            points = []
            redraw()

        elif key == ord("r"):  # reset current zone
            points = []
            redraw()

        elif key == ord("d"):  # delete last zone
            if zones:
                last = list(zones.keys())[-1]
                del zones[last]
                print(f"  Deleted zone '{last}'")
                redraw()

        elif key == ord("s"):  # save and exit
            if not zones:
                print("No zones to save")
                continue
            out_path = Path(args.output)
            with open(out_path, "w") as f:
                json.dump(zones, f, indent=2)
            print(f"\nSaved {len(zones)} zones to {out_path}")
            break

        elif key == ord("q"):  # quit without saving
            print("Quit without saving")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
