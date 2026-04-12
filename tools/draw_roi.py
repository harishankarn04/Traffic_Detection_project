"""
draw_roi.py — Interactive ROI zone drawing tool for lane-wise vehicle detection.

Run once per camera/video to define approach zones. Saves zones to a JSON file
that detect_video_onnx.py reads for per-zone counting.

Output structure:
    tools/roi/video/       — lane maps for local video files
    tools/roi/stream/      — lane maps for live streams (YouTube, RTSP, etc.)

Naming:
    Video:  tools/roi/video/<vidname>_laneMap.json
    Stream: tools/roi/stream/<streamname>_laneMap.json

Zone types:
    - direction lanes: e.g. "north_approaching_lane1", "south_leaving_lane2"
    - middle zone: roundabout or normal junction center

Usage:
    python tools/draw_roi.py --source dl/data/raw/traffic_cam1.mp4
    python tools/draw_roi.py --source 'https://www.youtube.com/live/6dp-bvQ7RWo'
    python tools/draw_roi.py --source dl/data/raw/traffic_cam1.mp4 --name junction_A

Controls:
    - Click       — add polygon point
    - 'u'         — undo last point
    - 'n'         — finish zone and name it (prompted in terminal)
    - 'r'         — reset current zone
    - 'd'         — delete last saved zone
    - 's'         — save all zones to JSON and exit
    - 'q'         — quit without saving
"""

import argparse
import json
import re
import subprocess
import cv2
import numpy as np
from pathlib import Path

TOOLS_DIR = Path(__file__).parent
ROI_VIDEO_DIR = TOOLS_DIR / "roi" / "video"
ROI_STREAM_DIR = TOOLS_DIR / "roi" / "stream"

points = []
zones = {}       # flat dict: zone_name → {"points": [...], "type": ..., "direction": ..., "lane": ...}
frame_display = None
frame_clean = None

# Colors per direction
DIR_COLORS = {
    "north": (0, 255, 0),      # green
    "south": (0, 0, 255),      # red
    "east":  (255, 255, 0),    # cyan
    "west":  (0, 255, 255),    # yellow
    "middle": (255, 0, 255),   # magenta
}
FALLBACK_COLOR = (200, 200, 200)


def is_stream(source: str) -> bool:
    return any(x in source for x in ["youtube.com", "youtu.be", "rtsp://", "http://", "https://"])


def is_youtube(source: str) -> bool:
    return "youtube.com" in source or "youtu.be" in source


def get_youtube_stream_url(url: str) -> str | None:
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "best[height<=720]", "-g", url],
            capture_output=True, text=True, timeout=30,
        )
        stream = result.stdout.strip()
        return stream if stream else None
    except Exception as e:
        print(f"yt-dlp error: {e}")
        return None


def get_youtube_title(url: str) -> str:
    try:
        result = subprocess.run(
            ["yt-dlp", "--get-title", url],
            capture_output=True, text=True, timeout=15,
        )
        title = result.stdout.strip()
        title = re.sub(r'[^\w\s-]', '', title)
        title = re.sub(r'[\s]+', '_', title).strip('_')
        return title[:50] if title else "stream"
    except Exception:
        return "stream"


def get_stream_name(source: str) -> str:
    if is_youtube(source):
        return get_youtube_title(source)
    cleaned = source.split("//")[-1]
    cleaned = re.sub(r'[^\w\s-]', '_', cleaned).strip('_')
    return cleaned[:50] if cleaned else "stream"


def grab_frame(source: str):
    actual_source = source
    if is_youtube(source):
        print("YouTube link detected. Extracting stream...")
        stream_url = get_youtube_stream_url(source)
        if not stream_url:
            print("Failed to extract YouTube stream.")
            return None
        actual_source = stream_url
    cap = cv2.VideoCapture(actual_source)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_zone_color(zone_info):
    direction = zone_info.get("direction", "")
    return DIR_COLORS.get(direction, FALLBACK_COLOR)


def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw()


def redraw():
    global frame_display
    frame_display = frame_clean.copy()

    # Draw saved zones
    for name, info in zones.items():
        color = get_zone_color(info)
        poly = np.array(info["points"], np.int32)
        overlay = frame_display.copy()
        cv2.fillPoly(overlay, [poly], (*color, 50))
        cv2.addWeighted(overlay, 0.2, frame_display, 0.8, 0, frame_display)
        cv2.polylines(frame_display, [poly], True, color, 2)
        cx = int(np.mean([p[0] for p in info["points"]]))
        cy = int(np.mean([p[1] for p in info["points"]]))
        cv2.putText(frame_display, name, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw current in-progress zone
    if points:
        for pt in points:
            cv2.circle(frame_display, pt, 5, (255, 255, 255), -1)
        if len(points) > 1:
            cv2.polylines(frame_display, [np.array(points, np.int32)], False, (255, 255, 255), 2)

    # Instructions
    y = 25
    instructions = [
        "Click: add point | u: undo | n: finish zone | r: reset",
        "d: delete last | s: save & exit | q: quit",
        f"Zones: {len(zones)} | Current points: {len(points)}",
    ]
    for text in instructions:
        cv2.putText(frame_display, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20


def prompt_zone_info() -> dict | None:
    """Prompt user in terminal for zone metadata."""
    # Flush any buffered stdin from OpenCV keypresses
    import time
    time.sleep(0.3)
    try:
        import sys, termios
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    except (ImportError, AttributeError):
        pass

    print("\n--- Zone Setup ---")
    print("Zone types: approaching, leaving, middle")
    zone_type = input("Type (approaching/leaving/middle): ").strip().lower()

    if zone_type == "middle":
        middle_type = input("Middle type (roundabout/normal): ").strip().lower()
        if middle_type not in ("roundabout", "normal"):
            middle_type = "normal"
        name = f"middle_{middle_type}"
        return {
            "name": name,
            "type": "middle",
            "middle_type": middle_type,
            "direction": "middle",
            "lane": 0,
        }

    if zone_type not in ("approaching", "leaving"):
        print("Invalid type. Use: approaching, leaving, or middle")
        return None

    direction = input("Direction (north/south/east/west): ").strip().lower()
    if direction not in ("north", "south", "east", "west"):
        print("Invalid direction.")
        return None

    lane = input("Lane number (1, 2, 3...): ").strip()
    if not lane.isdigit():
        print("Invalid lane number.")
        return None

    name = f"{direction}_{zone_type}_lane{lane}"
    return {
        "name": name,
        "type": zone_type,
        "direction": direction,
        "lane": int(lane),
    }


def main():
    global points, frame_display, frame_clean

    parser = argparse.ArgumentParser(description="Draw ROI zones for lane-wise detection")
    parser.add_argument("--source", required=True, help="Video file path or stream URL (YouTube, RTSP)")
    parser.add_argument("--name", default=None, help="Custom name override for the lane map file")
    args = parser.parse_args()

    ROI_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    ROI_STREAM_DIR.mkdir(parents=True, exist_ok=True)

    source = args.source
    stream = is_stream(source)

    if args.name:
        map_name = args.name
    elif stream:
        map_name = get_stream_name(source)
    else:
        map_name = Path(source).stem

    if stream:
        out_path = ROI_STREAM_DIR / f"{map_name}_laneMap.json"
    else:
        out_path = ROI_VIDEO_DIR / f"{map_name}_laneMap.json"

    print(f"Source type: {'stream' if stream else 'video'}")
    print(f"Lane map will save to: {out_path}")

    frame = grab_frame(source)
    if frame is None:
        print(f"Error: cannot read from '{source}'")
        return

    frame_clean = frame.copy()
    frame_display = frame.copy()

    cv2.namedWindow("Draw ROI Zones")
    cv2.setMouseCallback("Draw ROI Zones", mouse_callback)

    redraw()

    while True:
        cv2.imshow("Draw ROI Zones", frame_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("u"):  # undo last point
            if points:
                points.pop()
                redraw()

        elif key == ord("n"):  # finish current zone
            if len(points) < 3:
                print("Need at least 3 points for a zone")
                continue
            info = prompt_zone_info()
            if info:
                name = info.pop("name")
                info["points"] = list(points)
                zones[name] = info
                print(f"  Saved zone '{name}' ({info['type']}, {info.get('direction','')}, lane {info.get('lane','-')})")
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

            with open(out_path, "w") as f:
                json.dump({
                    "source": source,
                    "zones": zones,
                }, f, indent=2)
            print(f"\nSaved {len(zones)} zones to {out_path}")
            break

        elif key == ord("q"):
            print("Quit without saving")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
