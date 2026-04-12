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

Usage:
    # Local video
    python tools/draw_roi.py --source dl/data/raw/traffic_cam1.mp4

    # YouTube live stream
    python tools/draw_roi.py --source 'https://www.youtube.com/live/6dp-bvQ7RWo'

    # Custom name override
    python tools/draw_roi.py --source dl/data/raw/traffic_cam1.mp4 --name junction_A

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
import re
import subprocess
import cv2
import numpy as np
from pathlib import Path

TOOLS_DIR = Path(__file__).parent
ROI_VIDEO_DIR = TOOLS_DIR / "roi" / "video"
ROI_STREAM_DIR = TOOLS_DIR / "roi" / "stream"

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


def is_stream(source: str) -> bool:
    """Check if source is a live stream URL."""
    return any(x in source for x in ["youtube.com", "youtu.be", "rtsp://", "http://", "https://"])


def is_youtube(source: str) -> bool:
    return "youtube.com" in source or "youtu.be" in source


def get_youtube_stream_url(url: str) -> str | None:
    """Extract direct stream URL via yt-dlp."""
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
    """Extract YouTube video title for naming."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--get-title", url],
            capture_output=True, text=True, timeout=15,
        )
        title = result.stdout.strip()
        # Clean title to filesystem-safe name
        title = re.sub(r'[^\w\s-]', '', title)
        title = re.sub(r'[\s]+', '_', title).strip('_')
        return title[:50] if title else "stream"
    except Exception:
        return "stream"


def get_stream_name(source: str) -> str:
    """Derive a name from a stream URL."""
    if is_youtube(source):
        return get_youtube_title(source)
    # For RTSP or other URLs, use host + path
    cleaned = source.split("//")[-1]
    cleaned = re.sub(r'[^\w\s-]', '_', cleaned).strip('_')
    return cleaned[:50] if cleaned else "stream"


def grab_frame(source: str):
    """Get first frame from video or stream."""
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
    parser.add_argument("--source", required=True, help="Video file path or stream URL (YouTube, RTSP)")
    parser.add_argument("--name", default=None, help="Custom name override for the lane map file")
    args = parser.parse_args()

    # Create output directories
    ROI_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    ROI_STREAM_DIR.mkdir(parents=True, exist_ok=True)

    # Determine source type, output path, and name
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

    # Grab first frame
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

            # Save zone JSON
            with open(out_path, "w") as f:
                json.dump({
                    "source": source,
                    "zones": zones,
                }, f, indent=2)
            print(f"\nSaved {len(zones)} zones to {out_path}")
            break

        elif key == ord("q"):  # quit without saving
            print("Quit without saving")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
