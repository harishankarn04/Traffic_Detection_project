import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from config.settings import VEHICLE_CLASSES

# Colour per vehicle class (BGR)
CLASS_COLORS = {
    2: (0, 255, 0),    # car — green
    3: (255, 0, 255),  # motorcycle — magenta
    5: (0, 165, 255),  # bus — orange
    7: (0, 0, 255),    # truck — red
}


def draw_detections(frame, detections):
    """
    Draw bounding boxes and class labels for each detection.

    Args:
        frame: BGR image (numpy array)
        detections (list[dict]): Each dict has "cls" (int), "conf" (float), "box" ([x1,y1,x2,y2]).

    Returns:
        frame with boxes drawn in-place
    """
    for d in detections:
        x1, y1, x2, y2 = map(int, d["box"])
        color = CLASS_COLORS.get(d["cls"], (200, 200, 200))
        label = f"{VEHICLE_CLASSES[d['cls']]} {d['conf']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return frame


def draw_hud(frame, vehicle_count, density_level, fps):
    """
    Draw a HUD overlay showing vehicle count, density level, and FPS.

    Args:
        frame: BGR image (numpy array)
        vehicle_count (int): Total vehicles detected
        density_level (str): LOW / MEDIUM / HIGH / CONGESTED
        fps (float): Current frames per second

    Returns:
        frame with HUD drawn in-place
    """
    density_colors = {
        "LOW":       (0, 255, 0),
        "MEDIUM":    (0, 255, 255),
        "HIGH":      (0, 165, 255),
        "CONGESTED": (0, 0, 255),
    }
    density_color = density_colors.get(density_level, (255, 255, 255))

    lines = [
        (f"Vehicles: {vehicle_count}", (255, 255, 255)),
        (f"Density:  {density_level}", density_color),
        (f"FPS:      {fps:.1f}", (200, 200, 200)),
    ]

    y = 28
    for text, color in lines:
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)  # shadow
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 28

    return frame
