# COCO dataset vehicle class IDs
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Vehicle count → Density level thresholds
DENSITY_THRESHOLDS = {
    "LOW":       (0, 10),
    "MEDIUM":    (11, 25),
    "HIGH":      (26, 40),
    "CONGESTED": (41, float("inf")),
}

CONFIDENCE_THRESHOLD = 0.4

# YOLOv8 model to use (auto-downloads on first run)
# Options: yolov8n.pt (fastest), yolov8s.pt, yolov8m.pt (more accurate)
MODEL_NAME = "yolov8n.pt"

# MODEL_NAME = "yolov8s.pt"
