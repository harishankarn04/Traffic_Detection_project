# Fine-tuned model class IDs (car, bus, truck, motorcycle, auto_rickshaw)
VEHICLE_CLASSES = {
    0: "car",
    1: "bus",
    2: "truck",
    3: "motorcycle",
    4: "auto_rickshaw",
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
