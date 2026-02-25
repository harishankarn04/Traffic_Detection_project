def count_vehicles(detections):
    """
    Count vehicles from a list of detections.

    Args:
        detections (list[dict]): Each dict has keys "cls", "conf", "box".
                                 Confidence and class filtering is already applied
                                 by the caller (detect_video.py or detect_video_onnx.py).

    Returns:
        int: Number of detected vehicles
    """
    return len(detections)
