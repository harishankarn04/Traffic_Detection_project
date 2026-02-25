import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
from config.settings import DENSITY_THRESHOLDS


def get_density_level(vehicle_count):
    """
    Map a vehicle count to a density level string.

    Args:
        vehicle_count (int): Number of vehicles detected in the frame

    Returns:
        str: One of LOW / MEDIUM / HIGH / CONGESTED
    """
    for level, (low, high) in DENSITY_THRESHOLDS.items():
        if low <= vehicle_count <= high:
            return level
    return "CONGESTED"


def build_output(vehicle_count):
    """
    Build the JSON-ready output dict for a single frame.

    Args:
        vehicle_count (int): Number of vehicles detected

    Returns:
        dict: Output matching the project's data format spec
    """
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "vehicle_count": vehicle_count,
        "current_density": get_density_level(vehicle_count),
        "predicted_density": None,  # LSTM prediction added in a later phase
    }
