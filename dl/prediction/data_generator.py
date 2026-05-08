"""
data_generator.py — Generate simulated traffic time-series data for LSTM training.

Simulates realistic 24h traffic patterns:
- Low traffic at night
- Morning rush hour peak (7-9am)
- Evening rush hour peak (4-7pm)
- Weekday vs weekend variation + random noise

Run from dl/:
    python prediction/data_generator.py

Output: data/processed/vehicle_counts.csv
"""

import csv
import random
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "vehicle_counts.csv"
SAMPLES_PER_DAY = 1440   # 1 sample per minute
DAYS = 35                # ~50,000 samples total
DENSITY_THRESHOLDS = {"LOW": (0, 10), "MEDIUM": (11, 25), "HIGH": (26, 40), "CONGESTED": (41, 999)}


def density_label(count):
    for label, (lo, hi) in DENSITY_THRESHOLDS.items():
        if lo <= count <= hi:
            return label
    return "CONGESTED"


def base_traffic(hour, minute, is_weekend):
    """Return expected vehicle count for given time."""
    t = hour + minute / 60.0

    if is_weekend:
        # Weekends: gentler, later peaks
        if 0 <= t < 6:
            base = 2
        elif 6 <= t < 10:
            base = 2 + 10 * math.sin(math.pi * (t - 6) / 4)
        elif 10 <= t < 14:
            base = 12
        elif 14 <= t < 20:
            base = 12 + 8 * math.sin(math.pi * (t - 14) / 6)
        else:
            base = 4 - 2 * (t - 20) / 4
    else:
        # Weekdays: sharp morning + evening rush
        if 0 <= t < 5:
            base = 2
        elif 5 <= t < 7:
            base = 2 + 5 * (t - 5) / 2
        elif 7 <= t < 9:
            base = 7 + 35 * math.sin(math.pi * (t - 7) / 2)  # morning peak ~42
        elif 9 <= t < 12:
            base = 18 - 6 * (t - 9) / 3
        elif 12 <= t < 14:
            base = 12 + 8 * math.sin(math.pi * (t - 12) / 2)  # lunch bump
        elif 14 <= t < 16:
            base = 14
        elif 16 <= t < 19:
            base = 14 + 30 * math.sin(math.pi * (t - 16) / 3)  # evening peak ~44
        elif 19 <= t < 22:
            base = 20 - 14 * (t - 19) / 3
        else:
            base = 6 - 4 * (t - 22) / 2

    return max(0, base)


def generate():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    start = datetime(2024, 1, 1, 0, 0, 0)
    rows = []

    for day in range(DAYS):
        is_weekend = (day % 7) in (5, 6)
        for minute in range(SAMPLES_PER_DAY):
            ts = start + timedelta(days=day, minutes=minute)
            base = base_traffic(ts.hour, ts.minute, is_weekend)
            noise = random.gauss(0, 2.5)
            count = max(0, int(round(base + noise)))
            rows.append((ts.strftime("%Y-%m-%d %H:%M:%S"), count, density_label(count)))

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "vehicle_count", "density_level"])
        writer.writerows(rows)

    total = len(rows)
    print(f"Generated {total} samples → {OUTPUT_PATH}")

    # Class distribution
    from collections import Counter
    dist = Counter(r[2] for r in rows)
    for label in ["LOW", "MEDIUM", "HIGH", "CONGESTED"]:
        print(f"  {label:10s}: {dist[label]:6d} ({dist[label]/total*100:.1f}%)")


if __name__ == "__main__":
    random.seed(42)
    generate()
