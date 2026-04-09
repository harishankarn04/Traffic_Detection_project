"""
preprocess.py — Load vehicle_counts.csv → sliding window sequences → PyTorch DataLoaders.

Sequence: 30 consecutive vehicle counts → predict density at t+1
"""

import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

CSV_PATH    = Path(__file__).parent.parent / "data" / "processed" / "vehicle_counts.csv"
WINDOW_SIZE = 30
LABEL_MAP   = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CONGESTED": 3}


class TrafficDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.float32)  # [N, 30, 1]
        self.y = torch.tensor(labels, dtype=torch.long)        # [N]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(csv_path=CSV_PATH, window=WINDOW_SIZE, val_split=0.2):
    counts, labels = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts.append(int(row["vehicle_count"]))
            labels.append(LABEL_MAP[row["density_level"]])

    counts = np.array(counts, dtype=np.float32)

    # Normalize counts to [0, 1] using fixed max (50 vehicles = CONGESTED upper bound)
    MAX_COUNT = 50.0
    counts_norm = np.clip(counts / MAX_COUNT, 0.0, 1.0)

    # Sliding window: X = counts[i:i+window], y = labels[i+window]
    X, y = [], []
    for i in range(len(counts_norm) - window):
        X.append(counts_norm[i:i + window].reshape(window, 1))
        y.append(labels[i + window])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Train/val split (no shuffle — preserve time order)
    split = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_loader = DataLoader(TrafficDataset(X_train, y_train), batch_size=256, shuffle=True)
    val_loader   = DataLoader(TrafficDataset(X_val,   y_val),   batch_size=256, shuffle=False)

    print(f"Train: {len(X_train)} sequences | Val: {len(X_val)} sequences")
    return train_loader, val_loader, MAX_COUNT
