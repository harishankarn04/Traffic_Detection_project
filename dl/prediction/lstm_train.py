"""
lstm_train.py — Define and train LSTM model for traffic density prediction.

Architecture: Input(30,1) → LSTM(128) → Dropout(0.3) → LSTM(64) → FC(32) → Softmax(4)

Run from dl/:
    python prediction/lstm_train.py

Output: prediction/lstm_best.pt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from pathlib import Path
from preprocess import load_data

DEVICE    = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = Path(__file__).parent / "lstm_best.pt"
EPOCHS    = 50
LR        = 0.001
PATIENCE  = 5


class TrafficLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1   = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
        self.drop    = nn.Dropout(0.3)
        self.lstm2   = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.fc1     = nn.Linear(64, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, 4)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.drop(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]    # take last timestep
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def train():
    print(f"Device: {DEVICE}")
    train_loader, val_loader, _ = load_data()

    model     = TrafficLSTM().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    no_improve   = 0

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- Validate ---
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch:3d}/{EPOCHS} | loss: {total_loss/len(train_loader):.4f} | val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best val_acc: {best_val_acc:.4f}")
                break

    print(f"\nBest val accuracy: {best_val_acc:.4f} → saved to {SAVE_PATH}")


if __name__ == "__main__":
    train()
