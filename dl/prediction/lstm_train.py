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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

    history_loss = []
    history_acc = []

    print(f"Train: {len(train_loader.dataset)} sequences | Val: {len(val_loader.dataset)} sequences")
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
        avg_loss = total_loss / len(train_loader)
        
        history_loss.append(avg_loss)
        history_acc.append(val_acc)
        
        print(f"Epoch {epoch:3d}/{EPOCHS} | loss: {avg_loss:.4f} | val_acc: {val_acc:.4f}")

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

    # --- Generate Training Curve Graph ---
    print("\nGenerating Training Curve...")
    epochs_range = range(1, len(history_loss) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Training Epochs', fontsize=12)
    ax1.set_ylabel('Training Loss', color=color, fontsize=12)
    ax1.plot(epochs_range, history_loss, color=color, marker='o', linewidth=2, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Accuracy (%)', color=color, fontsize=12)
    ax2.plot(epochs_range, [acc * 100 for acc in history_acc], color=color, marker='s', linewidth=2, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-5, 105)
    
    plt.title('LSTM Traffic Density Model Convergence', fontsize=14, pad=15)
    fig.tight_layout()
    
    curve_path = Path(__file__).parent / "lstm_training_curve.png"
    plt.savefig(curve_path, dpi=300)
    print(f"Training Curve saved to: {curve_path}")

    # --- Generate Confusion Matrix ---
    print("\nGenerating Confusion Matrix...")
    # Load best weights
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    labels = ["LOW", "MEDIUM", "HIGH", "CONGESTED"]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Density')
    plt.ylabel('Actual (True) Density')
    plt.title('LSTM Traffic Density Prediction Confusion Matrix')
    
    cm_path = Path(__file__).parent / "lstm_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    print(f"Confusion Matrix saved to: {cm_path}")


if __name__ == "__main__":
    train()
