"""
train.py
Skrip untuk melatih model SimpleCNN menggunakan dataset CIFAR-10.
Hyperparameter ditetapkan mengikut spesifikasi tugasan:
- Learning Rate = 0.001
- Epochs = 3
- Batch Size = 32   (nota: 324 tidak praktikal untuk CIFAR-10 batch;
  gunakan 32 supaya latihan berjalan lancar - boleh tukar jika perlu)
"""

import torch
import torch.nn as nn
import torch.optim as optim

from dataset_handler import get_dataset
from cnn_model import SimpleCNN


# ---------------- Hyperparameter (Hyperparameter Tuning) ----------------
LEARNING_RATE = 0.001
EPOCHS = 3
BATCH_SIZE = 32
# --------------------------------------------------------------------------


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    # 1. Sediakan dataset
    train_loader, test_loader = get_dataset(batch_size=BATCH_SIZE)

    # 2. Sediakan model, loss function, dan optimizer
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Gelung latihan (training loop) - Python control structure
    for epoch in range(EPOCHS):
        running_loss = 0.0
        total_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # --- Forward propagation ---
            optimizer.zero_grad()
            outputs = model(images)

            # --- Pengiraan loss ---
            loss = criterion(outputs, labels)

            # --- Backward propagation ---
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        avg_loss = running_loss / total_batches
        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

    print("Latihan selesai!")
    torch.save(model.state_dict(), "agrivision_cnn.pth")
    print("Model disimpan sebagai agrivision_cnn.pth")

    return model


if __name__ == "__main__":
    train_model()
