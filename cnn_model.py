"""
cnn_model.py

Kelas CNN mudah untuk pengelasan imej (image classification).
Struktur asas (boilerplate) fail ini dijana dengan bantuan AI Code Assistant.

--- PROMPT AI CODE ASSISTANT YANG DIGUNAKAN (WAJIB - min 3 baris) ---
# Prompt: "Generate a simple PyTorch CNN class for image classification
# with 2 convolutional layers and 1 fully connected layer, input size
# 3x32x32 (RGB CIFAR-10 style images), output 10 classes. Include
# ReLU activation and max pooling after each conv layer."
# AI Code Assistant digunakan: ChatGPT (kod telah disemak dan diubahsuai
# secara manual untuk memastikan ia berfungsi tanpa ralat sintaks).
----------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Model CNN mudah:
    - 2 Convolutional layers
    - 1 Fully Connected layer (output layer)
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional layer 1: input 3 channel (RGB), output 16 channel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

        # Convolutional layer 2: input 16 channel, output 32 channel
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Selepas 2x pooling (32x32 -> 16x16 -> 8x8), saiz feature map = 8x8
        self.fc1 = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> (16, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))   # -> (32, 8, 8)
        x = torch.flatten(x, 1)                # flatten untuk fully connected
        x = self.fc1(x)                        # output layer
        return x


if __name__ == "__main__":
    # Ujian ringkas untuk pastikan model berfungsi tanpa ralat
    model = SimpleCNN(num_classes=10)
    dummy_input = torch.randn(1, 3, 32, 32)  # batch=1, RGB 32x32
    output = model(dummy_input)
    print("Output shape:", output.shape)  # sepatutnya [1, 10]
    print(model)
