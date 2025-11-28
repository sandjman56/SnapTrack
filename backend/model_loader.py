"""
Model loading utilities for the OCR CNN.

This module encapsulates model definition, weight loading, and prediction
helpers so FastAPI routes stay lean.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torchvision import transforms

# Character set used by the model. Adjust as needed to match training data.
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class CNN(nn.Module):
    """Simple CNN matching the training script shape expectations."""

    def __init__(self, num_classes: int = len(CHARSET)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, num_classes)

        self.act = nn.functional.relu

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


def load_model() -> Tuple[nn.Module, str]:
    """Load CNN model weights from disk.

    Returns:
        Tuple of (model, charset) ready for inference.
    """
    weights_path = Path(__file__).resolve().parent.parent / "machine_learning_model_victor" / "model_weights.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=len(CHARSET)).to(device)

    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    return model, CHARSET


# Shared preprocessing pipeline for character crops
TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def predict_character(model: nn.Module, charset: str, image) -> str:
    """Predict a single character from an image crop."""
    device = next(model.parameters()).device
    with torch.no_grad():
        tensor = TRANSFORM(image).unsqueeze(0).to(device)
        logits = model(tensor)
        idx = torch.argmax(logits, dim=1).item()
    return charset[idx]
