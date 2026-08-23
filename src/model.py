"""
Model definitions for image classification.

Two architectures are supported, selected via the `model.architecture` field
in the training config:

  - "simple_cnn": a small CNN built from scratch. Trains fast and works well
    directly on 32x32 (CIFAR-10) or 28x28 (Fashion-MNIST) images.
  - "resnet18": torchvision's ResNet-18, optionally initialized with
    ImageNet-pretrained weights and fine-tuned by replacing the final
    fully-connected layer. The first conv layer is swapped out automatically
    for single-channel (grayscale) inputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """A small convolutional network for 32x32-ish images."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> H/2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> H/4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> H/8
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_resnet18(num_classes: int = 10, pretrained: bool = True, in_channels: int = 3) -> nn.Module:
    """Build a ResNet-18, optionally with ImageNet-pretrained weights, fine-tuned
    for `num_classes` outputs. Handles non-RGB input by replacing conv1."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(architecture: str, num_classes: int = 10, pretrained: bool = True, in_channels: int = 3) -> nn.Module:
    """Factory function used by both training and serving code."""
    architecture = architecture.lower()
    if architecture == "simple_cnn":
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    elif architecture == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown architecture: {architecture!r}. Choose 'simple_cnn' or 'resnet18'.")
