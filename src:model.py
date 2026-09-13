"""
model.py — Transfer learning model definition
"""
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """
    Builds a ResNet-18 with a frozen ImageNet-pretrained backbone and a
    fresh final layer sized for our specific number of classes.
    """
    model = models.resnet18(weights='IMAGENET1K_V1')

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)
