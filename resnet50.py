"""
resnet50.py  --  Baseline: ResNet-50 (He et al., 2016)
=====================================================
ImageNet-pretrained ResNet-50 with the final fully connected layer replaced by a
num_classes output layer. Trained with the same protocol as AAE-CNN
(Adam 3e-4, weight decay 1e-4, batch 32, 100 epochs, cosine annealing, same split).

Usage:  python resnet50.py --data-dir /path/PlantVillage --out-dir runs
"""
import argparse

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

import common


def build_model(num_classes: int, pretrained: bool = True, img_size: int = 224, **_) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    args = common.add_common_args(argparse.ArgumentParser(description="Train ResNet-50 baseline")).parse_args()
    common.run_training("resnet50", args)


if __name__ == "__main__":
    main()
