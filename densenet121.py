"""
densenet121.py  --  Baseline: DenseNet-121 (Huang et al., 2017)
==============================================================
The AAE-CNN backbone without ADPA modules and without the auxiliary head:
ImageNet-pretrained DenseNet-121 with a new num_classes classifier.

Usage:  python densenet121.py --data-dir /path/PlantVillage --out-dir runs
"""
import argparse

import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121

import common


def build_model(num_classes: int, pretrained: bool = True, img_size: int = 224, **_) -> nn.Module:
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def main():
    args = common.add_common_args(argparse.ArgumentParser(description="Train DenseNet-121 baseline")).parse_args()
    common.run_training("densenet121", args)


if __name__ == "__main__":
    main()
