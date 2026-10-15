"""
efficientnet_b4.py  --  Baseline: EfficientNet-B4 (Tan and Le, 2019)
===================================================================
ImageNet-pretrained EfficientNet-B4 (compound scaling of depth, width and resolution)
with a new num_classes classifier. For a fair comparison it is trained at the same
224x224 input size as all other models (change with --img-size, e.g. 380 = native size).

Usage:  python efficientnet_b4.py --data-dir /path/PlantVillage --out-dir runs
"""
import argparse

import torch.nn as nn
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

import common


def build_model(num_classes: int, pretrained: bool = True, img_size: int = 224, **_) -> nn.Module:
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def main():
    args = common.add_common_args(argparse.ArgumentParser(description="Train EfficientNet-B4 baseline")).parse_args()
    common.run_training("efficientnet_b4", args)


if __name__ == "__main__":
    main()
