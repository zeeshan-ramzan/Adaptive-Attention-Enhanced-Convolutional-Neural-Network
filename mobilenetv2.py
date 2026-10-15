"""
mobilenetv2.py  --  Baseline: MobileNetV2 (Sandler et al., 2018)
===============================================================
ImageNet-pretrained MobileNetV2 (inverted residuals, linear bottlenecks) with a new
num_classes classifier.

Usage:  python mobilenetv2.py --data-dir /path/PlantVillage --out-dir runs
"""
import argparse

import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

import common


def build_model(num_classes: int, pretrained: bool = True, img_size: int = 224, **_) -> nn.Module:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def main():
    args = common.add_common_args(argparse.ArgumentParser(description="Train MobileNetV2 baseline")).parse_args()
    common.run_training("mobilenetv2", args)


if __name__ == "__main__":
    main()
