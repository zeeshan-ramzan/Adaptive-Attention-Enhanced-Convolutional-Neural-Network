"""
se_resnet50.py  --  Baseline: SE-ResNet-50 (Hu et al., 2018)
===========================================================
ResNet-50 in which every bottleneck block recalibrates its residual branch with a
Squeeze-and-Excitation module (reduction ratio 16) before the identity addition:

    out = SE(BN3(conv3(...)));  y = ReLU(out + shortcut(x))

The ResNet-50 part is initialised from ImageNet weights; the SE modules are trained
from scratch.

Usage:  python se_resnet50.py --data-dir /path/PlantVillage --out-dir runs
"""
import argparse

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.resnet import Bottleneck

import common


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))
        return x * s


class SEBottleneck(nn.Module):
    """Wraps a torchvision Bottleneck and inserts SE before the residual addition."""

    def __init__(self, block: Bottleneck, reduction: int = 16):
        super().__init__()
        self.block = block
        self.se = SEModule(block.bn3.num_features, reduction)

    def forward(self, x):
        b = self.block
        identity = x
        out = b.relu(b.bn1(b.conv1(x)))
        out = b.relu(b.bn2(b.conv2(out)))
        out = b.bn3(b.conv3(out))
        out = self.se(out)
        if b.downsample is not None:
            identity = b.downsample(x)
        return b.relu(out + identity)


def build_model(num_classes: int, pretrained: bool = True, img_size: int = 224, reduction: int = 16, **_):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    for lname in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, lname)
        for i in range(len(layer)):
            layer[i] = SEBottleneck(layer[i], reduction)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    args = common.add_common_args(argparse.ArgumentParser(description="Train SE-ResNet-50 baseline")).parse_args()
    common.run_training("se_resnet50", args)


if __name__ == "__main__":
    main()
