"""
cbam_resnet50.py  --  Baseline: CBAM-ResNet-50 (Woo et al., 2018)
================================================================
ResNet-50 with a Convolutional Block Attention Module in every bottleneck block.
CBAM applies channel attention and then spatial attention *sequentially* to the
residual branch before the identity addition:

    Mc = sigmoid(MLP(AvgPool(F)) + MLP(MaxPool(F)))        (shared MLP, r = 16)
    F1 = Mc * F
    Ms = sigmoid(conv7x7([AvgPool_c(F1); MaxPool_c(F1)]))
    F2 = Ms * F1

ResNet-50 weights come from ImageNet; the CBAM modules are trained from scratch.

Usage:  python cbam_resnet50.py --data-dir /path/PlantVillage --out-dir runs
"""
import argparse

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.resnet import Bottleneck

import common


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(channels, channels // reduction, 1, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(channels // reduction, channels, 1, bias=False))

    def forward(self, x):
        avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.mlp(x.amax(dim=(2, 3), keepdim=True))
        return x * torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        m = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(m))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class CBAMBottleneck(nn.Module):
    def __init__(self, block: Bottleneck, reduction: int = 16):
        super().__init__()
        self.block = block
        self.cbam = CBAM(block.bn3.num_features, reduction)

    def forward(self, x):
        b = self.block
        identity = x
        out = b.relu(b.bn1(b.conv1(x)))
        out = b.relu(b.bn2(b.conv2(out)))
        out = self.cbam(b.bn3(b.conv3(out)))
        if b.downsample is not None:
            identity = b.downsample(x)
        return b.relu(out + identity)


def build_model(num_classes: int, pretrained: bool = True, img_size: int = 224, reduction: int = 16, **_):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    for lname in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, lname)
        for i in range(len(layer)):
            layer[i] = CBAMBottleneck(layer[i], reduction)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    args = common.add_common_args(argparse.ArgumentParser(description="Train CBAM-ResNet-50 baseline")).parse_args()
    common.run_training("cbam_resnet50", args)


if __name__ == "__main__":
    main()
