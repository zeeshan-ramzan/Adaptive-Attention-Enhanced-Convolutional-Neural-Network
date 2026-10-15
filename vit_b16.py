"""
vit_b16.py  --  Baseline: Vision Transformer ViT-B/16 (Dosovitskiy et al., 2021)
===============================================================================
ImageNet-pretrained ViT-B/16 (16x16 patches, 12 layers, hidden size 768, 12 heads)
with a new num_classes classification head. Pretrained weights require 224x224 input;
other input sizes are only possible when training from scratch (--no-pretrained).

Usage:  python vit_b16.py --data-dir /path/PlantVillage --out-dir runs
"""
import argparse

import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

import common


def build_model(num_classes: int, pretrained: bool = True, img_size: int = 224, **_) -> nn.Module:
    if pretrained:
        if img_size != 224:
            raise ValueError("Pretrained ViT-B/16 needs --img-size 224 (or use --no-pretrained).")
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        model = vit_b_16(weights=None, image_size=img_size)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def main():
    args = common.add_common_args(argparse.ArgumentParser(description="Train ViT-B/16 baseline")).parse_args()
    common.run_training("vit_b16", args)


if __name__ == "__main__":
    main()
