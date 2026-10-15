"""
augmentation_visualization.py  --  Illustration of the training augmentation pipeline
====================================================================================
(paper Figure "Visualization of the sequential data-augmentation steps")

Shows one leaf image after each augmentation step: original -> horizontal flip ->
vertical flip -> rotation (+-30 deg) -> colour jitter (0.2) -> random erasing.

Usage:  python augmentation_visualization.py --image leaf.jpg --out augmentation.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF


def main():
    p = argparse.ArgumentParser(description="Augmentation illustration")
    p.add_argument("--image", required=True)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="augmentation.png")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    img = Image.open(a.image).convert("RGB").resize((a.img_size, a.img_size))
    steps = [("Original", img)]
    img = TF.hflip(img); steps.append(("H-Flip", img))
    img = TF.vflip(img); steps.append(("V-Flip", img))
    img = transforms.RandomRotation(30)(img); steps.append(("Rotate", img))
    img = transforms.ColorJitter(0.2, 0.2, 0.2)(img); steps.append(("Color Jitter", img))
    t = transforms.RandomErasing(p=1.0, value=0)(TF.to_tensor(img)); steps.append(("Rand. Erase", TF.to_pil_image(t)))
    fig, axes = plt.subplots(1, len(steps), figsize=(2.4 * len(steps), 2.8))
    for ax, (name, im) in zip(axes, steps):
        ax.imshow(im)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(a.out, dpi=300)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
