"""
make_dummy_dataset.py
=====================
Creates a tiny synthetic dataset with the PlantVillage folder layout
(<Crop>___<Condition>/image.jpg). It is ONLY meant to check, in a few minutes on a
CPU, that every script in this package runs end to end before you start the real
experiments on PlantVillage.

Usage:  python make_dummy_dataset.py --out dummy_plantvillage --per-class 40
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["Apple___Apple_scab", "Apple___healthy", "Potato___Early_blight", "Potato___healthy",
           "Tomato___Late_blight", "Tomato___healthy"]


def make_image(cls_idx: int, rng: np.random.Generator, size: int = 96) -> Image.Image:
    base = [(60, 140, 60), (70, 160, 70), (90, 150, 50), (80, 170, 60), (50, 120, 70), (75, 155, 80)][cls_idx]
    img = Image.new("RGB", (size, size), (235, 235, 235))
    d = ImageDraw.Draw(img)
    jitter = rng.integers(-15, 15, 3)
    leaf = tuple(int(np.clip(c + j, 0, 255)) for c, j in zip(base, jitter))
    d.ellipse([10, 20, size - 10, size - 20], fill=leaf)
    if "healthy" not in CLASSES[cls_idx]:          # add "lesions"
        n = 4 + cls_idx * 2
        for _ in range(n):
            x, y = rng.integers(20, size - 20, 2)
            r = int(rng.integers(2, 6 + cls_idx))
            d.ellipse([x - r, y - r, x + r, y + r], fill=(110 + 20 * cls_idx % 100, 70, 30))
    arr = np.asarray(img).astype(np.int16) + rng.integers(-8, 8, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="dummy_plantvillage")
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    for ci, c in enumerate(CLASSES):
        os.makedirs(os.path.join(a.out, c), exist_ok=True)
        n = a.per_class if "healthy" not in c else max(a.per_class // 2, 10)   # mild imbalance
        for i in range(n):
            make_image(ci, rng).save(os.path.join(a.out, c, f"img_{i:04d}.jpg"), quality=95)
    print(f"Dummy dataset written to {a.out} ({len(CLASSES)} classes)")


if __name__ == "__main__":
    main()
