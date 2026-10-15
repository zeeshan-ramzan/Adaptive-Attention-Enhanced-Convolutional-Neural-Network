"""
robustness_corruption.py  --  Accuracy under synthetic image corruption
======================================================================
(paper Table "Mean accuracy under synthetic image corruption")

Five corruption types, three severity levels each, applied to the test images:
  gaussian_noise, motion_blur, jpeg_compression, brightness_shift, contrast_reduction
The paper table reports the moderate level (severity 2) and the mean over corruption types.

Usage:
  python robustness_corruption.py --data-dir /path/PlantVillage \
      --checkpoints runs/densenet121/best.pt runs/cbam_resnet50/best.pt runs/aae_cnn/best.pt
"""
import argparse
import io
import os

import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import convolve
from torch.utils.data import DataLoader

import common

SEVERITY = {
    "gaussian_noise": [0.04, 0.08, 0.12],      # std of additive noise on [0, 1] pixels
    "motion_blur": [5, 9, 13],                  # length of horizontal blur kernel (pixels)
    "jpeg_compression": [30, 15, 8],            # JPEG quality
    "brightness_shift": [0.15, 0.30, 0.45],     # added to V channel (fraction of range)
    "contrast_reduction": [0.6, 0.4, 0.25],     # contrast factor
}


def corrupt(img: Image.Image, kind: str, severity: int, rng: np.random.Generator) -> Image.Image:
    v = SEVERITY[kind][severity - 1]
    if kind == "gaussian_noise":
        a = np.asarray(img).astype(np.float32) / 255.0
        a = np.clip(a + rng.normal(0, v, a.shape), 0, 1)
        return Image.fromarray((a * 255).astype(np.uint8))
    if kind == "motion_blur":
        k = np.zeros((v, v), np.float32)
        k[v // 2, :] = 1.0 / v
        a = np.asarray(img).astype(np.float32)
        out = np.stack([convolve(a[..., c], k, mode="reflect") for c in range(3)], -1)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if kind == "jpeg_compression":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(v))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if kind == "brightness_shift":
        hsv = np.asarray(img.convert("HSV")).astype(np.float32)
        hsv[..., 2] = np.clip(hsv[..., 2] + v * 255, 0, 255)
        return Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
    if kind == "contrast_reduction":
        return ImageEnhance.Contrast(img).enhance(v)
    raise ValueError(kind)


class Corruptor:
    def __init__(self, kind, severity, seed=0):
        self.kind, self.severity, self.rng = kind, severity, np.random.default_rng(seed)

    def __call__(self, img):
        return corrupt(img, self.kind, self.severity, self.rng)


def main():
    p = argparse.ArgumentParser(description="Robustness to synthetic corruption")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--split-file", default=None, help="split JSON written during training (default: next to runs)")
    p.add_argument("--severities", type=int, nargs="*", default=[1, 2, 3])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="robustness_results.csv")
    args = p.parse_args()
    device = common.get_device()
    rows = []
    for ck in args.checkpoints:
        model, meta = common.load_checkpoint(ck, device)
        split_file = args.split_file or os.path.join(os.path.dirname(os.path.dirname(ck)), f"split_seed{args.seed}.json")
        split = common.make_split(args.data_dir, seed=args.seed, split_file=split_file)
        tf = common.build_transforms(meta.get("img_size", 224), train=False)
        clean_dl = DataLoader(common.SubsetWithTransform(split.subset("test"), tf), batch_size=args.batch_size,
                              num_workers=args.workers)
        y, prob = common.predict(model, clean_dl, device)
        clean = (prob.argmax(1) == y).mean()
        row = {"checkpoint": ck, "model": meta["model_name"], "clean": round(100 * clean, 2)}
        for s in args.severities:
            accs = []
            for kind in SEVERITY:
                ds = common.SubsetWithTransform(split.subset("test"), tf, pre_transform=Corruptor(kind, s, args.seed))
                y, prob = common.predict(model, DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers), device)
                acc = 100 * (prob.argmax(1) == y).mean()
                accs.append(acc)
                row[f"{kind}_s{s}"] = round(acc, 2)
                print(f"{meta['model_name']:<14} {kind:<20} severity {s}: {acc:.2f}%")
            row[f"mean_s{s}"] = round(float(np.mean(accs)), 2)
            row[f"drop_s{s}"] = round(100 * clean - float(np.mean(accs)), 2)
        rows.append(row)
        common.write_csv(args.out, rows)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
