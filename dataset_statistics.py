"""
dataset_statistics.py  --  PlantVillage class distribution and imbalance
=======================================================================
(paper Table "PlantVillage dataset: all 38 categories" and Figure "Class distribution histogram")

Usage:  python dataset_statistics.py --data-dir /path/PlantVillage --out-dir analysis
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common


def main():
    p = argparse.ArgumentParser(description="Dataset statistics")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", default="analysis")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    names, samples = common.scan_dataset(a.data_dir)
    counts = common.class_counts(samples, len(names)).astype(int)
    rows = [{"id": i + 1, "class": n, "crop": common.crop_of(n), "images": int(c)} for i, (n, c) in enumerate(zip(names, counts))]
    common.write_csv(os.path.join(a.out_dir, "class_distribution.csv"), rows)
    ratio = counts.max() / max(counts.min(), 1)
    print(f"{len(names)} classes, {counts.sum()} images, {len(set(r['crop'] for r in rows))} crops")
    print(f"smallest: {names[counts.argmin()]} ({counts.min()}), largest: {names[counts.argmax()]} ({counts.max()}), "
          f"imbalance ratio {ratio:.1f}:1")
    top3 = set(np.argsort(counts)[-3:])
    colors = ["#C44E52" if i in top3 else "#DD8452" if i == counts.argmin() else "#4C72B0" for i in range(len(names))]
    plt.figure(figsize=(12, 4))
    bars = plt.bar(np.arange(1, len(names) + 1), counts, color=colors)
    for b, c in zip(bars, counts):
        plt.text(b.get_x() + b.get_width() / 2, c, f"{c:,}", ha="center", va="bottom", rotation=90, fontsize=7)
    plt.xticks(np.arange(1, len(names) + 1), fontsize=7)
    plt.xlabel("Class index")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(os.path.join(a.out_dir, "class_distribution.png"), dpi=300)


if __name__ == "__main__":
    main()
