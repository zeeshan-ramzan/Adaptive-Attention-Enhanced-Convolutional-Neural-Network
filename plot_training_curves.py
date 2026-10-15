"""
plot_training_curves.py  --  Training and validation accuracy curves
===================================================================
(paper Figure "Training and validation accuracy curves")

Usage:  python plot_training_curves.py --history runs/aae_cnn/history.csv --out training_curves.png
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description="Plot training curves")
    p.add_argument("--history", required=True)
    p.add_argument("--out", default="training_curves.png")
    a = p.parse_args()
    with open(a.history) as f:
        rows = list(csv.DictReader(f))
    ep = [int(r["epoch"]) for r in rows]
    tr = [100 * float(r["train_acc"]) for r in rows]
    va = [100 * float(r["val_acc"]) for r in rows]
    best = max(range(len(va)), key=lambda i: va[i])
    plt.figure(figsize=(6.5, 4))
    plt.plot(ep, tr, label="Training accuracy")
    plt.plot(ep, va, label="Validation accuracy")
    plt.scatter([ep[best]], [va[best]], color="red", zorder=3, label=f"Best val. {va[best]:.2f}% (epoch {ep[best]})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(a.out, dpi=300)
    print(f"Saved {a.out}; best validation accuracy {va[best]:.2f}% at epoch {ep[best]}")


if __name__ == "__main__":
    main()
