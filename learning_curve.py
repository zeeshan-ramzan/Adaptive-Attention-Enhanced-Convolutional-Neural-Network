"""
learning_curve.py  --  Validation accuracy versus training-data fraction
======================================================================
(paper Figure "Validation accuracy learning curve")

Retrains AAE-CNN and the DenseNet-121 backbone on stratified 10/25/50/75/100% subsets
of the training split (validation and test splits unchanged) and plots the curves.

Usage:  python learning_curve.py --data-dir /path/PlantVillage --out-dir runs_lc
"""
import argparse
import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aae_cnn
import common


def main():
    p = common.add_common_args(argparse.ArgumentParser(description="Learning curve"))
    p = aae_cnn.add_aae_args(p)
    p.add_argument("--fractions", type=float, nargs="*", default=[0.10, 0.25, 0.50, 0.75, 1.00])
    p.add_argument("--models", nargs="*", default=["aae_cnn", "densenet121"])
    args = p.parse_args()
    split = common.make_split(args.data_dir, seed=args.seed, split_file=common.default_split_file(args))
    labels = np_labels(split)
    rows = []
    for frac in args.fractions:
        idx = common.subsample_fraction(split.train_idx, labels, frac, args.seed)
        for m in args.models:
            a = copy.deepcopy(args)
            run = f"lc_{m}_{int(round(frac * 100))}"
            print(f"\n######## {m} with {frac:.0%} of the training data ({len(idx)} images) ########")
            r = (aae_cnn.train_aae(a, train_idx=idx, split=split, run_name=run) if m == "aae_cnn"
                 else common.run_training(m, a, train_idx=idx, split=split, run_name=run))
            rows.append({"model": m, "fraction": frac, "n_train": len(idx),
                         "val_accuracy": round(100 * r["best_val_acc"], 2), "test_accuracy": round(100 * r["accuracy"], 2)})
            common.write_csv(os.path.join(args.out_dir, "learning_curve.csv"), rows)
    plt.figure(figsize=(6, 4))
    for m in args.models:
        pts = [(r["fraction"] * 100, r["val_accuracy"]) for r in rows if r["model"] == m]
        plt.plot(*zip(*pts), marker="o", label=m)
    plt.xlabel("Training data used (%)")
    plt.ylabel("Validation accuracy (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "learning_curve.png"), dpi=300)


def np_labels(split):
    import numpy as np
    return np.array([s[1] for s in split.samples])


if __name__ == "__main__":
    main()
