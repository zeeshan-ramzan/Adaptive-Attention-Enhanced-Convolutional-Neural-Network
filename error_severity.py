"""
error_severity.py  --  Error severity tiers and crop-level confusion matrix
==========================================================================
(paper Table "Error severity distribution" and Figure "Crop-level confusion matrix")

Every test-set misclassification is assigned to
  Tier 1: disease confused with another disease of the same crop,
  Tier 2: healthy <-> diseased confusion within the same crop,
  Tier 3: confusion between different crops.
The 38 classes are then aggregated into crop species to draw (a) a count confusion
matrix with a logarithmic colour scale and (b) a row-normalised percentage matrix.

Usage:  python error_severity.py --npz runs/aae_cnn/test_outputs.npz --out-dir analysis
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import common


def main():
    p = argparse.ArgumentParser(description="Error severity analysis")
    p.add_argument("--npz", required=True)
    p.add_argument("--out-dir", default="analysis")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    d = np.load(a.npz, allow_pickle=True)
    y, prob, names = d["y_true"], d["y_prob"], list(d["class_names"])
    pred = prob.argmax(1)
    crop = [common.crop_of(n) for n in names]
    tiers = {1: 0, 2: 0, 3: 0}
    for t, pr in zip(y, pred):
        if t == pr:
            continue
        if crop[t] != crop[pr]:
            tiers[3] += 1
        elif common.is_healthy(names[t]) != common.is_healthy(names[pr]):
            tiers[2] += 1
        else:
            tiers[1] += 1
    n_err = sum(tiers.values())
    desc = {1: "Disease-to-disease (same crop)", 2: "Healthy <-> diseased", 3: "Cross-crop confusion"}
    rows = [{"tier": k, "description": desc[k], "count": v, "percent_of_errors": round(100 * v / max(n_err, 1), 1)}
            for k, v in tiers.items()]
    rows.append({"tier": "total", "description": "", "count": n_err, "percent_of_errors": 100.0})
    common.write_csv(os.path.join(a.out_dir, "error_severity.csv"), rows)
    for r in rows:
        print(r)

    crops = sorted(set(crop))
    ci = {c: i for i, c in enumerate(crops)}
    cm = np.zeros((len(crops), len(crops)), dtype=int)
    for t, pr in zip(y, pred):
        cm[ci[crop[t]], ci[crop[pr]]] += 1
    print(f"Crop-level accuracy: {100 * np.trace(cm) / cm.sum():.2f}%")
    labels = [f"{c} ({cm[i].sum()})" for i, c in enumerate(crops)]
    pct = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, mat, title, kw in ((axes[0], cm, "(a) Number of test images", {"norm": LogNorm(vmin=1, vmax=max(cm.max(), 2))}),
                               (axes[1], pct, "(b) Row-normalised percentage", {"vmin": 0, "vmax": 100})):
        shown = np.where(mat > 0, mat, np.nan) if "norm" in kw else mat
        im = ax.imshow(shown, cmap="Blues", **kw)
        for i in range(len(crops)):
            for j in range(len(crops)):
                if mat[i, j] > 0:
                    txt = f"{mat[i, j]}" if mat is cm else (f"{mat[i, j]:.1f}" if mat[i, j] >= 0.1 else "<0.1")
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                            color="white" if (mat[i, j] > (0.5 * mat.max())) else "black")
        ax.set_xticks(range(len(crops)))
        ax.set_xticklabels(crops, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(crops)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted crop")
        ax.set_ylabel("True crop")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out_dir, "crop_confusion_matrix.png"), dpi=300)
    np.savetxt(os.path.join(a.out_dir, "crop_confusion_matrix.csv"), cm, fmt="%d", delimiter=",",
               header=",".join(crops), comments="")


if __name__ == "__main__":
    main()
