"""
roc_analysis.py  --  Macro-averaged one-vs-rest ROC curves and AUC
=================================================================
(paper Table "Macro-averaged AUC-ROC comparison" and Figure "ROC curves")

Usage:  python roc_analysis.py --npz runs/aae_cnn/test_outputs.npz runs/cbam_resnet50/test_outputs.npz \
            --out roc_curves.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

import common


def macro_roc(y, prob):
    grid = np.linspace(0, 1, 1001)
    tprs = []
    for c in range(prob.shape[1]):
        pos = (y == c).astype(int)
        if pos.sum() in (0, len(pos)):
            continue
        fpr, tpr, _ = roc_curve(pos, prob[:, c])
        tprs.append(np.interp(grid, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    return grid, mean_tpr


def main():
    p = argparse.ArgumentParser(description="ROC analysis")
    p.add_argument("--npz", nargs="+", required=True)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--out", default="roc_curves.png")
    a = p.parse_args()
    labels = a.labels or [os.path.basename(os.path.dirname(f)) for f in a.npz]
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(11, 4.5))
    rows = []
    for lab, f in zip(labels, a.npz):
        d = np.load(f, allow_pickle=True)
        y, prob = d["y_true"], d["y_prob"]
        m_auc = common.macro_auc(y, prob)
        fpr, tpr = macro_roc(y, prob)
        rows.append({"model": lab, "macro_auc_ovr": round(m_auc, 4), "auc_of_macro_curve": round(auc(fpr, tpr), 4)})
        for axx in (ax, axz):
            axx.plot(fpr, tpr, label=f"{lab} (AUC = {m_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    axz.set_xlim(0, 0.05)
    axz.set_ylim(0.9, 1.001)
    axz.set_title("Zoom on the top-left corner")
    for axx in (ax, axz):
        axx.set_xlabel("False positive rate")
        axx.set_ylabel("True positive rate")
        axx.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(a.out, dpi=300)
    common.write_csv(os.path.splitext(a.out)[0] + ".csv", rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
