"""
per_class_analysis.py  --  Per-class and per-crop performance
============================================================
(paper Tables "Per-class precision, recall and F1-score", "Macro F1-score by crop species",
 Table "Classification performance" and Figures "accuracy comparison", "per-class F1")

Reads the test_outputs.npz files written by the training scripts and produces
  * overall_results.csv     : accuracy, macro precision / recall / F1, macro AUC per model
  * per_class_<model>.csv   : precision, recall, F1, support for every class
  * per_crop_f1.csv         : macro F1 aggregated by crop species for every model
  * accuracy_comparison.png : bar chart of test accuracy
  * per_class_f1.png        : grouped bars of F1 for selected classes

Usage:
  python per_class_analysis.py --npz runs/*/test_outputs.npz --out-dir analysis \
      --classes "Apple___Cedar_apple_rust" "Potato___healthy" "Tomato___Early_blight"
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

import common


def main():
    p = argparse.ArgumentParser(description="Per-class / per-crop analysis")
    p.add_argument("--npz", nargs="+", required=True)
    p.add_argument("--labels", nargs="*", default=None, help="display names (default: run folder names)")
    p.add_argument("--classes", nargs="*", default=None, help="classes for the per-class F1 figure (default 9)")
    p.add_argument("--out-dir", default="analysis")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    labels = a.labels or [os.path.basename(os.path.dirname(f)) for f in a.npz]
    overall, crop_rows, per_class_f1, names = [], {}, {}, None
    for lab, f in zip(labels, a.npz):
        d = np.load(f, allow_pickle=True)
        y, prob, names = d["y_true"], d["y_prob"], list(d["class_names"])
        m = common.compute_metrics(y, prob, names)
        overall.append({"model": lab, "accuracy": round(100 * m["accuracy"], 2),
                        "macro_precision": round(100 * m["macro_precision"], 2),
                        "macro_recall": round(100 * m["macro_recall"], 2),
                        "macro_f1": round(100 * m["macro_f1"], 2), "macro_auc": round(m["macro_auc_ovr"], 4)})
        common.write_csv(os.path.join(a.out_dir, f"per_class_{lab}.csv"),
                         [{"class": k, **{kk: round(100 * vv, 2) if kk != "support" else vv for kk, vv in v.items()}}
                          for k, v in m["per_class"].items()])
        per_class_f1[lab] = {k: 100 * v["f1"] for k, v in m["per_class"].items()}
        pred = prob.argmax(1)
        crops = np.array([common.crop_of(n) for n in names])
        for c in sorted(set(crops)):
            cls = np.where(crops == c)[0]
            mask = np.isin(y, cls)
            f1 = f1_score(y[mask], pred[mask], labels=cls, average="macro", zero_division=0)
            crop_rows.setdefault(c, {"crop": c, "classes": len(cls)})[lab] = round(100 * f1, 1)
    common.write_csv(os.path.join(a.out_dir, "overall_results.csv"), overall)
    common.write_csv(os.path.join(a.out_dir, "per_crop_f1.csv"), list(crop_rows.values()))
    for r in overall:
        print(r)

    plt.figure(figsize=(7, 4))
    accs = [r["accuracy"] for r in overall]
    bars = plt.bar(labels, accs, color=["#C44E52" if "aae" in l.lower() else "#4C72B0" for l in labels])
    for b, v in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.ylim(max(0, min(accs) - 1.5), 100)
    plt.ylabel("Test accuracy (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(a.out_dir, "accuracy_comparison.png"), dpi=300)

    sel = a.classes or names[:9]
    x = np.arange(len(sel))
    w = 0.8 / len(labels)
    plt.figure(figsize=(10, 4))
    for i, lab in enumerate(labels):
        plt.bar(x + i * w, [per_class_f1[lab][c] for c in sel], w, label=lab)
    plt.xticks(x + 0.4 - w / 2, [c.replace("___", "\n").replace("_", " ") for c in sel], fontsize=7)
    plt.ylabel("F1-score (%)")
    lo = min(per_class_f1[l][c] for l in labels for c in sel)
    plt.ylim(max(0, lo - 2), 100.5)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out_dir, "per_class_f1.png"), dpi=300)


if __name__ == "__main__":
    main()
