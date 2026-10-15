"""
cross_validation_significance.py  --  5-fold CV and paired significance tests
============================================================================
(paper Tables "Five-fold stratified cross-validation" and "Statistical significance")

For each of the K stratified folds, the fold is used as the test set and the remaining
images are split 8:1 into training and validation sets. Every model is trained on the
same folds. The script reports mean +- standard deviation of accuracy and macro F1 and,
for AAE-CNN against every baseline,
  * the mean per-fold accuracy difference,
  * a paired two-sided t-test (scipy.stats.ttest_rel),
  * a one-sided Wilcoxon signed-rank test (H1: AAE-CNN > baseline).

Usage:
  python cross_validation_significance.py --data-dir /path/PlantVillage --out-dir runs_cv
  python cross_validation_significance.py ... --models aae_cnn densenet121 cbam_resnet50
"""
import argparse
import copy
import os

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split

import aae_cnn
import common

DEFAULT_MODELS = ["aae_cnn", "resnet50", "densenet121", "efficientnet_b4", "se_resnet50", "cbam_resnet50", "vit_b16"]


def fold_splits(data_dir, k, seed):
    class_names, samples = common.scan_dataset(data_dir)
    y = np.array([s[1] for s in samples])
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (trval, test) in enumerate(skf.split(np.zeros(len(y)), y)):
        tr, va = train_test_split(trval, test_size=1 / 9, stratify=y[trval], random_state=seed + fold)
        yield fold, common.DataSplit(class_names, samples, sorted(tr.tolist()), sorted(va.tolist()),
                                     sorted(test.tolist()))


def main():
    p = common.add_common_args(argparse.ArgumentParser(description="K-fold CV + significance tests"))
    p = aae_cnn.add_aae_args(p)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = p.parse_args()
    res = {m: {"acc": [], "f1": []} for m in args.models}
    rows = []
    for fold, split in fold_splits(args.data_dir, args.folds, args.seed):
        for m in args.models:
            a = copy.deepcopy(args)
            a.out_dir = os.path.join(args.out_dir, f"fold{fold}")
            print(f"\n######## fold {fold + 1}/{args.folds} : {m} ########")
            if m == "aae_cnn":
                r = aae_cnn.train_aae(a, split=split, run_name=m)
            else:
                r = common.run_training(m, a, split=split, run_name=m)
            res[m]["acc"].append(r["accuracy"])
            res[m]["f1"].append(r["macro_f1"])
            rows.append({"fold": fold, "model": m, "accuracy": r["accuracy"], "macro_f1": r["macro_f1"]})
            common.write_csv(os.path.join(args.out_dir, "cv_per_fold.csv"), rows)

    summary = []
    print("\nCross-validation summary (mean +- SD)")
    for m in args.models:
        acc, f1 = 100 * np.array(res[m]["acc"]), 100 * np.array(res[m]["f1"])
        sd = lambda v: v.std(ddof=1) if len(v) > 1 else 0.0
        summary.append({"model": m, "acc_mean": acc.mean(), "acc_sd": sd(acc), "f1_mean": f1.mean(), "f1_sd": sd(f1)})
        print(f"{m:<16} acc {acc.mean():.2f} +- {sd(acc):.2f}   macro-F1 {f1.mean():.2f} +- {sd(f1):.2f}")
    common.write_csv(os.path.join(args.out_dir, "cv_summary.csv"), summary)

    if "aae_cnn" in args.models:
        sig = []
        ours = 100 * np.array(res["aae_cnn"]["acc"])
        print("\nSignificance of AAE-CNN improvement (per-fold accuracy)")
        for m in args.models:
            if m == "aae_cnn":
                continue
            base = 100 * np.array(res[m]["acc"])
            diff = ours - base
            if len(diff) > 1 and np.any(diff != diff[0]):
                t, p_t = stats.ttest_rel(ours, base)
            else:
                t, p_t = float("nan"), float("nan")
            try:
                p_w = stats.wilcoxon(ours, base, alternative="greater").pvalue
            except ValueError:
                p_w = float("nan")
            sig.append({"baseline": m, "mean_diff": diff.mean(), "t_statistic": t, "p_ttest_two_sided": p_t,
                        "p_wilcoxon_one_sided": p_w})
            print(f"{m:<16} diff {diff.mean():.2f}  t {t:.2f}  p(t) {p_t:.4f}  p(Wilcoxon) {p_w:.4f}")
        common.write_csv(os.path.join(args.out_dir, "significance_tests.csv"), sig)


if __name__ == "__main__":
    main()
