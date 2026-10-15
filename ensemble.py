"""
ensemble.py  --  Deep ensemble of independently trained models (Lakshminarayanan et al., 2017)
=============================================================================================
(paper Section "Ensemble performance")

Averages the softmax outputs of several models evaluated on the same test split
(e.g. AAE-CNN trained with three random seeds) and reports the ensemble metrics.
Each training script writes <run>/test_outputs.npz, which is what this script reads.

Train three seeds first, e.g.
  python aae_cnn.py --data-dir PV --out-dir runs --seed 42 --run-name aae_seed42 --split-file runs/split.json
  python aae_cnn.py --data-dir PV --out-dir runs --seed 43 --run-name aae_seed43 --split-file runs/split.json
  python aae_cnn.py --data-dir PV --out-dir runs --seed 44 --run-name aae_seed44 --split-file runs/split.json
(the same --split-file keeps the test images identical), then
  python ensemble.py --npz runs/aae_seed42/test_outputs.npz runs/aae_seed43/test_outputs.npz runs/aae_seed44/test_outputs.npz
"""
import argparse

import numpy as np

import common


def main():
    p = argparse.ArgumentParser(description="Softmax-averaging ensemble")
    p.add_argument("--npz", nargs="+", required=True)
    p.add_argument("--out", default="ensemble_metrics.json")
    args = p.parse_args()
    probs, y_ref, names = [], None, None
    for f in args.npz:
        d = np.load(f, allow_pickle=True)
        if y_ref is None:
            y_ref, names = d["y_true"], list(d["class_names"])
        elif not np.array_equal(y_ref, d["y_true"]):
            raise SystemExit(f"{f} was evaluated on a different test set; use the same --split-file.")
        probs.append(d["y_prob"])
        m = common.compute_metrics(y_ref, d["y_prob"], names)
        print(f"member {f}: accuracy {100 * m['accuracy']:.2f}%")
    ens = np.mean(probs, axis=0)
    m = common.compute_metrics(y_ref, ens, names)
    common.print_metrics(m, f"Ensemble of {len(probs)} models")
    common.save_json(args.out, m)


if __name__ == "__main__":
    main()
