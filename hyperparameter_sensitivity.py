"""
hyperparameter_sensitivity.py  --  One-factor-at-a-time sensitivity (paper Table "Hyperparameter sensitivity")
============================================================================================================
Varies one hyperparameter at a time around the default configuration
(r = 16, eta0 = 3e-4, curriculum rate = 2.0) and reports validation accuracy:

  SE reduction ratio r      : 8, 16, 32
  initial learning rate eta0: 1e-4, 3e-4, 1e-3
  curriculum rate           : 1.0, 2.0, 3.0      (lambda(t) = min(1, rate * t / T))

Usage:  python hyperparameter_sensitivity.py --data-dir /path/PlantVillage --out-dir runs
"""
import argparse
import copy
import os

import aae_cnn
import common

GRID = [("reduction", [8, 16, 32]), ("lr", [1e-4, 3e-4, 1e-3]), ("curriculum_rate", [1.0, 2.0, 3.0])]


def main():
    p = common.add_common_args(argparse.ArgumentParser(description="AAE-CNN hyperparameter sensitivity"))
    args = aae_cnn.add_aae_args(p).parse_args()
    split = common.make_split(args.data_dir, seed=args.seed, split_file=common.default_split_file(args))
    rows, done = [], {}
    for name, values in GRID:
        for v in values:
            a = copy.deepcopy(args)
            setattr(a, name, v)
            key = (a.reduction, a.lr, a.curriculum_rate)
            if key in done:                       # default configuration is trained only once
                val = done[key]
            else:
                print(f"\n######## {name} = {v} ########")
                res = aae_cnn.train_aae(a, split=split, run_name=f"hp_{name}_{v}")
                val = done[key] = res["best_val_acc"]
            rows.append({"hyperparameter": name, "value": v, "val_accuracy": round(100 * val, 2)})
            common.write_csv(os.path.join(args.out_dir, "hyperparameter_sensitivity.csv"), rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
