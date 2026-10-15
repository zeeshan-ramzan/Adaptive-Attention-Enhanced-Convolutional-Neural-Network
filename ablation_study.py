"""
ablation_study.py  --  Component ablation of AAE-CNN (paper Table "Ablation study")
==================================================================================
Trains the following configurations on the same split and reports test accuracy and
macro F1:

  1. Backbone only (DenseNet-121)            adpa=none,     no auxiliary loss
  2. + Spatial attention only                 adpa=sa,       no auxiliary loss
  3. + SE channel attention only              adpa=se,       no auxiliary loss
  4. + Fixed dual attention (a~ = b~ = 0.5)   adpa=fixed,    no auxiliary loss
  5. + Adaptive gate (ADPA)                   adpa=adaptive, no auxiliary loss
  6. + ADPA + weighted loss                   adpa=adaptive, aux WCE with lambda = 1
  7. AAE-CNN (full, + curriculum)             adpa=adaptive, aux WCE with lambda(t) = min(1, 2t/T)

Usage:  python ablation_study.py --data-dir /path/PlantVillage --out-dir runs
        (add --only 1 5 7 to run a subset)
"""
import argparse
import copy
import os

import aae_cnn
import common

CONFIGS = [
    (1, "Backbone only (DenseNet-121)", dict(adpa_mode="none", no_aux=True, loss_schedule="none")),
    (2, "+ Spatial attention only", dict(adpa_mode="sa", no_aux=True, loss_schedule="none")),
    (3, "+ SE channel attention only", dict(adpa_mode="se", no_aux=True, loss_schedule="none")),
    (4, "+ Fixed dual attention (0.5/0.5)", dict(adpa_mode="fixed", no_aux=True, loss_schedule="none")),
    (5, "+ Adaptive gate (ADPA)", dict(adpa_mode="adaptive", no_aux=True, loss_schedule="none")),
    (6, "+ ADPA + weighted loss", dict(adpa_mode="adaptive", no_aux=False, loss_schedule="constant")),
    (7, "AAE-CNN (full, + curriculum)", dict(adpa_mode="adaptive", no_aux=False, loss_schedule="curriculum")),
]


def main():
    p = common.add_common_args(argparse.ArgumentParser(description="AAE-CNN component ablation"))
    p = aae_cnn.add_aae_args(p)
    p.add_argument("--only", type=int, nargs="*", default=None, help="configuration numbers to run")
    args = p.parse_args()
    split = common.make_split(args.data_dir, seed=args.seed, split_file=common.default_split_file(args))
    rows = []
    for num, label, cfg in CONFIGS:
        if args.only and num not in args.only:
            continue
        a = copy.deepcopy(args)
        for k, v in cfg.items():
            setattr(a, k, v)
        a.placement = "per_block"
        print(f"\n######## Ablation {num}: {label} ########")
        res = aae_cnn.train_aae(a, split=split, run_name=f"ablation_{num}")
        rows.append({"config": num, "description": label, "test_accuracy": round(100 * res["accuracy"], 2),
                     "macro_f1": round(100 * res["macro_f1"], 2), "best_val_acc": round(100 * res["best_val_acc"], 2),
                     "params_M": round(res["params"] / 1e6, 3)})
        common.write_csv(os.path.join(args.out_dir, "ablation_results.csv"), rows)
    print("\nAblation results")
    for r in rows:
        print(f"{r['config']}. {r['description']:<36} acc {r['test_accuracy']:6.2f}  macro-F1 {r['macro_f1']:6.2f}")


if __name__ == "__main__":
    main()
