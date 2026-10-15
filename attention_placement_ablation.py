"""
attention_placement_ablation.py  --  Where to place ADPA (paper Table "Attention placement")
==========================================================================================
Compares four placement strategies of the ADPA module (full AAE-CNN loss in all cases):

  early     : one ADPA after Dense Block 1
  late      : one ADPA after Dense Block 3
  all_layer : an ADPA module on the output of every dense layer of Dense Blocks 1-3
  per_block : one ADPA after each of Dense Blocks 1, 2 and 3 (AAE-CNN)

Reports validation accuracy and parameter overhead relative to the backbone with the
same heads but without ADPA.

Usage:  python attention_placement_ablation.py --data-dir /path/PlantVillage --out-dir runs
"""
import argparse
import copy
import os

import aae_cnn
import common


def main():
    p = common.add_common_args(argparse.ArgumentParser(description="ADPA placement ablation"))
    p = aae_cnn.add_aae_args(p)
    p.add_argument("--strategies", nargs="*", default=["early", "late", "all_layer", "per_block"])
    args = p.parse_args()
    split = common.make_split(args.data_dir, seed=args.seed, split_file=common.default_split_file(args))
    n_cls = len(split.class_names)
    base = common.count_params(aae_cnn.AAECNN(n_cls, pretrained=False, adpa_mode="none", placement="none"))
    rows = []
    for strat in args.strategies:
        a = copy.deepcopy(args)
        a.placement, a.adpa_mode = strat, "adaptive"
        print(f"\n######## Placement: {strat} ########")
        res = aae_cnn.train_aae(a, split=split, run_name=f"placement_{strat}")
        overhead = 100.0 * (res["params"] - base) / base
        rows.append({"placement": strat, "num_adpa_modules": res["num_adpa_modules"],
                     "val_accuracy": round(100 * res["best_val_acc"], 2),
                     "test_accuracy": round(100 * res["accuracy"], 2),
                     "param_overhead_percent": round(overhead, 2)})
        common.write_csv(os.path.join(args.out_dir, "placement_results.csv"), rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
