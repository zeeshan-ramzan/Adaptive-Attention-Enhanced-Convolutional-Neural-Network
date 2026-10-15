"""
input_resolution.py  --  Effect of input resolution on accuracy and latency
==========================================================================
(paper Figure "Test accuracy and inference latency as a function of input resolution")

Retrains AAE-CNN at 128, 160, 224 and 320 pixels and records test accuracy and
inference latency per image at each resolution.

Usage:  python input_resolution.py --data-dir /path/PlantVillage --out-dir runs_res
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
    p = common.add_common_args(argparse.ArgumentParser(description="Input-resolution study"))
    p = aae_cnn.add_aae_args(p)
    p.add_argument("--resolutions", type=int, nargs="*", default=[128, 160, 224, 320])
    args = p.parse_args()
    split = common.make_split(args.data_dir, seed=args.seed, split_file=common.default_split_file(args))
    device = common.get_device(args.device)
    rows = []
    for res in args.resolutions:
        a = copy.deepcopy(args)
        a.img_size = res
        print(f"\n######## input {res}x{res} ########")
        r = aae_cnn.train_aae(a, split=split, run_name=f"res_{res}")
        model = r["model"].eval()
        model.aux_head = None
        lat = common.measure_latency(model, res, device)
        rows.append({"resolution": res, "test_accuracy": round(100 * r["accuracy"], 2), "latency_ms": round(lat, 2)})
        common.write_csv(os.path.join(args.out_dir, "input_resolution.csv"), rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"{r['resolution']}" for r in rows]
    bars = ax.bar(labels, [r["test_accuracy"] for r in rows], color="#4C72B0")
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{r['test_accuracy']:.2f}%\n{r['latency_ms']:.1f} ms",
                ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Input resolution (pixels)")
    ax.set_ylabel("Test accuracy (%)")
    lo = min(r["test_accuracy"] for r in rows)
    ax.set_ylim(max(0, lo - 2), 100.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "input_resolution.png"), dpi=300)


if __name__ == "__main__":
    main()
