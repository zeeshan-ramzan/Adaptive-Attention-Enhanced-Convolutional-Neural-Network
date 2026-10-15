"""
loco_generalization.py  --  Leave-one-crop-out (LOCO) generalisation test
========================================================================
(paper Table "Leave-one-crop-out generalization accuracy")

For each held-out crop:
  1. all classes of that crop are removed and AAE-CNN is retrained on the remaining classes;
  2. the trained network is frozen and used as a feature extractor (1024-d pooled features);
  3. a linear probe (multinomial logistic regression) is fitted on a support set of
     20 training images per class for ALL classes (the held-out crop's classes and the
     classes seen during training), so that single-class crops such as Blueberry can
     also be evaluated;
  4. LOCO accuracy = fraction of the held-out crop's test images that the probe assigns
     to their correct class.

Usage:
  python loco_generalization.py --data-dir /path/PlantVillage --out-dir runs_loco \
      --crops Potato Cherry Corn Blueberry
"""
import argparse
import copy
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import aae_cnn
import common


@torch.no_grad()
def features(model, samples, img_size, device, batch=64, workers=4):
    ds = common.SubsetWithTransform(samples, common.build_transforms(img_size, train=False))
    feats, ys = [], []
    model.eval()
    for x, y in DataLoader(ds, batch_size=batch, num_workers=workers):
        feats.append(model.extract_features(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)


def main():
    p = common.add_common_args(argparse.ArgumentParser(description="Leave-one-crop-out test"))
    p = aae_cnn.add_aae_args(p)
    p.add_argument("--crops", nargs="*", default=["Potato", "Cherry", "Corn", "Blueberry"])
    p.add_argument("--support", type=int, default=20, help="support images per class for the linear probe")
    args = p.parse_args()
    device = common.get_device(args.device)
    full = common.make_split(args.data_dir, seed=args.seed, split_file=common.default_split_file(args))
    crops = {c: common.crop_of(c) for c in full.class_names}
    rng = np.random.default_rng(args.seed)
    rows = []
    for crop in args.crops:
        held = [i for i, c in enumerate(full.class_names) if crops[c].lower() == crop.lower()]
        if not held:
            print(f"Crop '{crop}' not found, skipped.")
            continue
        keep = [i for i in range(len(full.class_names)) if i not in held]
        remap = {old: new for new, old in enumerate(keep)}
        # reduced split without the held-out crop
        samples, tr, va, te = [], [], [], []
        for which, lst in (("train", tr), ("val", va), ("test", te)):
            for path, lab in full.subset(which):
                if lab in remap:
                    lst.append(len(samples))
                    samples.append((path, remap[lab]))
        reduced = common.DataSplit([full.class_names[i] for i in keep], samples, tr, va, te)
        a = copy.deepcopy(args)
        print(f"\n######## LOCO: holding out {crop} ({len(held)} classes) ########")
        res = aae_cnn.train_aae(a, split=reduced, run_name=f"loco_{crop}")
        model = res["model"].to(device)
        # linear probe: support set of `--support` images per class for all classes
        support = []
        for c in range(len(full.class_names)):
            pool = [smp for smp in full.subset("train") if smp[1] == c]
            idx = rng.choice(len(pool), size=min(args.support, len(pool)), replace=False)
            support += [pool[i] for i in idx]
        query = [smp for smp in full.subset("test") if smp[1] in held]
        Xs, ys = features(model, support, args.img_size, device, workers=args.workers)
        Xq, yq = features(model, query, args.img_size, device, workers=args.workers)
        sc = StandardScaler().fit(Xs)
        clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xs), ys)
        acc = float((clf.predict(sc.transform(Xq)) == yq).mean())
        note = "crops seen in training: " + ", ".join(sorted({crops[full.class_names[i]] for i in keep}))
        rows.append({"held_out_crop": crop, "n_classes": len(held), "support_per_class": args.support,
                     "n_query": len(query), "loco_accuracy": round(100 * acc, 2), "note": note})
        print(rows[-1])
        common.write_csv(os.path.join(args.out_dir, "loco_results.csv"), rows)


if __name__ == "__main__":
    main()
