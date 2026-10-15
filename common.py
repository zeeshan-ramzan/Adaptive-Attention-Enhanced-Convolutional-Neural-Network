"""
common.py
=========
Shared utilities used by every script in this package:

* reproducible stratified 80/10/10 split of the PlantVillage folder
* training / validation / test transforms (Section "Dataset splits and preprocessing")
* a generic training loop (Adam, cosine annealing, mixed precision, best-checkpoint saving)
* evaluation and metrics (accuracy, macro precision / recall / F1, macro one-vs-rest AUC)
* parameter / MAC counting and latency measurement
* a model registry so any script can build any model by name

The PlantVillage dataset must be arranged in the usual ImageFolder layout::

    PlantVillage/
        Apple___Apple_scab/xxx.jpg
        Apple___Black_rot/...
        ...
        Tomato___healthy/...

(The colour version of https://github.com/spMohanty/PlantVillage-Dataset already has this layout.)
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Names of the model modules that live in this folder (file name without .py)
MODEL_MODULES = {
    "aae_cnn": "aae_cnn",
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "mobilenetv2": "mobilenetv2",
    "efficientnet_b4": "efficientnet_b4",
    "se_resnet50": "se_resnet50",
    "cbam_resnet50": "cbam_resnet50",
    "vit_b16": "vit_b16",
}


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# --------------------------------------------------------------------------- #
# Class-name helpers
# --------------------------------------------------------------------------- #
def crop_of(class_name: str) -> str:
    """Return the crop species of a PlantVillage class folder name.

    'Corn_(maize)___Common_rust_' -> 'Corn', 'Pepper,_bell___healthy' -> 'Pepper',
    'Cherry_(including_sour)___healthy' -> 'Cherry'.
    """
    head = class_name.split("___")[0] if "___" in class_name else class_name.split("_")[0]
    head = head.split("(")[0].split(",")[0]
    return head.strip("_ ").replace("_", " ").split(" ")[0].capitalize()


def is_healthy(class_name: str) -> bool:
    return "healthy" in class_name.lower()


# --------------------------------------------------------------------------- #
# Transforms (paper: resize 224, ImageNet normalisation, flips p=0.5,
# rotation +-30 deg, colour jitter 0.2, random erasing p=0.1)
# --------------------------------------------------------------------------- #
def build_transforms(img_size: int = 224, train: bool = True, augment: bool = True):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train and augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(p=0.1),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        norm,
    ])


# --------------------------------------------------------------------------- #
# Dataset and split
# --------------------------------------------------------------------------- #
class SubsetWithTransform(Dataset):
    """A subset of (path, label) samples with its own transform.

    `pre_transform` (optional) is applied to the PIL image before `transform`;
    it is used e.g. for synthetic corruptions in the robustness experiment.
    """

    def __init__(self, samples: Sequence[Tuple[str, int]], transform=None,
                 pre_transform: Optional[Callable] = None, label_map: Optional[Dict[int, int]] = None):
        self.samples = list(samples)
        self.transform = transform
        self.pre_transform = pre_transform
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.pre_transform is not None:
            img = self.pre_transform(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.label_map is not None:
            label = self.label_map[label]
        return img, label


@dataclass
class DataSplit:
    class_names: List[str]
    samples: List[Tuple[str, int]]
    train_idx: List[int]
    val_idx: List[int]
    test_idx: List[int]

    def subset(self, which: str) -> List[Tuple[str, int]]:
        idx = {"train": self.train_idx, "val": self.val_idx, "test": self.test_idx}[which]
        return [self.samples[i] for i in idx]

    def labels(self, which: str) -> np.ndarray:
        return np.array([s[1] for s in self.subset(which)])


def scan_dataset(data_dir: str) -> Tuple[List[str], List[Tuple[str, int]]]:
    folder = datasets.ImageFolder(data_dir)
    return folder.classes, folder.samples


def make_split(data_dir: str, seed: int = 42, val_frac: float = 0.10, test_frac: float = 0.10,
               split_file: Optional[str] = None) -> DataSplit:
    """Stratified 80/10/10 split. If `split_file` exists it is re-used, so that
    every model is trained and tested on exactly the same images."""
    class_names, samples = scan_dataset(data_dir)
    if split_file and os.path.isfile(split_file):
        with open(split_file) as f:
            d = json.load(f)
        if d.get("n_samples") == len(samples) and d.get("class_names") == class_names:
            return DataSplit(class_names, samples, d["train"], d["val"], d["test"])
    labels = np.array([s[1] for s in samples])
    idx = np.arange(len(samples))
    trainval, test = train_test_split(idx, test_size=test_frac, stratify=labels, random_state=seed)
    rel_val = val_frac / (1.0 - test_frac)
    train, val = train_test_split(trainval, test_size=rel_val, stratify=labels[trainval], random_state=seed)
    split = DataSplit(class_names, samples, sorted(train.tolist()), sorted(val.tolist()), sorted(test.tolist()))
    if split_file:
        os.makedirs(os.path.dirname(os.path.abspath(split_file)), exist_ok=True)
        with open(split_file, "w") as f:
            json.dump({"n_samples": len(samples), "class_names": class_names, "seed": seed,
                       "train": split.train_idx, "val": split.val_idx, "test": split.test_idx}, f)
    return split


def subsample_fraction(indices: Sequence[int], labels: np.ndarray, fraction: float, seed: int) -> List[int]:
    """Stratified subsample of the training indices (learning-curve experiment)."""
    if fraction >= 1.0:
        return list(indices)
    indices = np.asarray(indices)
    keep, _ = train_test_split(indices, train_size=fraction, stratify=labels[indices], random_state=seed)
    return sorted(keep.tolist())


def make_loaders(split: DataSplit, img_size: int = 224, batch_size: int = 32, workers: int = 4,
                 train_idx: Optional[Sequence[int]] = None, augment: bool = True):
    train_samples = [split.samples[i] for i in (train_idx if train_idx is not None else split.train_idx)]
    pin = torch.cuda.is_available()
    train_ds = SubsetWithTransform(train_samples, build_transforms(img_size, True, augment))
    val_ds = SubsetWithTransform(split.subset("val"), build_transforms(img_size, False))
    test_ds = SubsetWithTransform(split.subset("test"), build_transforms(img_size, False))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                          pin_memory=pin, drop_last=len(train_ds) > batch_size)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    return train_dl, val_dl, test_dl


def class_counts(samples: Sequence[Tuple[str, int]], num_classes: int) -> np.ndarray:
    c = Counter(s[1] for s in samples)
    return np.array([c.get(i, 0) for i in range(num_classes)], dtype=np.float64)


def inverse_frequency_weights(counts: np.ndarray) -> torch.Tensor:
    """w_c = N / (C * n_c)  (paper, curriculum auxiliary loss)."""
    counts = np.maximum(counts, 1.0)
    n, c = counts.sum(), len(counts)
    return torch.tensor(n / (c * counts), dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
def load_model_module(name: str):
    if name not in MODEL_MODULES:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_MODULES)}")
    return importlib.import_module(MODEL_MODULES[name])


def build_model(name: str, num_classes: int, pretrained: bool = True, img_size: int = 224, **kwargs) -> nn.Module:
    return load_model_module(name).build_model(num_classes=num_classes, pretrained=pretrained,
                                               img_size=img_size, **kwargs)


def make_criterion(name: str, counts: np.ndarray, epochs: int, device, **kwargs):
    """Returns criterion(outputs, targets, epoch) -> loss.
    The proposed model provides its own curriculum loss; baselines use cross-entropy."""
    mod = load_model_module(name)
    if hasattr(mod, "make_criterion"):
        return mod.make_criterion(counts=counts, epochs=epochs, device=device, **kwargs)
    ce = nn.CrossEntropyLoss()

    def criterion(outputs, targets, epoch):
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        return ce(outputs, targets)
    return criterion


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_model(model: nn.Module, train_dl, val_dl, criterion, epochs: int = 100, lr: float = 3e-4,
                weight_decay: float = 1e-4, device=None, out_dir: Optional[str] = None,
                amp: bool = True, log_every: int = 50, extra_ckpt: Optional[dict] = None,
                verbose: bool = True) -> Dict:
    """Adam + cosine annealing (T_max = epochs, no warm restarts), mixed precision on GPU.
    Keeps the checkpoint with the best validation accuracy (Algorithm 1)."""
    device = device or get_device()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_acc, best_state, history = -1.0, None, []
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        t0, run_loss, correct, seen = time.time(), 0.0, 0, 0
        for it, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y, epoch)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            logits = out[0] if isinstance(out, (tuple, list)) else out
            run_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
            if verbose and log_every and (it + 1) % log_every == 0:
                print(f"  epoch {epoch} it {it + 1}/{len(train_dl)} loss {run_loss / seen:.4f}")
        sched.step()
        train_acc = correct / max(seen, 1)
        val_acc, val_loss = quick_eval(model, val_dl, device)
        history.append({"epoch": epoch, "train_loss": run_loss / max(seen, 1), "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "lr": opt.param_groups[0]["lr"],
                        "time_s": time.time() - t0})
        if verbose:
            print(f"Epoch {epoch:3d}/{epochs}  train_loss {run_loss / max(seen, 1):.4f}  "
                  f"train_acc {train_acc:.4f}  val_acc {val_acc:.4f}  ({time.time() - t0:.1f}s)")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if out_dir:
                ckpt = {"state_dict": best_state, "epoch": epoch, "val_acc": val_acc}
                if extra_ckpt:
                    ckpt.update(extra_ckpt)
                torch.save(ckpt, os.path.join(out_dir, "best.pt"))
    if out_dir:
        write_csv(os.path.join(out_dir, "history.csv"), history)
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_acc": best_acc, "history": history}


@torch.no_grad()
def quick_eval(model: nn.Module, loader, device) -> Tuple[float, float]:
    model.eval()
    correct, seen, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        loss_sum += F.cross_entropy(logits.float(), y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        seen += y.size(0)
    return correct / max(seen, 1), loss_sum / max(seen, 1)


@torch.no_grad()
def predict(model: nn.Module, loader, device=None) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (y_true, softmax probabilities)."""
    device = device or get_device()
    model.to(device).eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        ps.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Macro-averaged one-vs-rest AUC over the classes present in y_true."""
    aucs = []
    for c in range(y_prob.shape[1]):
        pos = (y_true == c)
        if pos.sum() == 0 or pos.sum() == len(y_true):
            continue
        aucs.append(roc_auc_score(pos.astype(int), y_prob[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: Sequence[str]) -> Dict:
    y_pred = y_prob.argmax(1)
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_auc_ovr": macro_auc(y_true, y_prob),
        "n_test": int(len(y_true)),
        "n_errors": int((y_pred != y_true).sum()),
        "per_class": {class_names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                       "f1": float(f[i]), "support": int(s[i])} for i in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def print_metrics(m: Dict, title: str = "") -> None:
    print(f"\n==== {title} ====")
    print(f"Accuracy        : {100 * m['accuracy']:.2f}%")
    print(f"Macro precision : {100 * m['macro_precision']:.2f}%")
    print(f"Macro recall    : {100 * m['macro_recall']:.2f}%")
    print(f"Macro F1        : {100 * m['macro_f1']:.2f}%")
    print(f"Macro AUC (OvR) : {m['macro_auc_ovr']:.4f}")
    print(f"Errors          : {m['n_errors']} / {m['n_test']}")


# --------------------------------------------------------------------------- #
# Complexity and latency
# --------------------------------------------------------------------------- #
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_macs(model: nn.Module, img_size: int = 224, device="cpu") -> int:
    """Multiply-accumulate count of Conv2d, Linear and multi-head attention layers
    for one image (hook based, no extra dependency)."""
    macs = [0]
    hooks = []

    def conv_hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        macs[0] += out.numel() // out.shape[0] * k

    def lin_hook(m, inp, out):
        macs[0] += out.numel() // out.shape[0] * m.in_features

    def mha_hook(m, inp, out):
        q = inp[0]  # (B, L, E) when batch_first
        L, E = (q.shape[1], q.shape[2]) if m.batch_first else (q.shape[0], q.shape[2])
        macs[0] += 4 * L * E * E + 2 * L * L * E  # projections + attention matmuls

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear) and not isinstance(mod, nn.modules.linear.NonDynamicallyQuantizableLinear):
            hooks.append(mod.register_forward_hook(lin_hook))
        elif isinstance(mod, nn.MultiheadAttention):
            hooks.append(mod.register_forward_hook(mha_hook))
    was_training = model.training
    model.eval().to(device)
    model(torch.zeros(1, 3, img_size, img_size, device=device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return macs[0]


@torch.no_grad()
def measure_latency(model: nn.Module, img_size: int = 224, device=None, runs: int = 50, warmup: int = 10,
                    batch: int = 1, half: bool = False) -> float:
    """Mean inference time per image in milliseconds."""
    device = device or get_device()
    model = model.to(device).eval()
    x = torch.randn(batch, 3, img_size, img_size, device=device)
    if half:
        model, x = model.half(), x.half()
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs / batch * 1000.0


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_checkpoint(path: str, device=None) -> Tuple[nn.Module, Dict]:
    """Rebuild a model saved by any training script in this package."""
    device = device or get_device()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    kw = ckpt.get("model_kwargs", {})
    model = build_model(ckpt["model_name"], num_classes=len(ckpt["class_names"]), pretrained=False,
                        img_size=ckpt.get("img_size", 224), **kw)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


# --------------------------------------------------------------------------- #
# Command-line interface shared by all training scripts
# --------------------------------------------------------------------------- #
def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--data-dir", required=True, help="PlantVillage root folder (ImageFolder layout)")
    p.add_argument("--out-dir", default="runs", help="where checkpoints and results are written")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-pretrained", action="store_true", help="train from scratch (no ImageNet weights)")
    p.add_argument("--no-amp", action="store_true", help="disable FP16 mixed precision")
    p.add_argument("--no-augment", action="store_true", help="disable training augmentation")
    p.add_argument("--split-file", default=None,
                   help="JSON with the fixed split (default: <out-dir>/split_seed<seed>.json)")
    return p


def default_split_file(args) -> str:
    return args.split_file or os.path.join(args.out_dir, f"split_seed{args.seed}.json")


def run_training(model_name: str, args, model_kwargs: Optional[dict] = None,
                 criterion_kwargs: Optional[dict] = None, run_name: Optional[str] = None,
                 train_idx: Optional[Sequence[int]] = None, split: Optional[DataSplit] = None,
                 verbose: bool = True) -> Dict:
    """Train one model with the common protocol, evaluate it on the test split and
    save best.pt, history.csv, test_metrics.json and test_outputs.npz."""
    model_kwargs = model_kwargs or {}
    criterion_kwargs = criterion_kwargs or {}
    set_seed(args.seed)
    device = get_device(args.device)
    split = split or make_split(args.data_dir, seed=args.seed, split_file=default_split_file(args))
    n_cls = len(split.class_names)
    out_dir = os.path.join(args.out_dir, run_name or model_name)
    os.makedirs(out_dir, exist_ok=True)
    train_dl, val_dl, test_dl = make_loaders(split, args.img_size, args.batch_size, args.workers,
                                             train_idx=train_idx, augment=not args.no_augment)
    tr_samples = train_dl.dataset.samples
    counts = class_counts(tr_samples, n_cls)
    model = build_model(model_name, n_cls, pretrained=not args.no_pretrained, img_size=args.img_size,
                        **model_kwargs)
    criterion = make_criterion(model_name, counts, args.epochs, device, **criterion_kwargs)
    if verbose:
        print(f"[{run_name or model_name}] classes={n_cls} train={len(tr_samples)} "
              f"val={len(val_dl.dataset)} test={len(test_dl.dataset)} params={count_params(model) / 1e6:.2f}M "
              f"device={device}")
    extra = {"model_name": model_name, "class_names": split.class_names, "img_size": args.img_size,
             "model_kwargs": model_kwargs}
    res = train_model(model, train_dl, val_dl, criterion, epochs=args.epochs, lr=args.lr,
                      weight_decay=args.weight_decay, device=device, out_dir=out_dir,
                      amp=not args.no_amp, extra_ckpt=extra, verbose=verbose)
    y_true, y_prob = predict(model, test_dl, device)
    metrics = compute_metrics(y_true, y_prob, split.class_names)
    metrics.update({"best_val_acc": res["best_val_acc"], "params": count_params(model),
                    "model_name": model_name, "run_name": run_name or model_name})
    np.savez_compressed(os.path.join(out_dir, "test_outputs.npz"), y_true=y_true, y_prob=y_prob,
                        class_names=np.array(split.class_names))
    save_json(os.path.join(out_dir, "test_metrics.json"), metrics)
    if verbose:
        print_metrics(metrics, run_name or model_name)
    metrics["model"] = model
    return metrics
