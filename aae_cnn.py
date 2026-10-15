"""
aae_cnn.py  --  PROPOSED METHOD
===============================
AAE-CNN: Adaptive Attention-Enhanced Convolutional Neural Network
(adaptive dual-path attention + curriculum-weighted auxiliary loss).

Architecture (paper, Section "Proposed methodology")
  * Backbone: standard DenseNet-121 (4 dense blocks with 6/12/24/16 layers, growth rate 32,
    transition compression 0.5), ImageNet initialisation.
  * ADPA module after Dense Blocks 1, 2 and 3 (feature maps 56x56, 28x28, 14x14 for 224 input):
        SE branch : z = GAP(F); s = sigmoid(W2 relu(W1 z)), r = 16;  F_SE = s * F          (Eqs. 2-3)
        SA branch : M = sigmoid(conv7x7([AvgPool_c(F); MaxPool_c(F)]));  F_SA = M * F       (Eq. 4)
        Gate      : F' = sigmoid(alpha) F_SE + sigmoid(beta) F_SA + F,  alpha = beta = 0.5 at init (Eq. 5)
  * Main head : GAP -> FC(1024) -> ReLU -> Dropout(0.4) -> FC(num_classes)
  * Auxiliary head (training only) after Dense Block 2 / ADPA-2:
                GAP -> FC(512) -> ReLU -> Dropout(0.4) -> FC(num_classes)
Loss (Eqs. 6-7)
  L_total = CE(p_main, y) + lambda(t) * WCE(p_aux, y),  w_c = N / (C n_c),
  lambda(t) = min(1, rate * t / T), rate = 2 (default)

Algorithm 1 (training): Adam (lr 3e-4, betas 0.9/0.999, weight decay 1e-4), batch 32, 100 epochs,
cosine annealing (T_max = T), FP16 mixed precision, best checkpoint on validation accuracy.

Ablation / placement variants are available through command-line flags:
  --adpa-mode  adaptive | fixed | se | sa | none
  --placement  per_block | early | late | all_layer | none
  --loss-schedule curriculum | constant | none      (constant = weighted loss without curriculum)

Usage
  python aae_cnn.py --data-dir /path/PlantVillage --out-dir runs
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121

import common


# --------------------------------------------------------------------------- #
# ADPA module
# --------------------------------------------------------------------------- #
class SEBranch(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al.)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        z = x.mean(dim=(2, 3))                                # squeeze, Eq. (2)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(z))))      # excitation, Eq. (3)
        return x * s[:, :, None, None]


class SpatialBranch(nn.Module):
    """Spatial attention: 7x7 convolution over channel-wise average and max maps."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        m = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))   # Eq. (4)
        return x * m


class ADPA(nn.Module):
    """Adaptive Dual-Path Attention.

    mode = 'adaptive' : F' = s(a) F_SE + s(b) F_SA + F  with learned scalars a, b (init 0.5)
           'fixed'    : F' = 0.5 F_SE + 0.5 F_SA + F   (fixed dual attention, ablation)
           'se'       : F' = s(a) F_SE + F             (SE channel attention only, ablation)
           'sa'       : F' = s(b) F_SA + F             (spatial attention only, ablation)
    """

    def __init__(self, channels: int, reduction: int = 16, mode: str = "adaptive", gate_init: float = 0.5):
        super().__init__()
        assert mode in ("adaptive", "fixed", "se", "sa")
        self.mode = mode
        self.se = SEBranch(channels, reduction) if mode != "sa" else None
        self.sa = SpatialBranch() if mode != "se" else None
        if mode != "fixed":
            self.alpha = nn.Parameter(torch.tensor(gate_init))
            self.beta = nn.Parameter(torch.tensor(gate_init))

    def gates(self):
        if self.mode == "fixed":
            return 0.5, 0.5
        a = torch.sigmoid(self.alpha).item() if self.se is not None else 0.0
        b = torch.sigmoid(self.beta).item() if self.sa is not None else 0.0
        return a, b

    def forward(self, x):
        out = x
        if self.mode == "fixed":
            return 0.5 * self.se(x) + 0.5 * self.sa(x) + x
        if self.se is not None:
            out = out + torch.sigmoid(self.alpha) * self.se(x)
        if self.sa is not None:
            out = out + torch.sigmoid(self.beta) * self.sa(x)
        return out                                              # Eq. (5)


class DenseLayerWithADPA(nn.Module):
    """Wraps a torchvision _DenseLayer so that its new feature maps pass through an
    ADPA module (used only for the 'all_layer' placement ablation)."""

    def __init__(self, layer: nn.Module, growth: int, reduction: int, mode: str):
        super().__init__()
        self.layer = layer
        self.adpa = ADPA(growth, reduction=min(reduction, growth // 2), mode=mode)

    def forward(self, *prev):
        return self.adpa(self.layer(*prev))


# --------------------------------------------------------------------------- #
# AAE-CNN
# --------------------------------------------------------------------------- #
class AAECNN(nn.Module):
    BLOCK_CHANNELS = {1: 256, 2: 512, 3: 1024}   # DenseNet-121 output channels of dense blocks 1-3

    def __init__(self, num_classes: int = 38, pretrained: bool = True, adpa_mode: str = "adaptive",
                 placement: str = "per_block", reduction: int = 16, aux_head: bool = True,
                 dropout: float = 0.4):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        f = densenet121(weights=weights).features
        self.stem = nn.Sequential(f.conv0, f.norm0, f.relu0, f.pool0)
        self.block1, self.trans1 = f.denseblock1, f.transition1
        self.block2, self.trans2 = f.denseblock2, f.transition2
        self.block3, self.trans3 = f.denseblock3, f.transition3
        self.block4, self.norm5 = f.denseblock4, f.norm5
        self.adpa_mode, self.placement = adpa_mode, placement

        use = {"per_block": (1, 2, 3), "early": (1,), "late": (3,), "all_layer": (), "none": ()}[placement]
        if adpa_mode == "none":
            use = ()
        self.adpa = nn.ModuleDict({str(k): ADPA(self.BLOCK_CHANNELS[k], reduction, adpa_mode) for k in use})
        if placement == "all_layer" and adpa_mode != "none":
            for blk in (self.block1, self.block2, self.block3):
                for name, layer in list(blk.named_children()):
                    setattr(blk, name, DenseLayerWithADPA(layer, 32, reduction, adpa_mode))

        self.head = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                  nn.Linear(1024, num_classes))
        self.aux_head = (nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                       nn.Linear(512, num_classes)) if aux_head else None)

    def _adpa(self, k: int, x):
        key = str(k)
        return self.adpa[key](x) if key in self.adpa else x

    def forward_features(self, x):
        x = self.stem(x)
        x = self._adpa(1, self.block1(x))
        x = self.trans1(x)
        x2 = self._adpa(2, self.block2(x))
        x = self.trans2(x2)
        x = self._adpa(3, self.block3(x))
        x = self.trans3(x)
        x = F.relu(self.norm5(self.block4(x)))
        return x, x2

    def extract_features(self, x):
        """1024-d pooled embedding (used by the LOCO linear-probe experiment)."""
        fmap, _ = self.forward_features(x)
        return F.adaptive_avg_pool2d(fmap, 1).flatten(1)

    def forward(self, x):
        fmap, x2 = self.forward_features(x)
        logits = self.head(F.adaptive_avg_pool2d(fmap, 1).flatten(1))
        if self.training and self.aux_head is not None:
            aux = self.aux_head(F.adaptive_avg_pool2d(x2, 1).flatten(1))
            return logits, aux                     # auxiliary head used during training only
        return logits

    def gate_values(self):
        out = {f"ADPA-{k}": dict(zip(("se_gate", "sa_gate"), m.gates())) for k, m in self.adpa.items()}
        return out

    def num_adpa_modules(self):
        return sum(1 for m in self.modules() if isinstance(m, ADPA))


def build_model(num_classes: int, pretrained: bool = True, img_size: int = 224, **kwargs) -> nn.Module:
    return AAECNN(num_classes=num_classes, pretrained=pretrained, **kwargs)


# --------------------------------------------------------------------------- #
# Curriculum auxiliary loss
# --------------------------------------------------------------------------- #
class CurriculumAuxLoss(nn.Module):
    """L = CE(main) + lambda(t) * WCE(aux), lambda(t) = min(1, rate * t / T)."""

    def __init__(self, class_weights: torch.Tensor, total_epochs: int, rate: float = 2.0,
                 schedule: str = "curriculum"):
        super().__init__()
        assert schedule in ("curriculum", "constant", "none")
        self.ce = nn.CrossEntropyLoss()
        self.wce = nn.CrossEntropyLoss(weight=class_weights)
        self.T, self.rate, self.schedule = total_epochs, rate, schedule

    def lam(self, epoch: int) -> float:
        if self.schedule == "none":
            return 0.0
        if self.schedule == "constant":
            return 1.0
        return min(1.0, self.rate * epoch / self.T)              # Eq. (7)

    def forward(self, outputs, targets, epoch: int):
        if isinstance(outputs, (tuple, list)):
            main, aux = outputs
            loss = self.ce(main.float(), targets)
            lam = self.lam(epoch)
            if lam > 0:
                loss = loss + lam * self.wce(aux.float(), targets)  # Eq. (6)
            return loss
        return self.ce(outputs.float(), targets)


def make_criterion(counts, epochs: int, device, rate: float = 2.0, schedule: str = "curriculum", **_):
    w = common.inverse_frequency_weights(counts).to(device)
    return CurriculumAuxLoss(w, epochs, rate=rate, schedule=schedule).to(device)


# --------------------------------------------------------------------------- #
# Training entry point (Algorithm 1)
# --------------------------------------------------------------------------- #
def add_aae_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--adpa-mode", default="adaptive", choices=["adaptive", "fixed", "se", "sa", "none"])
    p.add_argument("--placement", default="per_block", choices=["per_block", "early", "late", "all_layer", "none"])
    p.add_argument("--reduction", type=int, default=16, help="SE reduction ratio r")
    p.add_argument("--no-aux", action="store_true", help="remove the auxiliary head")
    p.add_argument("--loss-schedule", default="curriculum", choices=["curriculum", "constant", "none"])
    p.add_argument("--curriculum-rate", type=float, default=2.0, help="slope of lambda(t) = min(1, rate t/T)")
    p.add_argument("--run-name", default=None)
    return p


def aae_kwargs_from_args(args):
    model_kwargs = {"adpa_mode": args.adpa_mode, "placement": args.placement,
                    "reduction": args.reduction, "aux_head": not args.no_aux}
    crit_kwargs = {"rate": args.curriculum_rate,
                   "schedule": "none" if args.no_aux else args.loss_schedule}
    return model_kwargs, crit_kwargs


def train_aae(args, train_idx=None, split=None, run_name=None, verbose=True):
    model_kwargs, crit_kwargs = aae_kwargs_from_args(args)
    res = common.run_training("aae_cnn", args, model_kwargs=model_kwargs, criterion_kwargs=crit_kwargs,
                              run_name=run_name or args.run_name or "aae_cnn", train_idx=train_idx,
                              split=split, verbose=verbose)
    model = res["model"]
    res["gate_values"] = model.gate_values()
    res["num_adpa_modules"] = model.num_adpa_modules()
    out_dir = os.path.join(args.out_dir, run_name or args.run_name or "aae_cnn")
    common.save_json(os.path.join(out_dir, "gate_values.json"), res["gate_values"])
    if verbose:
        print("Learned gate values (sigmoid):", res["gate_values"])
        print("ADPA modules:", res["num_adpa_modules"])
    return res


def main():
    p = common.add_common_args(argparse.ArgumentParser(description="Train the proposed AAE-CNN"))
    args = add_aae_args(p).parse_args()
    train_aae(args)


if __name__ == "__main__":
    main()
