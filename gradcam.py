"""
gradcam.py  --  Grad-CAM class activation maps (Selvaraju et al., 2017)
======================================================================
(paper Figure "Grad-CAM activation maps")

Grad-CAM weights the feature maps A_k of the last convolutional stage by the spatially
averaged gradients of the class score y_c:
    w_k = mean_ij dy_c / dA_k(i,j),   L_c = ReLU(sum_k w_k A_k)
The map is upsampled to the input size and overlaid on the leaf image.

Default target layers: aae_cnn -> norm5 (output of Dense Block 4), densenet121 -> features.norm5,
(se_/cbam_)resnet50 -> layer4, mobilenetv2 / efficientnet_b4 -> features. ViT is not supported.

Usage:
  python gradcam.py --data-dir /path/PlantVillage --checkpoint runs/aae_cnn/best.pt \
      --classes "Apple___Apple_scab" "Corn_(maize)___Common_rust_" "Grape___Black_rot" \
                "Tomato___Tomato_Yellow_Leaf_Curl_Virus" --out gradcam.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import common

DEFAULT_LAYER = {"aae_cnn": "norm5", "densenet121": "features.norm5", "resnet50": "layer4",
                 "se_resnet50": "layer4", "cbam_resnet50": "layer4", "mobilenetv2": "features",
                 "efficientnet_b4": "features"}


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.act, self.grad = model, None, None
        self.h = layer.register_forward_hook(self._fwd)

    def _fwd(self, m, i, o):
        self.act = o
        o.register_hook(lambda g: setattr(self, "grad", g))
        return o.clone()          # protects the stored activation from later in-place ReLUs

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        c = int(logits.argmax(1)) if class_idx is None else class_idx
        logits[0, c].backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), c, F.softmax(logits, 1)[0, c].item()


def get_layer(model, name):
    mod = model
    for part in name.split("."):
        mod = getattr(mod, part)
    return mod


def main():
    p = argparse.ArgumentParser(description="Grad-CAM visualisation")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--classes", nargs="*", default=None, help="class folder names (default: first 4 diseased)")
    p.add_argument("--target-layer", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-file", default=None)
    p.add_argument("--out", default="gradcam.png")
    args = p.parse_args()
    device = common.get_device()
    model, meta = common.load_checkpoint(args.checkpoint, device)
    if meta["model_name"] == "vit_b16":
        raise SystemExit("Grad-CAM on convolutional feature maps is not defined for ViT-B/16.")
    layer = get_layer(model, args.target_layer or DEFAULT_LAYER[meta["model_name"]])
    split_file = args.split_file or os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)),
                                                 f"split_seed{args.seed}.json")
    split = common.make_split(args.data_dir, seed=args.seed, split_file=split_file)
    names = split.class_names
    wanted = args.classes or [c for c in names if not common.is_healthy(c)][:4]
    size = meta.get("img_size", 224)
    tf = common.build_transforms(size, train=False)
    cam = GradCAM(model, layer)
    fig, axes = plt.subplots(2, len(wanted), figsize=(3.2 * len(wanted), 6.6), squeeze=False)
    for j, cname in enumerate(wanted):
        if cname not in names:
            raise SystemExit(f"Class '{cname}' not found. Available: {names}")
        ci = names.index(cname)
        path = next(pth for pth, lab in split.subset("test") if lab == ci)
        img = Image.open(path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device).requires_grad_(True)
        heat, pred, conf = cam(x)
        shown = np.asarray(img.resize((size, size))) / 255.0
        axes[0, j].imshow(shown)
        axes[0, j].set_title(cname.replace("___", "\n").replace("_", " "), fontsize=9)
        axes[1, j].imshow(shown)
        axes[1, j].imshow(heat, cmap="jet", alpha=0.45)
        axes[1, j].set_title(f"pred: {names[pred].split('___')[-1].replace('_', ' ')} ({conf:.2f})", fontsize=8)
        for a in axes[:, j]:
            a.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Grad-CAM figure saved to {args.out}")


if __name__ == "__main__":
    main()
