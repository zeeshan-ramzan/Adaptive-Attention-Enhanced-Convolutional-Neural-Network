#!/usr/bin/env bash
# Quick end-to-end check on a tiny synthetic dataset (CPU, a few minutes).
set -e
python make_dummy_dataset.py --out dummy_plantvillage --per-class 30
C="--epochs 1 --batch-size 8 --img-size 64 --no-pretrained --workers 0"
for m in aae_cnn resnet50 densenet121 mobilenetv2 efficientnet_b4 se_resnet50 cbam_resnet50 vit_b16; do
  python $m.py --data-dir dummy_plantvillage --out-dir smoke_runs $C
done
python ablation_study.py --data-dir dummy_plantvillage --out-dir smoke_runs $C --only 1 7
python error_severity.py --npz smoke_runs/aae_cnn/test_outputs.npz --out-dir smoke_analysis
python gradcam.py --data-dir dummy_plantvillage --checkpoint smoke_runs/aae_cnn/best.pt --out smoke_analysis/gradcam.png
echo "Smoke test finished successfully."
