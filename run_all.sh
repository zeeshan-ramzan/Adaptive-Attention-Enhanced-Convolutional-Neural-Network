#!/usr/bin/env bash
# Full reproduction pipeline (Linux / macOS). Usage: bash run_all.sh /path/to/PlantVillage
set -e
DATA=${1:?give the PlantVillage folder}
OUT=runs
python dataset_statistics.py --data-dir "$DATA" --out-dir analysis
# 1. proposed model and the seven baselines (same split, same protocol)
for m in aae_cnn resnet50 densenet121 mobilenetv2 efficientnet_b4 se_resnet50 cbam_resnet50 vit_b16; do
  python $m.py --data-dir "$DATA" --out-dir $OUT
done
# 2. analyses of the trained models
python per_class_analysis.py --npz $OUT/*/test_outputs.npz --out-dir analysis
python roc_analysis.py --npz $OUT/aae_cnn/test_outputs.npz $OUT/cbam_resnet50/test_outputs.npz $OUT/densenet121/test_outputs.npz $OUT/vit_b16/test_outputs.npz --out analysis/roc_curves.png
python plot_training_curves.py --history $OUT/aae_cnn/history.csv --out analysis/training_curves.png
python error_severity.py --npz $OUT/aae_cnn/test_outputs.npz --out-dir analysis
python gradcam.py --data-dir "$DATA" --checkpoint $OUT/aae_cnn/best.pt --out analysis/gradcam.png
python robustness_corruption.py --data-dir "$DATA" --checkpoints $OUT/densenet121/best.pt $OUT/cbam_resnet50/best.pt $OUT/aae_cnn/best.pt --out analysis/robustness.csv
python deployment_benchmark.py --checkpoints $OUT/*/best.pt --data-dir "$DATA" --out-dir deploy
# 3. ensemble of three AAE-CNN seeds
for s in 43 44; do
  python aae_cnn.py --data-dir "$DATA" --out-dir $OUT --seed $s --run-name aae_seed$s --split-file $OUT/split_seed42.json
done
python ensemble.py --npz $OUT/aae_cnn/test_outputs.npz $OUT/aae_seed43/test_outputs.npz $OUT/aae_seed44/test_outputs.npz --out analysis/ensemble.json
# 4. studies that retrain models
python ablation_study.py --data-dir "$DATA" --out-dir $OUT
python attention_placement_ablation.py --data-dir "$DATA" --out-dir $OUT
python hyperparameter_sensitivity.py --data-dir "$DATA" --out-dir $OUT
python learning_curve.py --data-dir "$DATA" --out-dir $OUT
python input_resolution.py --data-dir "$DATA" --out-dir $OUT
python loco_generalization.py --data-dir "$DATA" --out-dir $OUT
python cross_validation_significance.py --data-dir "$DATA" --out-dir runs_cv
