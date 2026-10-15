@echo off
REM Full reproduction pipeline (Windows). Usage: run_all.bat D:\data\PlantVillage
set DATA=%1
if "%DATA%"=="" (echo Give the PlantVillage folder & exit /b 1)
set OUT=runs
python dataset_statistics.py --data-dir %DATA% --out-dir analysis || exit /b 1
for %%m in (aae_cnn resnet50 densenet121 mobilenetv2 efficientnet_b4 se_resnet50 cbam_resnet50 vit_b16) do (
  python %%m.py --data-dir %DATA% --out-dir %OUT% --workers 0 || exit /b 1
)
python per_class_analysis.py --npz %OUT%\aae_cnn\test_outputs.npz %OUT%\resnet50\test_outputs.npz %OUT%\densenet121\test_outputs.npz %OUT%\mobilenetv2\test_outputs.npz %OUT%\efficientnet_b4\test_outputs.npz %OUT%\se_resnet50\test_outputs.npz %OUT%\cbam_resnet50\test_outputs.npz %OUT%\vit_b16\test_outputs.npz --out-dir analysis
python roc_analysis.py --npz %OUT%\aae_cnn\test_outputs.npz %OUT%\cbam_resnet50\test_outputs.npz %OUT%\densenet121\test_outputs.npz %OUT%\vit_b16\test_outputs.npz --out analysis\roc_curves.png
python plot_training_curves.py --history %OUT%\aae_cnn\history.csv --out analysis\training_curves.png
python error_severity.py --npz %OUT%\aae_cnn\test_outputs.npz --out-dir analysis
python gradcam.py --data-dir %DATA% --checkpoint %OUT%\aae_cnn\best.pt --out analysis\gradcam.png
python robustness_corruption.py --data-dir %DATA% --workers 0 --checkpoints %OUT%\densenet121\best.pt %OUT%\cbam_resnet50\best.pt %OUT%\aae_cnn\best.pt --out analysis\robustness.csv
python deployment_benchmark.py --data-dir %DATA% --out-dir deploy --checkpoints %OUT%\aae_cnn\best.pt %OUT%\resnet50\best.pt %OUT%\densenet121\best.pt %OUT%\mobilenetv2\best.pt %OUT%\efficientnet_b4\best.pt %OUT%\cbam_resnet50\best.pt %OUT%\vit_b16\best.pt
python aae_cnn.py --data-dir %DATA% --out-dir %OUT% --workers 0 --seed 43 --run-name aae_seed43 --split-file %OUT%\split_seed42.json
python aae_cnn.py --data-dir %DATA% --out-dir %OUT% --workers 0 --seed 44 --run-name aae_seed44 --split-file %OUT%\split_seed42.json
python ensemble.py --npz %OUT%\aae_cnn\test_outputs.npz %OUT%\aae_seed43\test_outputs.npz %OUT%\aae_seed44\test_outputs.npz --out analysis\ensemble.json
python ablation_study.py --data-dir %DATA% --out-dir %OUT% --workers 0
python attention_placement_ablation.py --data-dir %DATA% --out-dir %OUT% --workers 0
python hyperparameter_sensitivity.py --data-dir %DATA% --out-dir %OUT% --workers 0
python learning_curve.py --data-dir %DATA% --out-dir %OUT% --workers 0
python input_resolution.py --data-dir %DATA% --out-dir %OUT% --workers 0
python loco_generalization.py --data-dir %DATA% --out-dir %OUT% --workers 0
python cross_validation_significance.py --data-dir %DATA% --out-dir runs_cv --workers 0
