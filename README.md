# AAE-CNN — code for all algorithms and experiments in the article

*AAE-CNN: an adaptive dual-path attention network with curriculum-weighted loss for robust crop leaf disease identification*

PyTorch implementation of the proposed AAE-CNN, the seven baselines, and every experiment reported in the manuscript. Each algorithm is in its own Python file. Shared code (data split, transforms, training loop, metrics) is in `common.py`, which every script imports.

## 1. Installation

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # for a CUDA GPU, install torch/torchvision from pytorch.org first
```

Python ≥ 3.9 is required. On the first run the ImageNet weights for each backbone are downloaded automatically by torchvision. This needs internet access; use `--no-pretrained` to train from scratch instead.

## 2. Data

Download the colour images of PlantVillage (https://github.com/spMohanty/PlantVillage-Dataset, folder `raw/color`). They are already arranged as one folder per class:

```
PlantVillage/
   Apple___Apple_scab/ *.jpg
   Apple___Black_rot/  *.jpg
   ...
   Tomato___healthy/   *.jpg
```

Every training script creates the same stratified 80 / 10 / 10 split (seed 42) and saves it to `runs/split_seed42.json`. All later scripts reuse that file, so every model is trained and tested on exactly the same images.

## 3. Quick check (≈ 3 minutes, CPU)

```bash
bash smoke_test.sh        # builds a tiny synthetic dataset and runs every model for 1 epoch
```

The numbers from the smoke test are meaningless. It only confirms that everything runs.

## 4. Files

### Proposed method
| File | Algorithm |
|---|---|
| `aae_cnn.py` | **AAE-CNN**: DenseNet-121 backbone with ADPA modules after Dense Blocks 1–3. Each ADPA combines an SE branch (r = 16) and a 7×7 spatial-attention branch using learned gates: F′ = σ(α)F_SE + σ(β)F_SA + F, with α = β = 0.5 at initialisation. An auxiliary head after Dense Block 2 is trained with the curriculum-weighted loss L = CE + λ(t)·WCE, where w_c = N/(C·n_c) and λ(t) = min(1, 2t/T). Training follows Algorithm 1: Adam 3e-4, weight decay 1e-4, batch 32, 100 epochs, cosine annealing, FP16. |

### Baselines (identical training protocol)
| File | Model |
|---|---|
| `resnet50.py` | ResNet-50 |
| `densenet121.py` | DenseNet-121 (backbone without ADPA) |
| `mobilenetv2.py` | MobileNetV2 |
| `efficientnet_b4.py` | EfficientNet-B4 |
| `se_resnet50.py` | SE-ResNet-50 (SE module in every bottleneck) |
| `cbam_resnet50.py` | CBAM-ResNet-50 (channel then spatial attention in every bottleneck) |
| `vit_b16.py` | Vision Transformer ViT-B/16 |

### Experiments and analyses
| File | Manuscript item |
|---|---|
| `dataset_statistics.py` | class table, class-distribution histogram, imbalance ratio |
| `augmentation_visualization.py` | augmentation illustration figure |
| `per_class_analysis.py` | overall results table, accuracy bar chart, per-class P/R/F1 table, per-crop macro F1, per-class F1 figure |
| `ablation_study.py` | component ablation (7 configurations) |
| `attention_placement_ablation.py` | ADPA placement (early / late / all-layer / per-block) |
| `hyperparameter_sensitivity.py` | sensitivity to r, η₀ and curriculum rate |
| `plot_training_curves.py` | training / validation accuracy curves |
| `learning_curve.py` | accuracy vs. training-data fraction |
| `roc_analysis.py` | macro AUC table and ROC curves |
| `robustness_corruption.py` | accuracy under 5 synthetic corruptions × 3 severities |
| `deployment_benchmark.py` | parameters, MACs, GPU/CPU latency, FP16, ONNX export, INT8 quantisation (for edge devices) |
| `loco_generalization.py` | leave-one-crop-out test with a linear probe (20 support images per class) |
| `input_resolution.py` | accuracy and latency at 128/160/224/320 px |
| `ensemble.py` | deep ensemble of 3 seeds |
| `error_severity.py` | Tier 1/2/3 error analysis and crop-level confusion matrices |
| `cross_validation_significance.py` | 5-fold stratified CV, paired t-test and one-sided Wilcoxon test |
| `gradcam.py` | Grad-CAM activation maps |
| `make_dummy_dataset.py` | tiny synthetic dataset for the smoke test |
| `run_all.sh` / `run_all.bat` | full pipeline in the order of the manuscript |

## 5. Typical use

```bash
# train the proposed model (outputs in runs/aae_cnn/)
python aae_cnn.py --data-dir /data/PlantVillage --out-dir runs

# train a baseline
python cbam_resnet50.py --data-dir /data/PlantVillage --out-dir runs

# full pipeline
bash run_all.sh /data/PlantVillage          # Windows: run_all.bat D:\data\PlantVillage
```

Each training run writes `best.pt` (checkpoint), `history.csv` (per-epoch curves), `test_metrics.json` and `test_outputs.npz` (test labels and softmax probabilities; the analysis scripts read these). AAE-CNN also writes `gate_values.json` (the learned σ(α), σ(β) of each ADPA module).

Options common to all training scripts: `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--img-size`, `--seed`, `--workers`, `--device`, `--no-pretrained`, `--no-amp`, `--no-augment`, `--split-file`.

AAE-CNN only: `--adpa-mode {adaptive,fixed,se,sa,none}`, `--placement {per_block,early,late,all_layer,none}`, `--reduction`, `--no-aux`, `--loss-schedule {curriculum,constant,none}`, `--curriculum-rate`, `--run-name`.

On Windows, add `--workers 0` if data loading raises a multiprocessing error.

## 6. Notes

* **Parameters.** At inference the auxiliary head is not used. `deployment_benchmark.py` removes it before counting, which gives about 8.2 M parameters for 38 classes; the training log counts the auxiliary head as well. MACs (Conv2d + Linear + attention) are counted with built-in hooks, so no extra package is needed.
* **Raspberry Pi / mobile.** `deployment_benchmark.py` exports `<model>_fp32.onnx` and a statically quantised `<model>_int8.onnx` (calibrated on validation images). Copy the INT8 file to the device and time it there with `python deployment_benchmark.py --onnx-only aae_cnn_int8.onnx` (the device only needs `numpy` and `onnxruntime`).
* **All-layer placement.** This puts one ADPA module on the new feature maps of every dense layer in Dense Blocks 1–3, which is 42 modules; the script prints the actual number.
* **Grad-CAM** uses the last convolutional stage of the network (for AAE-CNN, the output of Dense Block 4). It is not defined for ViT.
* **Reproducibility.** Seeds are fixed, but cuDNN kernels are non-deterministic, so results vary slightly from run to run on a GPU.
