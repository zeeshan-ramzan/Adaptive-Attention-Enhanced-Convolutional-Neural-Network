"""
deployment_benchmark.py  --  Efficiency and deployment benchmark
===============================================================
(paper Tables "Computational efficiency comparison" and "Inference latency and memory
footprint across deployment platforms")

For every checkpoint it reports
  * parameters at inference (the AAE-CNN auxiliary head is removed at inference),
  * MACs for one 224x224 image,
  * latency per image: PyTorch FP32 (GPU if available, else CPU), PyTorch FP16 (GPU),
  * ONNX export and ONNX Runtime FP32 latency on the CPU,
  * ONNX Runtime static INT8 post-training quantisation (calibrated on validation images):
    latency, model file size and test accuracy (if --data-dir is given),
  * peak GPU memory during inference.

To reproduce the edge (Raspberry Pi 4B) and mobile rows, copy the exported
`<name>_int8.onnx` / `<name>_fp32.onnx` to the device and run
  python deployment_benchmark.py --onnx-only model_int8.onnx
there (only numpy + onnxruntime are needed on the device).

Usage:
  python deployment_benchmark.py --checkpoints runs/aae_cnn/best.pt runs/densenet121/best.pt \
         --data-dir /path/PlantVillage --out-dir deploy
"""
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import common


def onnx_latency(path, img_size, runs=50, warmup=10, threads=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {name: x})
    return (time.perf_counter() - t0) / runs * 1000.0, sess


def onnx_accuracy(sess, loader):
    name = sess.get_inputs()[0].name
    correct = seen = 0
    for x, y in loader:
        for i in range(x.shape[0]):
            out = sess.run(None, {name: x[i:i + 1].numpy()})[0]
            correct += int(out.argmax(1)[0] == y[i].item())
            seen += 1
    return correct / max(seen, 1)


class CalibReader:
    """Feeds validation images to the ONNX Runtime static quantiser."""

    def __init__(self, loader, input_name, max_batches=20):
        self.it = iter(loader)
        self.name, self.left = input_name, max_batches

    def get_next(self):
        if self.left <= 0:
            return None
        try:
            x, _ = next(self.it)
        except StopIteration:
            return None
        self.left -= 1
        return {self.name: x.numpy()}


def main():
    p = argparse.ArgumentParser(description="Efficiency / deployment benchmark")
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--onnx-only", default=None, help="only time an existing ONNX file (e.g. on a Raspberry Pi)")
    p.add_argument("--data-dir", default=None, help="needed for INT8 calibration and accuracy")
    p.add_argument("--split-file", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--out-dir", default="deploy")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.onnx_only:
        ms, _ = onnx_latency(args.onnx_only, args.img_size, args.runs, threads=args.threads)
        print(f"{args.onnx_only}: {ms:.1f} ms / image, file {os.path.getsize(args.onnx_only) / 2**20:.1f} MB")
        return

    device = common.get_device()
    rows = []
    for ck in args.checkpoints:
        model, meta = common.load_checkpoint(ck, device)
        name = meta["model_name"]
        size = meta.get("img_size", args.img_size)
        if getattr(model, "aux_head", None) is not None:
            model.aux_head = None                      # removed at inference
        row = {"model": name, "params_M": round(common.count_params(model) / 1e6, 2),
               "MACs_G": round(common.count_macs(model, size, device) / 1e9, 2)}
        row[f"latency_{device.type}_fp32_ms"] = round(common.measure_latency(model, size, device, args.runs), 2)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            row["latency_gpu_fp16_ms"] = round(common.measure_latency(
                copy_model(model), size, device, args.runs, half=True), 2)
            with torch.no_grad():
                model(torch.randn(1, 3, size, size, device=device))
            row["peak_gpu_mem_MB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)

        # ---- ONNX export + ONNX Runtime FP32 / INT8 on CPU
        try:
            import onnxruntime  # noqa: F401
            from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
            from onnxruntime.quantization.shape_inference import quant_pre_process
            cpu_model = model.to("cpu").eval()
            fp32 = os.path.join(args.out_dir, f"{name}_fp32.onnx")
            torch.onnx.export(cpu_model, torch.randn(1, 3, size, size), fp32, input_names=["input"],
                              output_names=["logits"], opset_version=17, dynamic_axes={"input": {0: "batch"}},
                              dynamo=False)
            row["onnx_fp32_cpu_ms"], _ = onnx_latency(fp32, size, args.runs, threads=args.threads)
            row["onnx_fp32_MB"] = round(os.path.getsize(fp32) / 2**20, 1)
            if args.data_dir:
                split_file = args.split_file or os.path.join(os.path.dirname(os.path.dirname(ck)),
                                                             f"split_seed{args.seed}.json")
                split = common.make_split(args.data_dir, seed=args.seed, split_file=split_file)
                tf = common.build_transforms(size, train=False)
                val_dl = DataLoader(common.SubsetWithTransform(split.subset("val"), tf), batch_size=1, shuffle=True)
                test_dl = DataLoader(common.SubsetWithTransform(split.subset("test"), tf), batch_size=32)
                pre = os.path.join(args.out_dir, f"{name}_pre.onnx")
                try:
                    quant_pre_process(fp32, pre)
                except Exception:                      # e.g. transformer graphs
                    quant_pre_process(fp32, pre, skip_symbolic_shape=True)
                int8 = os.path.join(args.out_dir, f"{name}_int8.onnx")
                quantize_static(pre, int8, CalibReader(val_dl, "input", 50), quant_format=QuantFormat.QDQ,
                                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
                ms, sess8 = onnx_latency(int8, size, args.runs, threads=args.threads)
                row["onnx_int8_cpu_ms"] = round(ms, 2)
                row["onnx_int8_MB"] = round(os.path.getsize(int8) / 2**20, 1)
                _, sess32 = onnx_latency(fp32, size, 1, 1)
                row["acc_fp32_onnx"] = round(100 * onnx_accuracy(sess32, test_dl), 2)
                row["acc_int8_onnx"] = round(100 * onnx_accuracy(sess8, test_dl), 2)
            row["onnx_fp32_cpu_ms"] = round(row["onnx_fp32_cpu_ms"], 2)
        except ImportError:
            print("onnxruntime not installed: ONNX / INT8 rows skipped (pip install onnx onnxruntime)")
        except Exception as e:                         # keep benchmarking the other models
            print(f"ONNX/INT8 step failed for {name}: {e}")
            row["onnx_error"] = str(e)[:200]
        print(row)
        rows.append(row)
        common.write_csv(os.path.join(args.out_dir, "deployment_results.csv"), rows)
        model.to(device)


def copy_model(m):
    import copy
    return copy.deepcopy(m)


if __name__ == "__main__":
    main()
