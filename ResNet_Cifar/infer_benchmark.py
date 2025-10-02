"""
infer_benchmark.py

Benchmark inference latency, throughput, and system metrics for a trained model.

Metrics logged:
- Cold-start latency (ms): load + first inference
- Model load time (s)
- Warm latency: P50, P95, P99, std deviation
- Throughput (samples/sec) at batch sizes [1, 8, 32, 128]
- GPU utilization (%), GPU memory (MiB)
- CPU% and host RAM%
- Latency jitter (via per-invocation CSV)
- Logs saved to CSV and JSON

Usage:
    python infer_benchmark.py --model-path ./logs/resnet18_cifar10.pt --outdir ./logs/infer --batch-sizes 1 8 32 128
"""

import argparse
import time
import os
import json
import csv
from datetime import datetime, timezone
import numpy as np
import torch
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False
    print("[WARN] pynvml not available, GPU metrics disabled.")


# ----------------------
# GPU/CPU Monitoring
# ----------------------
def get_gpu_stats(device_index=0):
    if not NVML_AVAILABLE:
        return None, None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(util.gpu), int(meminfo.used / 1024 / 1024)
    except Exception:
        return None, None

def get_cpu_mem():
    try:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        return cpu, ram
    except Exception:
        return None, None


# ----------------------
# Model Loading (TorchScript not state_dict)
# ----------------------
def load_torchscript(model_path, device):
    load_start = time.perf_counter()
    model = torch.jit.load(model_path, map_location=device)
    load_end = time.perf_counter()
    return model, load_end - load_start


# ----------------------
# Inference Benchmark
# ----------------------
def benchmark_inference(model, device, batch_size, num_iters=200, warmup=20):
    model.eval()
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

    latencies = []
    gpu_utils, gpu_mems, cpu_utils, ram_utils = [], [], [], []

    for _ in range(num_iters):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        lat_ms = (end - start) * 1000
        latencies.append(lat_ms)

        # System stats
        gpu_util, gpu_mem = get_gpu_stats()
        cpu_pct, ram_pct = get_cpu_mem()
        if gpu_util is not None: gpu_utils.append(gpu_util)
        if gpu_mem is not None: gpu_mems.append(gpu_mem)
        if cpu_pct is not None: cpu_utils.append(cpu_pct)
        if ram_pct is not None: ram_utils.append(ram_pct)

    results = {
        "batch_size": batch_size,
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "std_ms": float(np.std(latencies)),
        "throughput_samples_per_s": batch_size / (np.mean(latencies) / 1000),
        "avg_gpu_util_pct": float(np.mean(gpu_utils)) if gpu_utils else None,
        "max_gpu_mem_MiB": int(max(gpu_mems)) if gpu_mems else None,
        "avg_cpu_percent": float(np.mean(cpu_utils)) if cpu_utils else None,
        "avg_ram_percent": float(np.mean(ram_utils)) if ram_utils else None,
    }
    return results, latencies


# ----------------------
# Main
# ----------------------
def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running inference on device:", device)

    # Load model (TorchScript)
    model, model_load_time = load_torchscript(args.model_path, device)

    # Cold-start latency
    cold_input = torch.randn(1, 3, 32, 32).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(cold_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    cold_start_latency = (end - start) * 1000

    # Prepare CSVs
    metrics_csv = os.path.join(args.outdir, f"infer_metrics_{ts}.csv")
    latencies_csv = os.path.join(args.outdir, f"infer_latencies_{ts}.csv")
    summary_json = os.path.join(args.outdir, f"infer_summary_{ts}.json")

    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp","batch_size","mean_ms","p50_ms","p95_ms","p99_ms","std_ms",
            "throughput_samples_per_s","avg_gpu_util_pct","max_gpu_mem_MiB",
            "avg_cpu_percent","avg_ram_percent","cold_start_latency_ms","model_load_time_s"
        ])

        all_results = []
        for bs in args.batch_sizes:
            results, latencies = benchmark_inference(model, device, batch_size=bs,
                                                     num_iters=args.num_iters, warmup=args.warmup)
            results["timestamp"] = datetime.now(timezone.utc).isoformat()
            writer.writerow([
                results["timestamp"],results["batch_size"],results["mean_ms"],
                results["p50_ms"],results["p95_ms"],results["p99_ms"],results["std_ms"],
                results["throughput_samples_per_s"],results["avg_gpu_util_pct"] or "",
                results["max_gpu_mem_MiB"] or "",results["avg_cpu_percent"] or "",
                results["avg_ram_percent"] or "",cold_start_latency,model_load_time
            ])
            all_results.append(results)

            # Save per-invocation latencies
            with open(latencies_csv, "a", newline="") as f_lat:
                lat_writer = csv.writer(f_lat)
                if f_lat.tell() == 0:
                    lat_writer.writerow(["timestamp","batch_size","iteration","latency_ms"])
                for i, lat in enumerate(latencies):
                    lat_writer.writerow([results["timestamp"], bs, i, lat])

    # Write JSON summary
    summary = {
        "timestamp": ts,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "model_load_time_s": model_load_time,
        "cold_start_latency_ms": cold_start_latency,
        "batch_results": all_results
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Results saved to:\n- {metrics_csv}\n- {latencies_csv}\n- {summary_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to TorchScript .pt model")
    parser.add_argument("--outdir", type=str, default="./logs/infer", help="Output directory")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1,8,32,128], help="Batch sizes to test")
    parser.add_argument("--num-iters", type=int, default=200, help="Number of iterations per batch size")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations before measuring")
    args = parser.parse_args()
    main(args)
