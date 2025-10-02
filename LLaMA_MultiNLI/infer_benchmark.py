#!/usr/bin/env python3
"""
infer_benchmark.py

Benchmark inference performance of fine-tuned LLaMA-3.2 3B model.
Collects latency distributions, throughput, GPU/CPU utilization, memory usage,
cold-start latency, model load time, etc. Matches Phase 1 output format.

Outputs:
 - infer_metrics_<timestamp>.csv
 - infer_latencies_<timestamp>.csv
 - infer_summary_<timestamp>.json
"""

import argparse
import os
import time
import csv
import json
import statistics
from datetime import datetime, timezone

import torch
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_cpu_mem():
    if psutil is None:
        return None, None
    try:
        return psutil.cpu_percent(interval=None), psutil.virtual_memory().percent
    except Exception:
        return None, None


def get_gpu_stats(handle_idx=0):
    if not NVML_AVAILABLE:
        return None, None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(handle_idx)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(util.gpu), int(mem.used / 1024 / 1024)
    except Exception:
        return None, None


def load_model(model_path: str, device: str):
    """Load TorchScript if available, else HF AutoModel with PEFT adapter."""
    start = time.perf_counter()
    if model_path.endswith(".pt"):
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        from peft import PeftModel

        print("[INFO] Loading HF model + PEFT adapter...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        if os.path.isdir(os.path.join(model_path, "adapter_model")):
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = base_model
        model.eval()
    load_time = time.perf_counter() - start
    return model, load_time


def benchmark_inference(model, device, batch_sizes, num_iters, warmup, max_length=128, outdir="./infer"):
    os.makedirs(outdir, exist_ok=True)
    ts = now_ts()
    run_id = ts

    # Prepare output files
    metrics_csv = os.path.join(outdir, f"infer_metrics_{ts}.csv")
    latencies_csv = os.path.join(outdir, f"infer_latencies_{ts}.csv")
    summary_json = os.path.join(outdir, f"infer_summary_{ts}.json")

    # Writers
    metrics_f = open(metrics_csv, "w", newline="")
    latencies_f = open(latencies_csv, "w", newline="")
    metrics_writer = csv.writer(metrics_f)
    latencies_writer = csv.writer(latencies_f)

    metrics_writer.writerow([
        "timestamp","batch_size","mean_ms","p50_ms","p95_ms","p99_ms","std_ms",
        "throughput_samples_per_s","avg_gpu_util_pct","max_gpu_mem_MiB",
        "avg_cpu_percent","avg_ram_percent","cold_start_latency_ms","model_load_time_s"
    ])
    latencies_writer.writerow(["timestamp","batch_size","iteration","latency_ms"])

    results_summary = {
        "timestamp": iso_now(),
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "batch_results": []
    }

    # Generate dummy input
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", use_fast=True)
    except Exception:
        pass

    def make_input(batch_size):
        prompt = "premise: A man inspects the uniform of a figure in some East Asian country.\nhypothesis: The man is sleeping.\nLabel:"
        if tokenizer:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"]
        else:
            input_ids = torch.randint(0, 1000, (1, max_length))
        return input_ids.repeat(batch_size, 1).to(device)

    # Cold start latency: first forward
    start = time.perf_counter()
    _ = model(make_input(1))
    cold_latency = (time.perf_counter() - start) * 1000.0

    # Loop over batch sizes
    for bsz in batch_sizes:
        latencies = []
        gpu_utils, gpu_mems, cpu_utils, ram_utils = [], [], [], []

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(make_input(bsz))

        # Timed iterations
        for i in range(num_iters):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(make_input(bsz))
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            lat = (end - start) * 1000.0
            latencies.append(lat)
            latencies_writer.writerow([iso_now(), bsz, i, f"{lat:.6f}"])

            gpu_util, gpu_mem = get_gpu_stats()
            cpu, ram = get_cpu_mem()
            if gpu_util is not None: gpu_utils.append(gpu_util)
            if gpu_mem is not None: gpu_mems.append(gpu_mem)
            if cpu is not None: cpu_utils.append(cpu)
            if ram is not None: ram_utils.append(ram)

        # Aggregate stats
        mean_ms = statistics.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        std_ms = statistics.pstdev(latencies)
        throughput = (bsz * num_iters) / (sum(lat / 1000 for lat in latencies))

        avg_gpu_util = float(np.mean(gpu_utils)) if gpu_utils else None
        max_gpu_mem = max(gpu_mems) if gpu_mems else None
        avg_cpu = float(np.mean(cpu_utils)) if cpu_utils else None
        avg_ram = float(np.mean(ram_utils)) if ram_utils else None

        metrics_writer.writerow([
            iso_now(), bsz,
            f"{mean_ms:.6f}", f"{p50:.6f}", f"{p95:.6f}", f"{p99:.6f}", f"{std_ms:.6f}",
            f"{throughput:.3f}",
            "" if avg_gpu_util is None else f"{avg_gpu_util:.2f}",
            "" if max_gpu_mem is None else f"{max_gpu_mem}",
            "" if avg_cpu is None else f"{avg_cpu:.2f}",
            "" if avg_ram is None else f"{avg_ram:.2f}",
            f"{cold_latency:.3f}", f"{load_time:.3f}"
        ])

        results_summary["batch_results"].append({
            "batch_size": bsz,
            "mean_ms": mean_ms,
            "p50_ms": float(p50),
            "p95_ms": float(p95),
            "p99_ms": float(p99),
            "std_ms": std_ms,
            "throughput_samples_per_s": throughput,
            "avg_gpu_util_pct": avg_gpu_util,
            "max_gpu_mem_MiB": max_gpu_mem,
            "avg_cpu_percent": avg_cpu,
            "avg_ram_percent": avg_ram,
            "timestamp": iso_now()
        })

    metrics_f.close()
    latencies_f.close()

    with open(summary_json, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"[DONE] Results saved to:\n- {metrics_csv}\n- {latencies_csv}\n- {summary_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference benchmark for LLaMA 3.2 3B fine-tuned model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to TorchScript .pt or HuggingFace model dir")
    parser.add_argument("--outdir", type=str, default="./infer")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running inference on device:", device)

    model, load_time = load_model(args.model_path, device)
    benchmark_inference(model, device, args.batch_sizes, args.num_iters, args.warmup, args.max_length, args.outdir)

    if NVML_AVAILABLE:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
