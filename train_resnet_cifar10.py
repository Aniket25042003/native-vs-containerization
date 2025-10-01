"""
train_resnet_cifar10.py

Fine-tune ResNet-18 on CIFAR-10 and log metrics for experimental comparison
(native vs Docker vs Kubernetes).

Primary in-script metrics logged:
 - step_time (s) and epoch_time (s)
 - throughput (images/sec)
 - train_loss
 - data_load_time per step (s)
 - CPU% (psutil), host RAM% (psutil)
 - GPU util% and GPU memory used (MiB) via pynvml

Outputs:
 - logs/step_metrics_{timestamp}.csv  (time-series, per-step)
 - logs/epoch_metrics_{timestamp}.csv (per-epoch summary)
 - logs/metadata_{timestamp}.json
 - stdout prints and light console progress

Usage (example):
    python train_resnet_cifar10.py --epochs 5 --batch-size 128 --num-workers 4 --outdir ./logs
"""

import argparse
import time
import csv
import os
import json
import subprocess
from datetime import datetime
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
# Monitoring libraries
try:
    import psutil
except Exception:
    psutil = None

try:
    import pynvml
except Exception:
    pynvml = None

# -----------------------
# Utility functions
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_gpu_monitoring():
    """
    Initialize NVML (for GPU monitoring). Safe if not installed.
    """
    if pynvml is None:
        print("[WARN] pynvml not installed. GPU metrics will be disabled.")
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception as e:
        print(f"[WARN] pynvml.nvmlInit() failed: {e}. GPU metrics disabled.")
        return False

def get_gpu_stats(handle_idx=0):
    """Return (gpu_util_percent, gpu_mem_MiB) or (None, None) if not available."""
    if pynvml is None:
        return None, None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(handle_idx)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(util.gpu), int(meminfo.used / 1024 / 1024)
    except Exception:
        return None, None

def get_cpu_mem():
    """Return (cpu_percent, ram_percent) or (None, None) if psutil not available."""
    if psutil is None:
        return None, None
    try:
        cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        return cpu, vm.percent
    except Exception:
        return None, None

# -----------------------
# Subprocess collectors
# -----------------------
def query_nvidia_smi():
    """Query GPU util/mem via subprocess nvidia-smi (alternative to pynvml)."""
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"
        ])
        gpu_util, gpu_mem = out.decode().strip().split(',')
        return int(gpu_util), int(gpu_mem)
    except Exception:
        return None, None

def query_docker_stats(container_name=None):
    """Query docker stats for a container (default: first listed)."""
    try:
        if container_name:
            out = subprocess.check_output([
                "docker", "stats", container_name,
                "--no-stream", "--format",
                "{{.CPUPerc}},{{.MemUsage}},{{.BlockIO}}"
            ])
        else:
            out = subprocess.check_output([
                "docker", "stats", "--no-stream", "--format",
                "{{.CPUPerc}},{{.MemUsage}},{{.BlockIO}}"
            ])
        cpu_perc, mem_usage, block_io = out.decode().strip().split(',')
        return cpu_perc.strip(), mem_usage.strip(), block_io.strip()
    except Exception:
        return None, None, None

def query_k8s_pod(pod_name=None, namespace="default"):
    """Query kubectl top pods (returns first pod if not specified)."""
    try:
        if pod_name:
            out = subprocess.check_output([
                "kubectl", "top", "pod", pod_name,
                "-n", namespace, "--no-headers"
            ])
            fields = out.decode().strip().split()
            return fields[1], fields[2]  # CPU, MEM
        else:
            out = subprocess.check_output([
                "kubectl", "top", "pods", "-n", namespace, "--no-headers"
            ])
            first_line = out.decode().strip().split("\n")[0]
            fields = first_line.split()
            return fields[1], fields[2]  # CPU, MEM
    except Exception:
        return None, None

def query_iostat():
    """Query iostat for disk read/write MB/s (last device line)."""
    try:
        out = subprocess.check_output(["iostat", "-dx", "1", "2"])
        lines = out.decode().strip().split("\n")
        last_line = lines[-1].split()
        if len(last_line) >= 4:
            device, tps, read_kB_s, write_kB_s = last_line[0:4]
            return float(read_kB_s)/1024, float(write_kB_s)/1024
        return None, None
    except Exception:
        return None, None
    
# -----------------------
# Training code
# -----------------------
def build_dataloaders(batch_size=128, num_workers=4):
    """
    Build CIFAR-10 train and validation dataloaders.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers, pin_memory=True)

    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return trainloader, valloader

def build_model(num_classes=10, weights=None):
    """
    Build ResNet-18 and adapt final FC layer.
    """
    model = torchvision.models.resnet18(weights=weights)
    # Modify last layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def train_and_log(args):
    # Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu_monitor = init_gpu_monitoring() and torch.cuda.is_available()
    print("Device:", device, "GPU monitoring:", use_gpu_monitor)

    trainloader, valloader = build_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
    model = build_model(num_classes=10, weights=None).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Prepare output files & metadata
    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    step_csv_path = os.path.join(args.outdir, f"step_metrics_{ts}.csv")
    epoch_csv_path = os.path.join(args.outdir, f"epoch_metrics_{ts}.csv")
    meta_json_path = os.path.join(args.outdir, f"metadata_{ts}.json")

    metadata = {
        "timestamp": ts,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "optimizer": "SGD",
        "lr": args.lr,
    }
    with open(meta_json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Open CSV writers
    step_csv_file = open(step_csv_path, "w", newline="")
    epoch_csv_file = open(epoch_csv_path, "w", newline="")
    step_writer = csv.writer(step_csv_file)
    epoch_writer = csv.writer(epoch_csv_file)

    # Write headers
    step_writer.writerow([
        "timestamp",
        "run_id",
        "epoch_idx",
        "step_idx",
        "batch_size",
        "data_load_time_s",
        "forward_backward_time_s",
        "step_time_s",
        "throughput_samples_per_s",
        "train_loss",
        "gpu_util_pct",
        "gpu_mem_MiB",
        "cpu_percent",
        "host_ram_percent"
    ])
    epoch_writer.writerow([
        "timestamp",
        "run_id",
        "epoch_idx",
        "epoch_time_s",
        "throughput_samples_per_s",
        "train_loss_epoch",
        "val_loss_epoch",
        "val_acc_epoch",
        "avg_gpu_util_pct",
        "max_gpu_mem_MiB",
        "avg_cpu_percent",
        "avg_host_ram_percent",
        "docker_cpu",
        "docker_mem",
        "docker_io",
        "k8s_cpu",
        "k8s_mem",
        "disk_read_MBps",
        "disk_write_MBps",
        "smi_util",
        "smi_mem"
    ])

    run_id = ts
    global_start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.perf_counter()

        # Per-epoch accumulators
        epoch_samples = 0
        epoch_loss_sum = 0.0
        gpu_util_samples = []
        gpu_mem_samples = []
        cpu_samples = []
        ram_samples = []

        # Warm-up note:
        if epoch == 0 and args.warmup_epochs > 0:
            print(f"[INFO] Running {args.warmup_epochs} warm-up epochs (not measured) ...")
            for _ in range(args.warmup_epochs):
                for _i, (inputs, targets) in enumerate(trainloader):
                    pass
            print("[INFO] Warm-up done. Starting measured epochs ...")

        for step_idx, (inputs, targets) in enumerate(trainloader):
            step_start = time.perf_counter()

            # Data loading time can be measured by considering time between iterations
            # but best approach: measure before and after moving to device
            data_to_device_start = time.perf_counter()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            data_to_device_end = time.perf_counter()
            data_load_time = data_to_device_end - data_to_device_start

            # Forward + backward
            compute_start = time.perf_counter()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            compute_end = time.perf_counter()
            forward_backward_time = compute_end - compute_start

            step_end = time.perf_counter()
            step_time = step_end - step_start

            batch_size_actual = inputs.size(0)
            throughput = batch_size_actual / step_time if step_time > 0 else None

            # collect system metrics
            if use_gpu_monitor:
                gpu_util, gpu_mem = get_gpu_stats(handle_idx=0)
            else:
                gpu_util, gpu_mem = None, None

            cpu_pct, ram_pct = get_cpu_mem()

            # accumulate epoch values
            epoch_samples += batch_size_actual
            epoch_loss_sum += loss.item() * batch_size_actual
            if gpu_util is not None:
                gpu_util_samples.append(gpu_util)
            if gpu_mem is not None:
                gpu_mem_samples.append(gpu_mem)
            if cpu_pct is not None:
                cpu_samples.append(cpu_pct)
            if ram_pct is not None:
                ram_samples.append(ram_pct)

            # write per-step row
            step_writer.writerow([
                datetime.utcnow().isoformat(),
                run_id,
                epoch,
                step_idx,
                batch_size_actual,
                f"{data_load_time:.6f}",
                f"{forward_backward_time:.6f}",
                f"{step_time:.6f}",
                f"{throughput:.3f}" if throughput is not None else "",
                f"{loss.item():.6f}",
                "" if gpu_util is None else gpu_util,
                "" if gpu_mem is None else gpu_mem,
                "" if cpu_pct is None else f"{cpu_pct:.2f}",
                "" if ram_pct is None else f"{ram_pct:.2f}"
            ])
            # flush occasionally to avoid data loss
            if step_idx % 50 == 0:
                step_csv_file.flush()

            # basic console progress
            if step_idx % args.print_freq == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step_idx}/{len(trainloader)}] "
                      f"step_time={step_time:.3f}s throughput={throughput:.1f} loss={loss.item():.4f}")

            # optional quick exit for debugging
            if args.debug and step_idx >= args.debug_steps:
                break

        # epoch end: compute validation metrics
        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        avg_throughput = epoch_samples / epoch_time if epoch_time > 0 else None
        avg_loss = epoch_loss_sum / epoch_samples if epoch_samples > 0 else None

        # validation
        val_loss, val_acc = evaluate(model, valloader, device, criterion)

        # aggregate epoch-level metrics
        avg_gpu_util = float(np.mean(gpu_util_samples)) if len(gpu_util_samples) else None
        max_gpu_mem = int(max(gpu_mem_samples)) if len(gpu_mem_samples) else None
        avg_cpu = float(np.mean(cpu_samples)) if len(cpu_samples) else None
        avg_ram = float(np.mean(ram_samples)) if len(ram_samples) else None
        
        # subprocess-based collectors (system-level, once per epoch)
        docker_cpu, docker_mem, docker_io = query_docker_stats()
        k8s_cpu, k8s_mem = query_k8s_pod()
        disk_read_MBps, disk_write_MBps = query_iostat()
        smi_util, smi_mem = query_nvidia_smi()

        # Write per-epoch row
        epoch_writer.writerow([
            datetime.utcnow().isoformat(),
            run_id,
            epoch,
            f"{epoch_time:.6f}",
            f"{avg_throughput:.3f}" if avg_throughput else "",
            f"{avg_loss:.6f}" if avg_loss else "",
            f"{val_loss:.6f}",
            f"{val_acc:.4f}",
            "" if avg_gpu_util is None else f"{avg_gpu_util:.2f}",
            "" if max_gpu_mem is None else f"{max_gpu_mem}",
            "" if avg_cpu is None else f"{avg_cpu:.2f}",
            "" if avg_ram is None else f"{avg_ram:.2f}",
            docker_cpu or "",
            docker_mem or "",
            docker_io or "",
            k8s_cpu or "",
            k8s_mem or "",
            f"{disk_read_MBps:.3f}" if disk_read_MBps else "",
            f"{disk_write_MBps:.3f}" if disk_write_MBps else "",
            smi_util or "",
            smi_mem or ""
        ])
        epoch_csv_file.flush()

        scheduler.step()

    total_time = time.perf_counter() - global_start
    print(f"Training completed in {total_time:.2f}s. Logs saved to: {args.outdir}")
    step_csv_file.close()
    epoch_csv_file.close()
    
    # -----------------------
    # Save models
    # -----------------------
    model.eval()
    # Save state_dict (reproducibility, retraining possible)
    state_dict_path = os.path.join(args.outdir, f"resnet18_cifar10_state_{ts}.pth")
    torch.save(model.state_dict(), state_dict_path)
    print(f"[INFO] Saved PyTorch state_dict to {state_dict_path}")
    
    # Save TorchScript model (for inference benchmarks & deployment)
    example_input = torch.randn(1, 3, 32, 32).to(device)
    scripted_model = torch.jit.trace(model, example_input)
    torchscript_path = os.path.join(args.outdir, f"resnet18_cifar10_scripted_{ts}.pt")
    scripted_model.save(torchscript_path)
    print(f"[INFO] Saved TorchScript model to {torchscript_path}")

    if use_gpu_monitor:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def evaluate(model, valloader, device, criterion):
    """
    Simple validation loop returning (val_loss, val_acc)
    """
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in valloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_sum += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)
    val_loss = loss_sum / total if total > 0 else 0.0
    val_acc = correct / total if total > 0 else 0.0
    return val_loss, val_acc

# -----------------------
# Argument parser
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-18 on CIFAR-10 with metric logging")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Warm-up epochs (not measured)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="./logs", help="Directory to store logs")
    parser.add_argument("--print-freq", type=int, default=200, help="Steps between console prints")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (exit early)")
    parser.add_argument("--debug-steps", type=int, default=10, help="Steps in debug mode")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_and_log(args)
