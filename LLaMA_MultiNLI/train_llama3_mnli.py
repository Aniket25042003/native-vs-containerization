#!/usr/bin/env python3
"""
train_llama3_mnli.py

Fine-tune LLaMA-3.2 3B on MultiNLI using QLoRA (PEFT + bitsandbytes).
Logs the same metrics as Phase 1 (per-step CSV, per-epoch CSV, metadata JSON).

NOTE (very important):
 - Requires: transformers, datasets, peft, bitsandbytes, accelerate, torch, psutil, pynvml
 - TorchScript saving of large models may fail. We attempt it and fall back to saving state_dict + peft adapter.
"""

import argparse
import os
import time
import csv
import json
import subprocess
import random
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, get_scheduler
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel, PeftConfig
import numpy as np

# monitoring libs
try:
    import psutil
except Exception:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

# -----------------------
# Utility & monitoring
# -----------------------
def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_cpu_mem():
    if psutil is None:
        return None, None
    try:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        return cpu, ram
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

# subprocess collectors (best-effort)
def query_nvidia_smi():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"
        ], stderr=subprocess.DEVNULL)
        gpu_util, gpu_mem = out.decode().strip().split(',')
        return gpu_util.strip(), gpu_mem.strip()
    except Exception:
        return None, None

def query_docker_stats(container_name=None):
    try:
        if container_name:
            out = subprocess.check_output([
                "docker", "stats", container_name,
                "--no-stream", "--format", "{{.CPUPerc}},{{.MemUsage}},{{.BlockIO}}"
            ], stderr=subprocess.DEVNULL)
        else:
            out = subprocess.check_output([
                "docker", "stats", "--no-stream", "--format", "{{.CPUPerc}},{{.MemUsage}},{{.BlockIO}}"
            ], stderr=subprocess.DEVNULL)
        cpu_perc, mem_usage, block_io = out.decode().strip().split(',')
        return cpu_perc.strip(), mem_usage.strip(), block_io.strip()
    except Exception:
        return None, None, None

def query_k8s_pod(pod_name=None, namespace="default"):
    try:
        if pod_name:
            out = subprocess.check_output(["kubectl","top","pod",pod_name,"-n",namespace,"--no-headers"], stderr=subprocess.DEVNULL)
            fields = out.decode().strip().split()
            return fields[1], fields[2]
        else:
            out = subprocess.check_output(["kubectl","top","pods","-n",namespace,"--no-headers"], stderr=subprocess.DEVNULL)
            first_line = out.decode().strip().split("\n")[0]
            fields = first_line.split()
            return fields[1], fields[2]
    except Exception:
        return None, None

def query_iostat():
    try:
        out = subprocess.check_output(["iostat","-dx","1","2"], stderr=subprocess.DEVNULL)
        lines = out.decode().strip().split("\n")
        last_line = lines[-1].split()
        if len(last_line) >= 4:
            device, tps, read_kB_s, write_kB_s = last_line[:4]
            return float(read_kB_s)/1024, float(write_kB_s)/1024
        return None, None
    except Exception:
        return None, None

# -----------------------
# Dataset / tokenization utilities
# -----------------------
LABELS = ["entailment", "neutral", "contradiction"]

def prompt_from_example(example: Dict, label_as_text: bool=True) -> Tuple[str,int]:
    """
    Convert a MultiNLI example into a prompt string and target label token id(s).
    We will produce prompts like:
      "premise: {premise}\nhypothesis: {hypothesis}\nLabel:"
    and target text e.g. "entailment"
    """
    premise = example.get("premise","")
    hypothesis = example.get("hypothesis","")
    prompt = f"premise: {premise}\nhypothesis: {hypothesis}\nLabel:"
    return prompt

def encode_example(tokenizer, example, max_length, label_map):
    prompt = prompt_from_example(example)
    label = example.get("label")
    # dataset label mapping: for multi_nli, label is int (0/1/2) mapping to labels maybe -1 for no label
    # in HF multi_nli, labels are 0,1,2 with mapping to entailment/neutral/contradiction depending on dataset provider;
    # We'll be robust: assume label in {0,1,2} and map by index -> LABELS
    if label is None:
        target_label = ""
    else:
        try:
            target_label = LABELS[int(label)]
        except Exception:
            # fallback: if label already string
            target_label = str(label)

    # We will create a tokenized input where we append a space + target_label and compute LM loss on the label tokens.
    # input_ids = tokens for prompt + label_tokens
    prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length, padding=False, return_tensors=None)
    label_enc = tokenizer(" " + target_label, truncation=True, max_length=16, padding=False, return_tensors=None)
    # We'll return concatenated ids and also the length of prompt tokens so we know where label starts.
    input_ids = prompt_enc["input_ids"] + label_enc["input_ids"]
    attention_mask = prompt_enc["attention_mask"] + label_enc["attention_mask"]
    label_token_ids = label_enc["input_ids"]
    # position where label tokens start:
    label_start = len(prompt_enc["input_ids"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_start": label_start,
        "label_token_ids": label_token_ids
    }

def collate_fn(batch, pad_token_id, max_len=None):
    # batch: list of dicts from encode_example
    input_ids = [b["input_ids"] for b in batch]
    attention_masks = [b["attention_mask"] for b in batch]
    label_starts = [b["label_start"] for b in batch]
    label_token_ids = [b["label_token_ids"] for b in batch]
    maxlen = max(len(x) for x in input_ids)
    if max_len:
        maxlen = min(maxlen, max_len)
    padded_input_ids = []
    padded_attn = []
    label_positions = []
    for i, ids in enumerate(input_ids):
        pad_len = maxlen - len(ids)
        padded = ids + [pad_token_id] * pad_len
        attn = attention_masks[i] + [0] * pad_len
        padded_input_ids.append(padded)
        padded_attn.append(attn)
        label_positions.append((label_starts[i], label_token_ids[i]))
    batch_input = {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attn, dtype=torch.long),
        "label_positions": label_positions
    }
    return batch_input

# -----------------------
# Model save helpers
# -----------------------
def save_peft_and_state(output_dir: str, base_model, peft_model: PeftModel, run_id: str):
    os.makedirs(output_dir, exist_ok=True)
    # Save adapter (peft) checkpoint
    adapter_dir = os.path.join(output_dir, f"peft_adapter_{run_id}")
    peft_model.save_pretrained(adapter_dir)
    # Save base model state dict (if feasible)
    try:
        sd_path = os.path.join(output_dir, f"model_state_dict_{run_id}.pth")
        torch.save(base_model.state_dict(), sd_path)
    except Exception as e:
        print(f"[WARN] Could not save base model state_dict: {e}")

def try_save_torchscript(script_save_path: str, model, device):
    """
    Attempt to convert model to TorchScript (may fail for big LLMs).
    Return bool indicating success.
    """
    try:
        model.to("cpu")
        model.eval()
        # Tracing/script for causal LM is tricky; attempt a simple tracing with sample input.
        example_input = {"input_ids": torch.tensor([[0]], dtype=torch.long)}
        scripted = torch.jit.trace(model, example_input, strict=False)
        scripted.save(script_save_path)
        return True
    except Exception as e:
        print(f"[WARN] TorchScript save failed: {e}")
        return False

# -----------------------
# Training function
# -----------------------
def train_loop(args):
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load tokenizer + dataset
    print("[INFO] Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        # ensure pad token
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("multi_nli")
    train_ds = dataset["train"]
    val_ds = dataset["validation_matched"]  # you can also evaluate on mismatched later

    # Map & tokenize: we will build a small in-memory tokenized dataset where each example stores input_ids list
    print("[INFO] Preprocessing dataset (tokenization). This may take time)...")
    def encode_fn(ex):
        return encode_example(tokenizer, ex, max_length=args.max_length, label_map=None)
    # We'll do a lightweight map with batched=False to easily keep label positions
    tokenized_train = []
    for i, ex in enumerate(train_ds):
        if args.debug and i >= args.debug_examples:
            break
        enc = encode_example(tokenizer, ex, max_length=args.max_length, label_map=None)
        tokenized_train.append(enc)
    tokenized_val = []
    for i, ex in enumerate(val_ds):
        if args.debug and i >= args.debug_examples_val:
            break
        enc = encode_example(tokenizer, ex, max_length=args.max_length, label_map=None)
        tokenized_val.append(enc)

    print(f"[INFO] Tokenized {len(tokenized_train)} train and {len(tokenized_val)} val examples (debug mode={args.debug})")

    # DataLoaders
    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(tokenized_train, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, pad_id, max_len=args.max_length),
                              num_workers=args.num_workers)
    val_loader = DataLoader(tokenized_val, batch_size=args.eval_batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, pad_id, max_len=args.max_length),
                            num_workers=args.num_workers)

    # Load base model with 4-bit weights via bitsandbytes & prepare for k-bit training
    print("[INFO] Loading base model (4-bit) - this may take time and require bitsandbytes)...")
    config = AutoConfig.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # Prepare model for k-bit training (peft helper)
    model = prepare_model_for_kbit_training(model)

    # Create LoRA config and wrap model
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(",") if args.lora_target_modules else None,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # nice summary

    # Optimizer: Only train PEFT params (LoRA)
    trainable_params = [p for n,p in model.named_parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    total_update_steps = math.ceil((len(train_loader) * args.epochs) / args.grad_accum_steps)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_update_steps
    )

    # Monitoring init
    use_gpu_monitor = NVML_AVAILABLE and torch.cuda.is_available()
    print("GPU monitoring:", use_gpu_monitor)

    # Prepare output files
    os.makedirs(args.outdir, exist_ok=True)
    ts = now_ts()
    run_id = ts
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
        "optimizer": "AdamW",
        "lr": args.lr,
    }
    with open(meta_json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Open files & writers
    step_csv_file = open(step_csv_path, "w", newline="")
    epoch_csv_file = open(epoch_csv_path, "w", newline="")
    step_writer = csv.writer(step_csv_file)
    epoch_writer = csv.writer(epoch_csv_file)

    # Headers (match Phase 1)
    step_writer.writerow([
        "timestamp","run_id","epoch_idx","step_idx","batch_size",
        "data_load_time_s","forward_backward_time_s","step_time_s",
        "throughput_samples_per_s","train_loss","gpu_util_pct","gpu_mem_MiB",
        "cpu_percent","host_ram_percent"
    ])
    epoch_writer.writerow([
        "timestamp","run_id","epoch_idx","epoch_time_s","throughput_samples_per_s",
        "train_loss_epoch","val_loss_epoch","val_acc_epoch","avg_gpu_util_pct","max_gpu_mem_MiB",
        "avg_cpu_percent","avg_host_ram_percent",
        "docker_cpu","docker_mem","docker_io","k8s_cpu","k8s_mem",
        "disk_read_MBps","disk_write_MBps","smi_util","smi_mem"
    ])

    global_start = time.perf_counter()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.perf_counter()
        epoch_samples = 0
        epoch_loss_sum = 0.0
        gpu_util_samples = []
        gpu_mem_samples = []
        cpu_samples = []
        ram_samples = []

        # optional warmup (skip measurement)
        if epoch == 0 and args.warmup_epochs > 0:
            print(f"[INFO] Doing {args.warmup_epochs} warm-up epochs (not measured)")
            for _ in range(args.warmup_epochs):
                for batch in train_loader:
                    pass

        optimizer.zero_grad()
        for step_idx, batch in enumerate(train_loader):
            step_start = time.perf_counter()
            # data to device
            data_to_dev_start = time.perf_counter()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            data_to_dev_end = time.perf_counter()
            data_load_time = data_to_dev_end - data_to_dev_start

            # We compute LM loss only on label tokens.
            # For each item in batch, label_positions contains (label_start, label_token_ids)
            # We will build labels tensor with -100 everywhere except at label token positions, where we put token ids.
            bsz, seq_len = input_ids.shape
            labels = torch.full((bsz, seq_len), -100, dtype=torch.long, device=device)
            for i, (lbl_start, lbl_token_ids) in enumerate(batch["label_positions"]):
                # place label_token_ids starting at lbl_start (if within seq_len)
                for j, tid in enumerate(lbl_token_ids):
                    pos = lbl_start + j
                    if pos < seq_len:
                        labels[i, pos] = tid

            compute_start = time.perf_counter()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss_scalar = loss.item()
            # gradient accumulation
            (loss / args.grad_accum_steps).backward()
            if (step_idx + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            compute_end = time.perf_counter()
            forward_backward_time = compute_end - compute_start

            step_end = time.perf_counter()
            step_time = step_end - step_start

            # throughput: number of input sequences / step_time
            batch_size_actual = input_ids.size(0)
            throughput = batch_size_actual / step_time if step_time > 0 else None

            # system metrics
            if use_gpu_monitor:
                gpu_util, gpu_mem = get_gpu_stats()
            else:
                gpu_util, gpu_mem = None, None
            cpu_pct, ram_pct = get_cpu_mem()

            # accumulate
            epoch_samples += batch_size_actual
            epoch_loss_sum += loss_scalar * batch_size_actual
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
                iso_now(),
                run_id,
                epoch,
                step_idx,
                batch_size_actual,
                f"{data_load_time:.6f}",
                f"{forward_backward_time:.6f}",
                f"{step_time:.6f}",
                f"{throughput:.3f}" if throughput is not None else "",
                f"{loss_scalar:.6f}",
                "" if gpu_util is None else gpu_util,
                "" if gpu_mem is None else gpu_mem,
                "" if cpu_pct is None else f"{cpu_pct:.2f}",
                "" if ram_pct is None else f"{ram_pct:.2f}"
            ])

            if step_idx % args.print_freq == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step_idx}/{len(train_loader)}] "
                      f"step_time={step_time:.3f}s throughput={throughput:.1f} loss={loss_scalar:.4f}")

            if args.debug and step_idx >= args.debug_steps:
                break

        # end epoch: validation
        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        avg_throughput = epoch_samples / epoch_time if epoch_time > 0 else None
        avg_loss_epoch = epoch_loss_sum / epoch_samples if epoch_samples else None

        # Evaluate (simple val loop)
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for vb in val_loader:
                input_ids = vb["input_ids"].to(device)
                attention_mask = vb["attention_mask"].to(device)
                # build labels similar to train
                bsz, seq_len = input_ids.shape
                labels = torch.full((bsz, seq_len), -100, dtype=torch.long, device=device)
                for i, (lbl_start, lbl_token_ids) in enumerate(vb["label_positions"]):
                    for j, tid in enumerate(lbl_token_ids):
                        pos = lbl_start + j
                        if pos < seq_len:
                            labels[i, pos] = tid
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                l = out.loss.item()
                val_loss_sum += l * input_ids.size(0)
                # For accuracy, compute next-token logits and check predicted label tokens
                # Simplified heuristic: pick the argmax token at first label position and compare to first label token
                logits = out.logits  # (bsz, seq_len, vocab)
                for i, (lbl_start, lbl_token_ids) in enumerate(vb["label_positions"]):
                    # get predicted token id at lbl_start
                    if lbl_start < logits.size(1) - 1:
                        pred_id = int(torch.argmax(logits[i, lbl_start, :]).item())
                        true_id = lbl_token_ids[0] if len(lbl_token_ids) > 0 else None
                        if true_id is not None and pred_id == true_id:
                            val_correct += 1
                    val_total += 1
        val_loss = val_loss_sum / val_total if val_total else None
        val_acc = val_correct / val_total if val_total else None

        # aggregate epoch-level metrics
        avg_gpu_util = float(np.mean(gpu_util_samples)) if len(gpu_util_samples) else None
        max_gpu_mem = int(max(gpu_mem_samples)) if len(gpu_mem_samples) else None
        avg_cpu = float(np.mean(cpu_samples)) if len(cpu_samples) else None
        avg_ram = float(np.mean(ram_samples)) if len(ram_samples) else None

        # system-level subprocess collectors (best-effort)
        docker_cpu, docker_mem, docker_io = query_docker_stats()
        k8s_cpu, k8s_mem = query_k8s_pod()
        disk_read_MBps, disk_write_MBps = query_iostat()
        smi_util, smi_mem = query_nvidia_smi()

        # write epoch row
        epoch_writer.writerow([
            iso_now(),
            run_id,
            epoch,
            f"{epoch_time:.6f}",
            f"{avg_throughput:.3f}" if avg_throughput else "",
            f"{avg_loss_epoch:.6f}" if avg_loss_epoch else "",
            f"{val_loss:.6f}" if val_loss else "",
            f"{val_acc:.4f}" if val_acc is not None else "",
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

        # Save checkpoint (PEFT adapter + state)
        save_peft_and_state(args.outdir, model.base_model if hasattr(model,'base_model') else model, model, run_id)

    total_time = time.perf_counter() - global_start
    print(f"[INFO] Training completed in {total_time:.2f}s. Logs saved to: {args.outdir}")
    step_csv_file.close()
    epoch_csv_file.close()

    # Final save: attempt TorchScript (best-effort)
    ts_script = os.path.join(args.outdir, f"llama3.2_3b_scripted_{run_id}.pt")
    try_script_ok = try_save_torchscript(ts_script, model, device)
    if try_script_ok:
        print(f"[INFO] TorchScript saved to {ts_script}")
    else:
        print("[WARN] TorchScript saving failed; saved PEFT adapter + (attempted) state_dict instead.")

    # close NVML
    if NVML_AVAILABLE:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

# -----------------------
# CLI
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA-3.2 3B on MultiNLI with QLoRA and log system metrics.")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-3B", help="HF repo id for base model")
    parser.add_argument("--outdir", type=str, default="./logs/llama3_2_3b_mnli", help="Output logs/checkpoints dir")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-GPU batch size (sequences)")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="comma-separated target modules (model-specific, tune if needed)")
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-examples", type=int, default=200)
    parser.add_argument("--debug-examples-val", type=int, default=200)
    parser.add_argument("--debug-steps", type=int, default=20)
    parser.add_argument("--outdir-prefix", type=str, default="")
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # small path tweak
    if args.outdir_prefix:
        args.outdir = os.path.join(args.outdir_prefix, args.outdir)
    os.makedirs(args.outdir, exist_ok=True)
    train_loop(args)
