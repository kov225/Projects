"""
Simple and optimized inference benchmark for Mistral-style LLMs.

This script focuses on understanding inference-time performance:
batching, warmup runs, quantization, and GPU memory usage.
"""

import argparse
import json
import time

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# LoRA support is optional
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def parse_args():
    """Read command-line arguments so the script is easy to experiment with."""
    parser = argparse.ArgumentParser(
        description="Inference benchmark for Mistral LLMs"
    )

    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)

    parser.add_argument(
        "--mode",
        type=str,
        default="optimized",
        choices=["baseline", "optimized"],
        help="baseline = simple run, optimized = batching + warmup + quantization"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"]
    )

    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=[None, "8bit", "4bit"]
    )

    parser.add_argument("--input_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=32)

    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--benchmark_runs", type=int, default=5)

    return parser.parse_args()


def get_dtype(dtype_str):
    """Convert dtype string to torch dtype."""
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_gpu_memory():
    """Return current GPU memory usage in MB."""
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return allocated, reserved


def load_model_and_tokenizer(args):
    """Load tokenizer and model, optionally applying quantization and LoRA."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    dtype = get_dtype(args.dtype)

    quant_config = None
    if args.mode == "optimized" and args.quantization:
        print(f"Applying {args.quantization} quantization")
        if args.quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        elif args.quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

    if quant_config:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
            torch_dtype=dtype,
            quantization_config=quant_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
        ).to(device)

    if args.lora_path:
        if not PEFT_AVAILABLE:
            raise RuntimeError("PEFT is not installed but a LoRA path was provided.")
        print(f"Loading LoRA adapter from {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()
    return model, tokenizer, device


def prepare_inputs(tokenizer, prompt, batch_size, target_len, device):
    """Tokenize the prompt and duplicate it to simulate concurrent requests."""
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=target_len,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    if input_ids.shape[-1] < target_len:
        pad_len = target_len - input_ids.shape[-1]
        pad_ids = torch.full((1, pad_len), tokenizer.pad_token_id)
        pad_mask = torch.zeros((1, pad_len))
        input_ids = torch.cat([pad_ids, input_ids], dim=1)
        attention_mask = torch.cat([pad_mask, attention_mask], dim=1)

    input_ids = input_ids.expand(batch_size, -1).to(device)
    attention_mask = attention_mask.expand(batch_size, -1).to(device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def generate_once(model, tokenizer, inputs, args):
    """Run one generation pass and return elapsed time."""
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if args.do_sample:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        })

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        model.generate(**inputs, **gen_kwargs)

    torch.cuda.synchronize()
    return time.time() - start


def benchmark(model, tokenizer, inputs, args):
    """Run multiple generations and compute latency and throughput."""
    latencies = []

    for _ in range(args.benchmark_runs):
        t = generate_once(model, tokenizer, inputs, args)
        latencies.append(t)

    latencies = np.array(latencies)

    batch_size, input_len = inputs["input_ids"].shape
    total_tokens = batch_size * (input_len + args.max_new_tokens)

    mean_latency = latencies.mean()
    tokens_per_sec = total_tokens / mean_latency

    return {
        "mean_latency_sec": mean_latency,
        "p50_latency_sec": float(np.percentile(latencies, 50)),
        "p95_latency_sec": float(np.percentile(latencies, 95)),
        "tokens_per_sec": tokens_per_sec,
    }


def main():
    args = parse_args()

    print("\nLLM Inference Benchmark\n")

    model, tokenizer, device = load_model_and_tokenizer(args)

    prompt = input("Enter a prompt: ")

    batch_size = 1 if args.mode == "baseline" else args.concurrency

    inputs = prepare_inputs(
        tokenizer,
        prompt,
        batch_size,
        args.input_length,
        device
    )

    if args.mode == "optimized":
        print("Running warmup...")
        for _ in range(args.warmup_steps):
            generate_once(model, tokenizer, inputs, args)
        print("Warmup complete.\n")

    alloc_before, _ = get_gpu_memory()

    metrics = benchmark(model, tokenizer, inputs, args)

    alloc_after, reserved = get_gpu_memory()

    results = {
        "model": args.model_id,
        "mode": args.mode,
        "quantization": args.quantization,
        "batch_size": batch_size,
        "metrics": metrics,
        "vram_mb": {
            "allocated_before": alloc_before,
            "allocated_after": alloc_after,
            "reserved": reserved,
        },
    }

    print("\nResults:")
    print(json.dumps(results, indent=2))

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results to benchmark_results.json")


if __name__ == "__main__":
    main()
