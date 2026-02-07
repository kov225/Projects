#!/usr/bin/env python
import argparse
import time
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# LoRA support (optional)
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimized inference script for Mistral-style LLMs"
    )

    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--quantization", type=str, default=None,
                        choices=[None, "8bit", "4bit"])
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--input_length", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--compile", action="store_true")

    return parser.parse_args()


def get_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def load_model_and_tokenizer(model_id, dtype_str, quantization, lora_path):
    # Show what device we’re using
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO]  Using device: {device}")

    # Load tokenizer
    print("[INFO]  Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    dtype = get_dtype(dtype_str)

    # Setup quantization
    quant_config = None
    if quantization:
        print(f"[INFO] ⚡ Applying {quantization} quantization…")
        if quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

    # Load main model
    print("[INFO] 🧠 Loading model… (this may take a minute)")
    if quant_config:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        ).to(device)

    # Load LoRA if provided
    if lora_path:
        if not PEFT_AVAILABLE:
            raise RuntimeError("PEFT is not installed but a LoRA path was provided.")
        print(f"[INFO] 🪄 Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    print("[INFO]  Model + tokenizer loaded successfully!")

    model.eval()
    return model, tokenizer, device


def maybe_compile_model(model, use_compile):
    if use_compile and hasattr(torch, "compile"):
        print("\n[INFO] Compiling model for extra speed…")
        return torch.compile(model)
    return model


def prepare_batch_inputs(tokenizer, prompt, batch_size, target_length, device):
    print("\n[INFO]  Preparing your prompt…")
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=target_length,
        add_special_tokens=True,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Pad or truncate to exact length for fair benchmarking
    if input_ids.shape[-1] < target_length:
        pad_len = target_length - input_ids.shape[-1]
        pad = torch.full((1, pad_len), tokenizer.pad_token_id)
        pad_mask = torch.zeros((1, pad_len))
        input_ids = torch.cat([pad, input_ids], dim=1)
        attention_mask = torch.cat([pad_mask, attention_mask], dim=1)
    elif input_ids.shape[-1] > target_length:
        input_ids = input_ids[:, -target_length:]
        attention_mask = attention_mask[:, -target_length:]

    # Duplicate input for concurrency batch
    print(f"[INFO] 📦 Replicating prompt {batch_size} times for concurrency test…")
    input_ids = input_ids.expand(batch_size, -1).to(device)
    attention_mask = attention_mask.expand(batch_size, -1).to(device)

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def generate_with_timing(model, tokenizer, inputs,
                         max_new_tokens, do_sample, temperature, top_k, top_p):

    batch_size, input_len = inputs["input_ids"].shape
    total_input_tokens = batch_size * input_len

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if do_sample:
        gen_kwargs.update(dict(
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ))

    # Sync and start timing
    torch.cuda.synchronize()
    start = time.time()

    # Actual generation
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Stop timing
    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Count tokens generated
    new_tokens_each = min(output_ids.shape[-1] - input_len, max_new_tokens)
    total_new_tokens = new_tokens_each * batch_size
    total_tokens = total_input_tokens + total_new_tokens

    return output_ids, elapsed, total_input_tokens, total_new_tokens, total_tokens


def main():
    args = parse_args()

    print("\n==============================")
    print("Mistral Optimized Inference")
    print("==============================\n")

    model, tokenizer, device = load_model_and_tokenizer(
        args.model_id, args.dtype, args.quantization, args.lora_path
    )

    model = maybe_compile_model(model, args.compile)

    # Ask for user prompt
    print("\n🗣️  Enter a prompt for the model:")
    prompt = input("> ")

    # Prepare batched inputs
    batch_inputs = prepare_batch_inputs(
        tokenizer, prompt, args.concurrency, args.input_length, device
    )

    # Warmup runs
    print(f"\n[INFO]  Running {args.warmup_steps} warmup step(s) to stabilize speed…")
    for _ in range(args.warmup_steps):
        generate_with_timing(
            model, tokenizer, batch_inputs,
            max_new_tokens=16,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    print("[INFO] Warmup done! 👍")

    # Main benchmark
    print("\n[INFO] ⚡ Running the main benchmark…")
    output_ids, elapsed, total_input_tokens, total_new_tokens, total_tokens = generate_with_timing(
        model, tokenizer, batch_inputs,
        args.max_new_tokens,
        args.do_sample,
        args.temperature,
        args.top_k,
        args.top_p,
    )

    # Show output
    print("\n==============================")
    print("📤 MODEL OUTPUT (example from batch)")
    print("==============================\n")
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

    # Metrics
    tps = total_tokens / elapsed

    print("\n==============================")
    print(" PERFORMANCE METRICS")
    print("==============================")
    print(f"• Total input tokens:     {total_input_tokens}")
    print(f"• Total generated tokens: {total_new_tokens}")
    print(f"• Tokens/sec (overall):   {tps:.2f}")
    print(f"• Time taken:             {elapsed:.3f} sec")

    # Benchmark requirement
    print("\n==============================")
    print("BENCHMARK RESULT")
    print("==============================")
    if tps >= 200:
        print(" PASS! You hit the target speed (≥200 tokens/sec)")
    else:
        print(" Did not reach 200 tokens/sec.")
        print("   Try enabling 4bit quantization or reducing output length.")


if __name__ == "__main__":
    main()

