# Mistral-7B Inference Optimisation

**Stack:** PyTorch, Hugging Face Transformers, `bitsandbytes`,
FlashAttention-2, Accelerate.

A small inference setup for Mistral-7B that wires together the standard
single GPU optimization knobs (4-bit NF4 quantization, double quantization,
FlashAttention-2) and a memory profiler that lets me reason about KV cache
growth before launching a run.

This is the project I am the most cautious about over claiming on. The
codebase is **a working setup, not a benchmarking paper**. The goal was to
internalize how each knob affects VRAM and throughput, not to publish
production numbers.

## What the code actually does

### KV cache memory profiler

A `KVCacheSimulator` class computes the bytes consumed by the K and V
tensors for a given batch, sequence length, and attention head configuration
before the model is loaded:

```
bytes = batch * layers * 2 * heads * seq_len * head_dim * dtype_size
```

This makes it cheap to check whether a target context length will fit on a
given GPU before paying the model load cost.

### 4-bit NF4 quantization

```
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

NF4 is the data dependent 4-bit format from QLoRA (Dettmers et al., 2023).
Double quantization halves the overhead from the per block constants.
Together these put a 7B parameter model comfortably under about 6 GB of
VRAM, which is what makes the experiment runnable on consumer hardware or
T4 class cloud instances.

### FlashAttention-2

When the host environment supports it, the model is loaded with
`attn_implementation="flash_attention_2"`. FlashAttention recomputes
attention in tiled passes that fit in SRAM, trading FLOPs for far fewer
memory reads. The benefit is largest at long context lengths.

## Repository layout

```
inference_mistral.py    Loader, generation loop, KVCacheSimulator
generate_plot.py        Plotting helpers for tokens/sec vs. batch size
assets/                 Saved figures
requirements.txt
```

## Reproduction

4-bit plus double quant:

```bash
python inference_mistral.py \
  --model_id mistralai/Mistral-7B-v0.1 \
  --quantization 4bit \
  --concurrency 16
```

FP16 plus FlashAttention-2:

```bash
python inference_mistral.py \
  --model_id mistralai/Mistral-7B-v0.1 \
  --dtype float16 \
  --use_flash_attn
```

## Honest framing

- The README originally listed throughput targets ("> 200 tokens/sec",
  "< 100 ms per token"). Those are reasonable expectations on an A100 or
  H100 with batched generation, but I have not yet measured them on
  hardware I control. The numbers I do trust come from my consumer GPU and
  are too hardware specific to advertise as a project metric. I would
  rather report no number than a misleading one.
- This project is paired with [credit-intelligence-platform](../credit-intelligence-platform),
  which uses Mistral-7B (via Ollama) for adverse action notices. Closing
  that loop is part of why the experiment was worth doing.
- The next step is to run a structured sweep (batch size by sequence
  length) on a fixed cloud instance and record both throughput and end to
  end latency.

## References

- Dao, T. (2023). *FlashAttention-2: Faster Attention with Better
  Parallelism and Work Partitioning.*
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized
  LLMs.*
- Mistral AI (2023). *Mistral 7B.* arXiv:2310.06825.
