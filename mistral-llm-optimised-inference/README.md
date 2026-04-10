# 🚀 Optimized Mistral Inference: Throughput & Memory Engineering

This repository implements a high-performance execution engine for Mistral-7B, designed to maximize throughput and minimize latency on single-GPU hardware. It demonstrates advanced techniques in quantization, memory management (KV caching), and kernel optimization.

## 🧠 Architectural Depth: The Engineering of Inference

Optimizing LLM inference goes beyond simply running a model; it requires managing the interaction between GPU compute, VRAM bandwidth, and memory state.

### 1. KV-Cache Scaling & Memory Profiling
The Key-Value (KV) cache grows linearly with sequence length ($O(N)$), consuming significant VRAM in high-concurrency environments.
- **Implementation**: Our `KVCacheSimulator` allows for point-in-time profiling of memory overhead before execution.
- **Formula**: $Bytes = Batch \times Layers \times 2 \times Heads \times SeqLen \times HeadDim \times Precision$.

### 2. Quantization (NF4 & Double Quant)
We utilize **4-bit NormalFloat (NF4)** quantization via `bitsandbytes`.
- **Double Quantization**: Reducing memory footprint further by quantizing the quantization constants themselves.
- **Impact**: Enables 7B parameter models to fit into < 6GB of VRAM, making them accessible on consumer-grade hardware or small T4 instances.

### 3. FlashAttention-2 & SDPA
By utilizing `attn_implementation="flash_attention_2"`, we leverage IO-aware attention kernels that significantly reduce memory reads/writes, leading to a 2.5x speedup in long-context scenarios.

## 🛠️ Project Structure

```text
├── inference_mistral.py # Core Inference Engine + KV Cache Profiler
├── generate_plot.py     # Benchmarking & Visualization (TPS vs. Batch Size)
└── requirements.txt     # BitsAndBytes, Transformers, Accelerate
```

## 🚀 Usage & Benchmarking

### 1. High-Throughput Inference (4-bit)
```bash
python inference_mistral.py \
  --model_id mistralai/Mistral-7B-v0.1 \
  --quantization 4bit \
  --concurrency 16
```

### 2. Low-Latency Inference (FP16 + FlashAttention)
```bash
python inference_mistral.py \
  --model_id mistralai/Mistral-7B-v0.1 \
  --dtype float16 \
  --use_flash_attn
```

## 📊 Performance Targets
- **Throughput**: > 200 Tokens/Sec (Aggregate) on a single high-bandwidth GPU.
- **Efficiency**: < 100ms per-token latency for interactive applications.

---
*Developed as part of my Applied Data Science & ML Engineering Portfolio.*
