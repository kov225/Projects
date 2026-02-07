# Optimized Inference Pipeline for Mistral LLMs

This repository provides a **generalized, high-performance inference engine** for running Mistral-based large language models on a single NVIDIA T4 GPU.  
It is built to maximize **throughput, latency efficiency, and GPU utilization**, while remaining simple, reproducible, and compatible with both base and LoRA-fine-tuned models.

The project was developed as part of an ML Engineer assignment focused on **efficient LLM deployment**, quantization, concurrency scaling, and benchmarking.

---

## 🚀 Key Features

- Works with any `MistralForCausalLM` model from HuggingFace  
- Supports **4-bit and 8-bit quantization** via `bitsandbytes`  
- FP16/BF16 compute support for T4 GPUs  
- **Batch concurrency** (default: 32 parallel prompts)  
- **Warmup passes** to stabilize CUDA kernel performance  
- Optional **LoRA adapter** loading via `peft`  
- End-to-end **performance benchmarking** (tokens/sec)  
- Designed for environments that offer free T4 GPUs:
  - Google Colab  
  - Kaggle  
  - Amazon SageMaker Studio Lab  

---

## 🎯 Benchmark Target

The assignment benchmark:

### **≥ 200 tokens/sec throughput**  
(input + output tokens combined)

This script includes batching, warmup, and quantization to meet or approach this target reliably on a single T4 GPU.

---

## 📦 Installation

Install all dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:
- transformers  
- accelerate  
- bitsandbytes  
- torch  
- peft (LoRA support)

---

## ▶️ Usage

Basic example:

```bash
python inference_mistral.py \
  --model_id mistralai/Mistral-7B-v0.1 \
  --quantization 4bit \
  --dtype float16 \
  --concurrency 32 \
  --input_length 128 \
  --max_new_tokens 128 \
  --warmup_steps 2
```

The script will:
1. Load the model (with quantization if selected)
2. Ask for your input prompt
3. Replicate the prompt into a batch of size 32
4. Run warmup passes
5. Execute the benchmark
6. Output metrics including tokens/sec

---

## 🧩 LoRA Support

You can run LoRA-fine-tuned Mistral models by adding:

```bash
python inference_mistral.py \
  --model_id mistralai/Mistral-7B-v0.1 \
  --lora_path <path-or-hf-repo>
```

The script automatically merges the LoRA adapter with the base model for inference.

---

## 📊 Output & Metrics

The script prints:

- Total input tokens  
- Total generated tokens  
- Wall-clock inference time  
- Throughput (tokens/sec)  
- Benchmark pass/fail  

Example result:

```
Total tokens/sec: 215.3
Success! Benchmark target (≥200 tokens/sec) achieved.
```

---

## 🗂️ Repository Structure

```
.
├── inference_mistral.py      # Main optimized inference + benchmark script
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

---

## 🧠 Technical Highlights

- **Quantization (4-bit / 8-bit):**  
  Reduces VRAM usage and speeds up inference, making 7B models feasible on T4.

- **Batch concurrency:**  
  Simulates 32 parallel requests → boosts throughput.

- **Warmup passes:**  
  Avoids the common “first batch is slow” issue in CUDA.

- **Device-aware loading:**  
  Automatically places layers across available GPU memory.

- **LoRA compatibility:**  
  Allows inference with small, domain-specific adapters without retraining.

---


