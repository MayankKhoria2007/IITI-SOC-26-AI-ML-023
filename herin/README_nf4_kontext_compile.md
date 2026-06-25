# FLUX.1-Kontext Inference Optimization (NF4 + torch.compile)

This repository demonstrates memory-efficient compiled image editing inference for FLUX.1-Kontext-dev using a combination of:

- BitsAndBytes NF4 4-bit Quantization (Transformer + T5 Text Encoder)
- BF16 Mixed Precision Inference
- torch.compile with reduce-overhead mode
- Full-Resolution Warm-up Pass for Kernel Caching

The objective is to reduce GPU memory consumption while achieving faster repeated inference through compiled CUDA kernels on a single L4 GPU.

---

## Model Information

| Property | Value |
|----------|-------|
| Model | black-forest-labs/FLUX.1-Kontext-dev |
| Framework | Hugging Face Diffusers |
| Precision | BF16 |
| Transformer Quantization | BitsAndBytes NF4 4-bit |
| Text Encoder (T5) Quantization | BitsAndBytes NF4 4-bit |
| Runtime | Lightning AI Studio |
| GPU | NVIDIA L4 (22.5 GB) |

---

## Optimization Techniques

### 1. Transformer Quantization (BitsAndBytes NF4)

The diffusion transformer is quantized using BitsAndBytes NF4 4-bit quantization with double quantization enabled.

**Benefits:**
- ~75% reduction in transformer VRAM usage
- Double quantization reduces quantization constants memory footprint
- BF16 compute dtype preserves numerical stability
- Minimal visual quality degradation

### 2. Text Encoder (T5) Quantization (BitsAndBytes NF4)

The T5 text encoder is quantized using the same NF4 4-bit configuration.

**Benefits:**
- Efficient prompt encoding under constrained VRAM
- Enables transformer, T5, VAE, and CLIP to coexist on a single GPU
- Negligible impact on prompt fidelity

### 3. torch.compile (reduce-overhead)

The transformer is compiled using torch.compile with reduce-overhead mode after loading.

**Benefits:**
- Reduces Python overhead between CUDA kernel launches
- Speeds up all inference jobs after the first warm-up pass
- Compatible with NF4 quantized models via fullgraph=False
- Most effective for repeated inference jobs in interactive sessions

### 4. Full-Resolution Warm-up Pass

A full-resolution (1024×1024) warm-up inference pass is run at the configured number of steps before any real job.

**Benefits:**
- Compiles and caches all CUDA kernels at full resolution
- Ensures accurate latency measurement for all subsequent jobs
- Eliminates first-job slowdown caused by torch.compile kernel tracing

### 5. Deterministic CUDA Settings

CUDA benchmarking is disabled and deterministic mode is enabled.

**Benefits:**
- Ensures consistent and reproducible inference results
- Prevents non-deterministic CUDA kernel selection from affecting latency measurements

---

## Benchmark Methodology

The benchmark follows the steps below:

1. Load quantized transformer with NF4 BitsAndBytes config and use_safetensors=True.
2. Load quantized T5 text encoder with NF4 BitsAndBytes config and use_safetensors=True.
3. Build FluxKontextPipeline with pre-quantized components.
4. Move VAE and CLIP text encoder to GPU.
5. Compile transformer with torch.compile (reduce-overhead, fullgraph=False).
6. Run full-resolution warm-up inference pass to cache compiled kernels.
7. Clear CUDA cache after warm-up.
8. For each job: encode prompt, reset peak memory stats, run inference, measure latency and peak VRAM.
9. Save output image per job.

---

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Resolution | 1024 × 1024 |
| Sampling Steps | 4 (default) |
| Guidance Scale | 2.5 |
| Precision | BF16 |
| Transformer Quantization | BitsAndBytes NF4 4-bit |
| Text Encoder Quantization | BitsAndBytes NF4 4-bit |
| Max Sequence Length | 64 |
| VAE Tiling | Disabled |
| VAE Slicing | Disabled |
| torch.compile | Enabled (reduce-overhead) |
| use_safetensors | Enabled |

---

## Results

| Metric | Value |
|--------|-------|
| Compile Time | ~160 seconds |
| Inference Time | ~13.54 seconds |
| Peak VRAM Usage | ~12.36 GB |
| Output Resolution | 1024 × 1024 |

---

## Usage

Run the script and follow the interactive prompts:

```bash
python nf4_kontext_compile.py
```

```
[ Job #1 ]
Image (URL or local path): https://example.com/bedroom.jpg
Prompt: Change the bedsheet and pillows to navy blue color
```

Type `quit` or `exit` at any prompt to end the session. All outputs are saved to:

```
/teamspace/studios/this_studio/output/result_001.png
/teamspace/studios/this_studio/output/result_002.png
...
```

---

## Comparison with nf4_kontext.py (No Compile)

| Metric | nf4_kontext.py | nf4_kontext_compile.py |
|--------|---------------|----------------------|
| torch.compile | No | Yes |
| Compile Time | — | ~160 seconds |
| Inference Time | ~14.92s | ~13.54s |
| Peak VRAM | ~12.36 GB | ~12.36 GB |
| Resolution | 384 × 384 | 1024 × 1024 |
| Best For | Quick sessions | Repeated high-res jobs |

---

## Key Findings

- Successfully executed FLUX.1-Kontext-dev on a single NVIDIA L4 GPU with torch.compile enabled.
- NF4 4-bit double quantization enabled the full model (transformer + T5 + VAE + CLIP) to fit within ~12.36 GB VRAM.
- torch.compile with reduce-overhead mode reduces per-job inference time compared to the non-compiled version.
- Full-resolution warm-up is necessary — a low-resolution warm-up would not cache 1024×1024 CUDA kernels correctly.
- use_safetensors=True ensures faster and safer model weight loading.
- This script is recommended for sessions with many repeated inference jobs where the compile cost is amortized over multiple runs.

---

## References

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- [torch.compile Documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [SafeTensors](https://github.com/huggingface/safetensors)
