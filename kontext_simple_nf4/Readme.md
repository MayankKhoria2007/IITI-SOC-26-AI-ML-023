# FLUX.1-Kontext Inference Optimization (Interactive, No Compile)

This repository demonstrates memory-efficient interactive image editing inference for FLUX.1-Kontext-dev using a combination of:

- BitsAndBytes NF4 4-bit Quantization (Transformer + T5 Text Encoder)
- BF16 Mixed Precision Inference
- Warm-up Pass for Stable Benchmarking
- Interactive CLI for Multi-Job Editing Sessions

The objective is to run FLUX.1-Kontext-dev on a single GPU for repeated interactive image editing without torch.compile overhead.

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

### 3. Warm-up Pass

A lightweight warm-up inference is run at 64×64 resolution with 1 step before any real job.

**Benefits:**
- Initializes CUDA kernels and memory allocators
- Ensures accurate latency measurement for all subsequent jobs
- Clears any initialization overhead from benchmark results

### 4. Deterministic CUDA Settings

CUDA benchmarking is disabled and deterministic mode is enabled.

**Benefits:**
- Ensures consistent and reproducible inference results
- Prevents non-deterministic CUDA kernel selection from affecting latency measurements

---

## Benchmark Methodology

The benchmark follows the steps below:

1. Load quantized transformer with NF4 BitsAndBytes config.
2. Load quantized T5 text encoder with NF4 BitsAndBytes config.
3. Build FluxKontextPipeline with pre-quantized components.
4. Move VAE and CLIP text encoder to GPU.
5. Run a warm-up inference pass at 64×64, 1 step.
6. Clear CUDA cache after warm-up.
7. For each job: encode prompt, reset peak memory stats, run inference, measure latency and peak VRAM.
8. Save output image per job.

---

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Resolution | 384 × 384 |
| Sampling Steps | 4 (default, user-adjustable) |
| Guidance Scale | 2.5 |
| Precision | BF16 |
| Transformer Quantization | BitsAndBytes NF4 4-bit |
| Text Encoder Quantization | BitsAndBytes NF4 4-bit |
| Max Sequence Length | 77 |
| VAE Tiling | Disabled |
| VAE Slicing | Disabled |
| torch.compile | Not Used |

---

## Results

| Metric | Value |
|--------|-------|
| Inference Time | ~14.92 seconds |
| Peak VRAM Usage | ~12.36 GB |
| Output Resolution | 384 × 384 |

---

## Usage

Run the script and follow the interactive prompts:

```bash
python nf4_kontext.py
```

```
[ Job #1 ]
Image (URL or local path): https://example.com/bedroom.jpg
Prompt: Change the bedsheet to navy blue color
Steps [4]: 4
```

Type `quit` or `exit` at any prompt to end the session. All outputs are saved to:

```
/teamspace/studios/this_studio/output/result_001.png
/teamspace/studios/this_studio/output/result_002.png
...
```

---

## Key Findings

- Successfully executed FLUX.1-Kontext-dev on a single NVIDIA L4 GPU without torch.compile.
- NF4 4-bit double quantization enabled the full model (transformer + T5 + VAE + CLIP) to fit within ~12.36 GB VRAM.
- Warm-up at low resolution (64×64, 1 step) is sufficient to initialize CUDA kernels without significant time cost.
- 4–5 inference steps provide the best balance of speed and quality for interior image editing tasks.
- This script is recommended for interactive multi-job sessions where compile overhead is undesirable.

---

## References

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- [torch Documentation](https://pytorch.org/docs/stable/index.html)
