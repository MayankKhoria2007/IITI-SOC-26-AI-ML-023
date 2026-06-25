# FLUX.1-Schnell Inference Optimization — BitsAndBytes NF4 + FP16

This repository demonstrates memory-efficient inference for **FLUX.1-Schnell** using a combination of:

- BitsAndBytes NF4 4-bit Double Quantization
- FP16 Mixed Precision Inference
- `torch.compile` with `reduce-overhead` mode
- Memory-Efficient SDP Attention
- VAE Tiling + Slicing

The objective is to reduce GPU memory consumption while maintaining fast, high-quality image generation performance on a single L4 GPU.

---

## Model Information

| Property | Value |
|---|---|
| Model | black-forest-labs/FLUX.1-schnell |
| Framework | Hugging Face Diffusers |
| Precision | FP16 |
| Transformer Quantization | BitsAndBytes NF4 4-bit + Double Quant |
| Text Encoder (T5) Quantization | BitsAndBytes NF4 4-bit + Double Quant |
| GPU | NVIDIA L4 (22.5 GB) |
| Runtime | Lightning AI Studio |

---

## Optimization Techniques

### 1. Transformer Quantization (BitsAndBytes NF4)

The diffusion transformer is quantized using BitsAndBytes NF4 4-bit quantization with double quantization enabled.

```python
DiffusersBnbConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

**Benefits:**
- ~75% reduction in transformer VRAM compared to full FP16
- Double quantization further reduces the memory footprint of quantization constants
- NF4 (NormalFloat4) is optimized for normally distributed weights — ideal for diffusion transformers
- FP16 compute dtype preserves numerical stability during matrix multiplications

### 2. Text Encoder (T5) Quantization (BitsAndBytes NF4)

The T5 text encoder is quantized using the same NF4 4-bit double quantization configuration.

```python
TransformersBnbConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

**Benefits:**
- Efficient prompt encoding under constrained VRAM
- Enables both transformer and T5 to coexist on a single GPU
- Negligible impact on prompt fidelity

### 3. `torch.compile` (`reduce-overhead`)

The transformer is compiled using `torch.compile` after model loading and before inference.

```python
pipe.transformer = torch.compile(
    pipe.transformer,
    mode="reduce-overhead",
    fullgraph=False,
)
```

**Benefits:**
- Eliminates Python overhead between CUDA kernel launches
- All inference jobs after warm-up benefit from cached compiled kernels
- `fullgraph=False` ensures compatibility with NF4 quantized models

### 4. Memory-Efficient SDP Attention

Flash SDP is disabled and memory-efficient SDP is explicitly enabled.

```python
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

**Benefits:**
- Reduces peak VRAM during attention computation
- More stable on L4 with quantized models than Flash attention

### 5. VAE Tiling + Slicing

VAE decode is configured with both tiling and slicing enabled.

```python
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

**Benefits:**
- Reduces peak VRAM during VAE decode at 1024×1024
- Processes latents in chunks instead of all at once
- Prevents OOM during decode on memory-constrained GPUs

---

## Benchmark Methodology

The benchmark follows the steps below:

1. Load NF4 4-bit quantized transformer and T5 encoder.
2. Build `FluxPipeline` with pre-quantized components.
3. Move full pipeline to CUDA.
4. Compile transformer with `torch.compile`.
5. Pre-compute prompt embeddings via `encode_prompt`.
6. Run a warm-up inference pass with 1 step (caches CUDA kernels).
7. Reset CUDA peak memory statistics.
8. Measure end-to-end inference latency.
9. Record peak allocated GPU memory.
10. Save generated output image.

---

## Experimental Configuration

| Parameter | Value |
|---|---|
| Resolution | 1024 × 1024 |
| Sampling Steps | 2 |
| Guidance Scale | 0.0 |
| Precision | FP16 |
| Transformer Quantization | BitsAndBytes NF4 4-bit + Double Quant |
| Text Encoder Quantization | BitsAndBytes NF4 4-bit + Double Quant |
| Max Sequence Length | 32 |
| VAE Tiling | Enabled |
| VAE Slicing | Enabled |
| Flash SDP | Disabled |
| Memory-Efficient SDP | Enabled |

---

## Results

| Metric | Value |
|---|---|
| Inference Time | 10.69s |
| Peak VRAM Usage | 13.73 GB |
| Resolution | 1024 × 1024 |
| Inference Steps | 2 |

---

## Sample Prompt

```
a futuristic city at sunset, cinematic lighting, ultra detailed
```

---

## Key Findings

- Successfully executed FLUX.1-Schnell at 1024×1024 on a single NVIDIA L4 GPU.
- NF4 4-bit double quantization on both transformer and T5 encoder kept peak VRAM at **13.73 GB**.
- FP16 precision (instead of BF16) combined with NF4 quantization achieved inference in **10.69 seconds**.
- Pre-computing prompt embeddings once and reusing them eliminates repeated text encoder overhead across multiple runs.
- Warm-up with 1 step (instead of 2) is sufficient to cache CUDA kernels and costs less time during startup.
- `guidance_scale=0.0` is correct for FLUX.1-Schnell — it is a distilled model that does not use classifier-free guidance.
- Only 2 inference steps are required due to FLUX.1-Schnell's distillation training, enabling fast generation.
- NF4 double quantization uses more VRAM than TorchAO INT4 (13.73 GB vs 8.43 GB) but achieves faster inference (10.69s vs 14.31s) due to optimized BitsAndBytes dequantization kernels.

---


## References

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
- [FLUX.1-Schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- [torch.compile Documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [BitsAndBytes NF4 Quantization](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes)
- [PyTorch SDP Attention](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_mem_efficient_sdp)