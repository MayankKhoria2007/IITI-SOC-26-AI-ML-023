# FLUX.1-Schnell Inference Optimization

This repository demonstrates memory-efficient text-to-image inference for FLUX.1-schnell using a combination of:

- BitsAndBytes NF4 4-bit Quantization (Transformer + T5 Text Encoder)
- FP16 Mixed Precision Inference
- VAE Slicing + Tiling for memory efficiency

The objective is to reduce GPU memory consumption while maintaining high-quality image generation performance on a single GPU.

---

## Model Information

| Property | Value |
|----------|-------|
| Model | black-forest-labs/FLUX.1-schnell |
| Framework | Hugging Face Diffusers |
| Precision | FP16 |
| Transformer Quantization | BitsAndBytes NF4 4-bit |
| Text Encoder (T5) Quantization | BitsAndBytes NF4 4-bit |
| Runtime | Lightning AI Studio |

---

## Optimization Techniques

### 1. Transformer Quantization (BitsAndBytes NF4)

The diffusion transformer is quantized using BitsAndBytes NF4 4-bit quantization with double quantization enabled.

**Benefits:**
- ~75% reduction in transformer VRAM usage
- Double quantization reduces quantization constants memory footprint
- FP16 compute dtype preserves numerical stability
- Minimal visual quality degradation

### 2. Text Encoder (T5) Quantization (BitsAndBytes NF4)

The T5 text encoder is quantized using the same NF4 4-bit configuration.

**Benefits:**
- Efficient prompt encoding under constrained VRAM
- Enables both transformer and T5 to coexist on a single GPU
- Negligible impact on prompt fidelity

### 3. VAE Slicing and Tiling

VAE slicing and tiling are enabled to reduce peak VRAM during image decoding.

**Benefits:**
- Reduces memory spikes during VAE decode step
- Allows higher resolution outputs without OOM errors

---

## Benchmark Methodology

The benchmark follows the steps below:

1. Load quantized transformer and T5 encoder with NF4 BitsAndBytes config.
2. Build FluxPipeline with pre-quantized components.
3. Move VAE and CLIP text encoder to CUDA.
4. Enable VAE slicing and tiling.
5. Run inference and measure end-to-end latency.
6. Record peak allocated GPU memory.
7. Save output image.

---

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Resolution | 1024 × 1024 |
| Sampling Steps | 2 |
| Guidance Scale | 0.0 (schnell is guidance-free) |
| Precision | FP16 |
| Transformer Quantization | BitsAndBytes NF4 4-bit |
| Text Encoder Quantization | BitsAndBytes NF4 4-bit |
| Max Sequence Length | 128 |
| VAE Slicing | Enabled |
| VAE Tiling | Enabled |

---

## Results

| Metric | Value |
|--------|-------|
| Model Load Time | ~62 seconds |
| Inference Time | ~13.26 seconds |
| Peak VRAM Usage | ~11.85 GB |
| Output Resolution | 1024 × 1024 |

---

## Sample Prompt

```
a futuristic city at sunset, cinematic lighting, ultra detailed
```

---

## Key Findings

- Successfully ran FLUX.1-schnell with NF4 4-bit quantization on both transformer and T5 encoder.
- Double quantization reduced memory footprint without noticeable quality loss.
- Guidance scale of 0.0 is correct for schnell — it is a guidance-distilled model and does not use CFG.
- VAE slicing + tiling effectively prevents memory spikes during decoding.
- 2 inference steps provide fast generation suitable for rapid prototyping.

---

## References

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- [torch Documentation](https://pytorch.org/docs/stable/index.html)
