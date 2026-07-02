# FLUX.1-Fill Inpainting Inference Optimization

This repository demonstrates memory-efficient inpainting inference for FLUX.1-Fill-dev using a combination of:

- TorchAO INT4 Weight-Only Quantization
- BF16 Mixed Precision Inference
- torch.compile with reduce-overhead mode
- VAE Slicing + Tiling for memory efficiency
- Text Encoder Offloading after prompt encoding

The objective is to reduce GPU memory consumption while performing high-quality masked image editing on a single GPU.

---

## Model Information

| Property | Value |
|----------|-------|
| Model | black-forest-labs/FLUX.1-Fill-dev |
| Framework | Hugging Face Diffusers |
| Precision | BF16 |
| Transformer Quantization | TorchAO INT4 Weight-Only |
| Post-Processing | None |
| Runtime | Lightning AI Studio |

---

## Optimization Techniques

### 1. Transformer Quantization (TorchAO INT4)

The diffusion transformer is quantized using TorchAO INT4 weight-only quantization.

**Benefits:**
- Lowest VRAM usage among all scripts (~6.66 GB peak)
- INT4 weight-only quantization is fast and compatible with torch.compile
- BF16 activations preserve numerical stability during inference
- Suitable for GPUs with less than 8 GB VRAM

### 2. torch.compile (reduce-overhead)

The transformer is compiled using torch.compile with reduce-overhead mode after loading.

**Benefits:**
- Reduces Python overhead between CUDA kernel launches
- Speeds up inference for all runs after the first warm-up pass
- Compatible with INT4 quantized models via fullgraph=False

### 3. Text Encoder Offloading

After prompt embeddings are computed, both text encoders are set to None and CUDA cache is cleared.

**Benefits:**
- Frees significant VRAM before the diffusion inference step
- Allows the quantized transformer to use the freed memory during generation
- No quality impact since embeddings are already computed and cached

### 4. VAE Slicing and Tiling

VAE slicing and tiling are enabled to reduce peak VRAM during image decoding.

**Benefits:**
- Prevents memory spikes during the VAE decode step
- Critical for keeping total VRAM under 8 GB

### 5. Warm-up Pass

A 2-step warm-up inference is run before the actual benchmark pass.

**Benefits:**
- Triggers CUDA kernel compilation for torch.compile
- Ensures accurate latency measurement for the real inference pass
- Stabilizes GPU memory allocation before benchmarking

---

## Benchmark Methodology

The benchmark follows the steps below:

1. Load quantized transformer with TorchAO INT4 weight-only config.
2. Build FluxFillPipeline with pre-quantized transformer.
3. Enable VAE slicing and tiling.
4. Compile transformer with torch.compile.
5. Encode prompts and cache embeddings.
6. Offload text encoders and clear CUDA cache.
7. Run a 2-step warm-up inference pass.
8. Reset CUDA peak memory statistics.
9. Run actual 8-step inference and measure latency.
10. Record peak allocated GPU memory.
11. Save output image.

---

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Resolution | 512 × 512 |
| Sampling Steps | 8 |
| Guidance Scale | 30.0 |
| Precision | BF16 |
| Transformer Quantization | TorchAO INT4 Weight-Only |
| Max Sequence Length | 128 |
| VAE Slicing | Enabled |
| VAE Tiling | Enabled |
| Mask Region | [100, 100, 412, 412] |
| TF32 | Enabled |
| Flash SDP | Disabled |
| Memory Efficient SDP | Enabled |

---

## Results

| Metric | Value |
|--------|-------|
| Model Load Time | ~167 seconds |
| Compile Time | ~330 seconds (~5.5 minutes) |
| Inference Time | ~14.54 seconds |
| Peak VRAM Usage | ~6.66 GB |
| Output Resolution | 512 × 512 |

---

## Sample Prompt

```
Change the color of the white bed bedding sheets to a rich, elegant midnight blue color.
```

---

## Key Findings

- TorchAO INT4 quantization achieves the lowest VRAM usage (~6.66 GB) among all tested configurations, making it suitable for entry-level GPUs.
- Text encoder offloading after prompt encoding is critical for keeping memory under control during the diffusion step.
- torch.compile adds a significant one-time compile cost (~5.5 minutes) but eliminates kernel launch overhead for all subsequent inference runs.
- Guidance scale of 30.0 is recommended for Fill-dev to produce strong inpainting edits.
- 8 inference steps provide a good balance between quality and speed for inpainting tasks.

---

## References

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [TorchAO](https://github.com/pytorch/ao)
- [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
- [torch.compile Documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [PIL ImageDraw](https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html)
