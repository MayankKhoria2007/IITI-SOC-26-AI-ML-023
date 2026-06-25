# FLUX.1-Kontext Inference Optimization

This repository demonstrates memory-efficient inference for **FLUX.1-Kontext-dev** using a combination of:

- BitsAndBytes NF4 4-bit Quantization
- BF16 Mixed Precision Inference
- `torch.compile` with `reduce-overhead` mode
- PIL-based Post-Processing Enhancement (zero VRAM cost)

The objective is to reduce GPU memory consumption while maintaining high-quality image editing performance on a single L4 GPU.

---

## Model Information

| Property | Value |
|---|---|
| Model | black-forest-labs/FLUX.1-Kontext-dev |
| Framework | Hugging Face Diffusers |
| Precision | BF16 |
| Transformer Quantization | BitsAndBytes NF4 4-bit |
| Text Encoder (T5) Quantization | BitsAndBytes NF4 4-bit |
| Post-Processing | PIL UnsharpMask + Sharpness + Contrast |
| GPU | NVIDIA L4 (22.5 GB) |
| Runtime | Lightning AI Studio |

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
- Enables both transformer and T5 to coexist on a single GPU
- Negligible impact on prompt fidelity

### 3. Torch Compile (`reduce-overhead`)

The transformer is compiled using `torch.compile` with `reduce-overhead` mode.

**Benefits:**
- Reduces Python overhead between CUDA kernel launches
- Speeds up all inference jobs after the first warm-up pass
- Compatible with NF4 quantized models via `fullgraph=False`

### 4. PIL Post-Processing Enhancement (Zero VRAM)

Instead of a neural upscaler, a lightweight PIL pipeline enhances the 1024×1024 output at essentially no cost.

**Pipeline:**
```python
ImageFilter.UnsharpMask(radius=2.5, percent=180, threshold=2)  # edge sharpening
ImageEnhance.Sharpness(1.5)                                      # overall crispness
ImageEnhance.Contrast(1.08)                                      # dark area lift
```

**Benefits:**
- Runs in ~0.05s with zero VRAM cost
- No additional model to load
- Particularly effective for dark interior/bedroom images
- Replaces AuraSR (which added 18s+ and required 2–4 GB extra VRAM)

---

## Benchmark Methodology

The benchmark follows the steps below:

1. Load quantized transformer and T5 encoder.
2. Build `FluxKontextPipeline` with pre-quantized components.
3. Compile transformer with `torch.compile`.
4. Run a warm-up inference pass (cached CUDA kernels for job #1).
5. Reset CUDA peak memory statistics before each job.
6. Measure end-to-end inference latency per job.
7. Record peak allocated and current GPU memory.
8. Apply PIL enhancement and save output image.

---

## Experimental Configuration

| Parameter | Value |
|---|---|
| Resolution | 1024 × 1024 |
| Sampling Steps | 4–5 |
| Guidance Scale | 2.5–3.5 |
| Precision | BF16 |
| Transformer Quantization | BitsAndBytes NF4 4-bit |
| Text Encoder Quantization | BitsAndBytes NF4 4-bit |
| Max Sequence Length | 64 |
| VAE Tiling | Disabled |
| VAE Slicing | Disabled |

---

## Results

| Metric | Value |
|---|---|
| Inference Time | ~13.2s |
| Peak VRAM Usage | ~14.23 GB |
| Output Resolution | 1024 × 1024 |

---

## Sample Prompts

```
Change the bedsheet and pillows to navy blue color. 
```

---

## Key Findings

- Successfully executed FLUX.1-Kontext-dev on a single NVIDIA L4 GPU.
- NF4 4-bit double quantization enabled the full model (transformer + T5 + VAE + CLIP) to fit within ~14 GB VRAM.
- `torch.compile` with `reduce-overhead` mode eliminates Python kernel launch overhead for all jobs after warm-up.
- 4–5 inference steps provide the best balance of speed and quality for interior image editing.
- BF16 + NF4 quantization is a practical trade-off between memory efficiency and generation fidelity for real-world image editing tasks.

---

## References

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
- [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- [torch.compile Documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [PIL ImageFilter](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html)
- [PIL ImageEnhance](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html)
- [OpenCV GaussianBlur](https://docs.opencv.org/4.13.0/d4/d13/tutorial_py_filtering.html)
- [OpenCV CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)