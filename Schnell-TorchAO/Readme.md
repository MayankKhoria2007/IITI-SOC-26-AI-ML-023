# FLUX.1-Schnell Inference Optimization

This repository demonstrates memory-efficient inference for **FLUX.1-Schnell** using a combination of:

- TorchAO INT4 Weight-Only Quantization
- BF16 Mixed Precision Inference
- `torch.compile` with `reduce-overhead` mode
- Text Encoder Offloading after prompt encoding
- Memory-Efficient SDP Attention

The objective is to reduce GPU memory consumption while maintaining fast, high-quality image generation performance on a single L4 GPU.

---

## Model Information

| Property | Value |
|---|---|
| Model | black-forest-labs/FLUX.1-schnell |
| Framework | Hugging Face Diffusers |
| Precision | BF16 |
| Transformer Quantization | TorchAO INT4 Weight-Only |
| Text Encoder | Offloaded after prompt encoding |
| GPU | NVIDIA L4 (22.5 GB) |
| Runtime | Lightning AI Studio |

---

## Optimization Techniques

### 1. Transformer Quantization (TorchAO INT4)

The diffusion transformer is quantized using TorchAO INT4 weight-only quantization via `TorchAoConfig`.

**Benefits:**
- ~80% reduction in transformer VRAM compared to full BF16
- Weight-only quantization keeps activations in BF16 for numerical stability
- Faster memory bandwidth utilization during inference
- Minimal visual quality degradation at 1024×1024

### 2. `torch.compile` (`reduce-overhead`)

The transformer is compiled using `torch.compile` with `reduce-overhead` mode after model loading.

**Benefits:**
- Eliminates Python overhead between CUDA kernel launches
- All inference jobs after warm-up benefit from cached compiled kernels
- Compatible with TorchAO INT4 quantized models via `fullgraph=False`

### 3. Text Encoder Offloading

Both text encoders are deleted from GPU memory after prompt embeddings are computed.

```python
pipe.text_encoder = None
pipe.text_encoder_2 = None
gc.collect()
torch.cuda.empty_cache()
```

**Benefits:**
- Frees significant VRAM before inference begins
- Text encoders are only needed once for prompt encoding
- Enables more headroom for transformer and VAE during generation

### 4. Memory-Efficient SDP Attention

Flash SDP is disabled and memory-efficient SDP is explicitly enabled.

```python
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

**Benefits:**
- Memory-efficient attention reduces peak VRAM during attention computation
- More stable on L4 compared to Flash attention for quantized models

### 5. VAE Tiling + Slicing

VAE decode is configured with both tiling and slicing enabled.

```python
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

**Benefits:**
- Reduces peak VRAM during VAE decode at 1024×1024
- Processes image in chunks instead of all at once

---

## Benchmark Methodology

The benchmark follows the steps below:

1. Load TorchAO INT4 quantized transformer.
2. Build `FluxPipeline` with quantized transformer.
3. Encode prompt and offload text encoders from GPU.
4. Compile transformer with `torch.compile`.
5. Run a warm-up inference pass (caches CUDA kernels).
6. Reset CUDA peak memory statistics.
7. Measure end-to-end inference latency.
8. Record peak allocated GPU memory.
9. Save generated output image.

---

## Experimental Configuration

| Parameter | Value |
|---|---|
| Resolution | 1024 × 1024 |
| Sampling Steps | 2 |
| Guidance Scale | 0.0 |
| Precision | BF16 |
| Transformer Quantization | TorchAO INT4 Weight-Only |
| Max Sequence Length | 32 |
| VAE Tiling | Enabled |
| VAE Slicing | Enabled |
| Flash SDP | Disabled |
| Memory-Efficient SDP | Enabled |

---

## Results

| Metric | Value |
|---|---|
| Inference Time | 14.31s |
| Peak VRAM Usage | 8.43 GB |
| Resolution | 1024 × 1024 |
| Inference Steps | 2 |

---

## Sample Prompt

```
a futuristic city at sunset, cinematic lighting, ultra detailed
```

---

## Usage

Run inference:

```bash
python main.py
```

The generated image will be saved as:

```
output/image_torchao2.png
```

---

## Key Findings

- Successfully executed FLUX.1-Schnell at 1024×1024 on a single NVIDIA L4 GPU.
- TorchAO INT4 weight-only quantization reduced peak VRAM to just **8.43 GB** — less than half of the full BF16 requirement.
- Text encoder offloading after prompt encoding freed significant VRAM with zero quality impact.
- `torch.compile` with `reduce-overhead` eliminates Python-level kernel launch overhead for all post-warmup jobs.
- Guidance scale of `0.0` is correct for FLUX.1-Schnell as it is a distilled model that does not use classifier-free guidance.
- Only 2 inference steps are needed for FLUX.1-Schnell due to its distillation training, enabling extremely fast generation.
- Demonstrates feasibility of running a 12B parameter diffusion transformer within 8.43 GB VRAM on an L4 GPU.

---

## References

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [TorchAO](https://github.com/pytorch/ao)
- [FLUX.1-Schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- [torch.compile Documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [TorchAO Quantization Docs](https://pytorch.org/ao/stable/index.html)
- [PyTorch SDP Attention](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_mem_efficient_sdp)