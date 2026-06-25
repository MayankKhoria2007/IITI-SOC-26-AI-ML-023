# PixArt-Sigma Inference Optimization

This repository demonstrates memory-efficient inference for **PixArt-Sigma-XL-2-1024-MS** using a combination of:

- FP16 Mixed Precision Inference
- DPM-Solver++ with Karras Sigmas Scheduler
- `torch.compile` on Transformer + VAE
- Text Encoder Offloading after prompt encoding
- Memory-Efficient SDP Attention
- VAE Tiling + Slicing
- Ablation Logging via CSV

The objective is to achieve ultra-low VRAM consumption while maintaining high-quality photorealistic image generation on a single GPU.

---

## Model Information

| Property | Value |
|---|---|
| Model | PixArt-alpha/PixArt-Sigma-XL-2-1024-MS |
| Framework | Hugging Face Diffusers |
| Precision | FP16 |
| Scheduler | DPM-Solver++ with Karras Sigmas |
| Transformer Quantization | None (FP16) |
| Text Encoder | Offloaded after prompt encoding |
| Compiled Components | Transformer + VAE |
| GPU | NVIDIA L4 (22.5 GB) |
| Runtime | Lightning AI Studio |

---

## Optimization Techniques

### 1. DPM-Solver++ with Karras Sigmas

The default scheduler is replaced with `DPMSolverMultistepScheduler` using Karras sigmas.

```python
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True,
    algorithm_type="dpmsolver++"
)
```

**Benefits:**
- DPM-Solver++ converges faster than DDPM/DDIM — high quality at 8–10 steps
- Karras sigmas provide better noise scheduling for sharper outputs
- Enables fewer steps without quality degradation

### 2. `torch.compile` on Transformer + VAE

Both the transformer and VAE are compiled using `torch.compile` with `reduce-overhead` mode.

```python
pipe.transformer = torch.compile(
    pipe.transformer,
    mode="reduce-overhead",
    fullgraph=False,
)
pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=False)
```

**Benefits:**
- Eliminates Python overhead between CUDA kernel launches for both components
- Compiling the VAE further reduces decode latency
- All timed runs after warm-up benefit from fully cached kernels
- `fullgraph=False` ensures compatibility with dynamic control flow in both models

### 3. Text Encoder Offloading

The T5 text encoder and tokenizer are deleted from GPU memory after prompt embeddings are computed.

```python
pipe.text_encoder = None
pipe.tokenizer = None
gc.collect()
torch.cuda.empty_cache()
```

**Benefits:**
- Frees significant VRAM before inference begins
- T5 is only needed once for prompt encoding
- Critical for achieving ultra-low peak VRAM of 1.8 GB during generation

### 4. Memory-Efficient SDP Attention

Flash SDP is disabled and memory-efficient SDP is explicitly enabled.

```python
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

**Benefits:**
- Reduces peak VRAM during attention computation
- More stable with FP16 and compiled models than Flash attention

### 5. VAE Tiling + Slicing

VAE decode is configured with both tiling and slicing enabled.

```python
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

**Benefits:**
- Processes latents in chunks instead of all at once
- Prevents memory spikes during VAE decode
- Key contributor to the ultra-low 1.8 GB peak VRAM

### 6. Ablation Logging (CSV)

Each run is automatically logged to `ablation_log.csv` with full metadata.

```python
row = {
    "model", "steps", "height", "width", "scheduler",
    "compiled", "encode_time_sec", "gen_time_sec",
    "total_time_sec", "peak_vram_gb"
}
```

**Benefits:**
- Enables systematic comparison across step counts and configurations
- Reproducible benchmarking without manual note-taking
- Easy to extend with new configs

---

## Benchmark Methodology

The benchmark follows the steps below:

1. Load PixArt-Sigma pipeline in FP16.
2. Replace scheduler with DPM-Solver++ + Karras sigmas.
3. Move transformer, VAE, and text encoder to CUDA.
4. Compile transformer and VAE with `torch.compile`.
5. Encode prompt and offload text encoder + tokenizer from GPU.
6. Run warm-up inference at 512×512 with 1 step (caches CUDA kernels).
7. For each config: reset CUDA peak memory stats, run timed inference, log results to CSV.

---

## Experimental Configuration

| Parameter | Value |
|---|---|
| Resolution | 512 × 512 |
| Sampling Steps | 8, 10 |
| Guidance Scale | 4.5 |
| Precision | FP16 |
| Scheduler | DPM-Solver++ + Karras Sigmas |
| Compiled | Transformer + VAE |
| VAE Tiling | Enabled |
| VAE Slicing | Enabled |
| Flash SDP | Disabled |
| Memory-Efficient SDP | Enabled |

---

## Results

| Steps | Gen Time | Peak VRAM |
|---|---|---|
| 8 | 9.4s | 1.8 GB |
| 10 | ~11s | ~1.8 GB |

---

## Sample Prompt

```
A professional portrait photograph of a young woman with a short asymmetric bob hairstyle, vivid copper and deep red-orange hair color, piercing green eyes, natural makeup with defined brows and neutral lips, small silver hoop nose piercing, intricate dark floral and bird tattoo covering the chest and shoulder, wearing a black tank top, dark studio background with soft gradient lighting, shallow depth of field, photorealistic, highly detailed, 8k resolution
```

---

## Key Findings

- Successfully executed PixArt-Sigma-XL-2-1024-MS at 512×512 on a single NVIDIA L4 GPU.
- Achieved an ultra-low peak VRAM of just **1.8 GB** — the lowest of all benchmarked models — through text encoder offloading combined with VAE tiling and slicing.
- DPM-Solver++ with Karras sigmas delivers high-quality photorealistic results in as few as 8 steps, making it significantly more efficient than DDPM-based schedulers.
- Compiling both the transformer and VAE (not just the transformer) provides additional latency reduction compared to transformer-only compilation.
- Warm-up with 1 step at 512×512 is sufficient to fully cache CUDA kernels before timed runs.
- PixArt-Sigma's smaller transformer architecture compared to FLUX models enables dramatically lower VRAM usage even without quantization.
- The ablation CSV logger enables systematic comparison across step counts and configurations without manual tracking.

---

## Comparison with FLUX Models

| Model | Inference Time | Peak VRAM | Quantization |
|---|---|---|---|
| PixArt-Sigma (this) | **9.4s** | **1.8 GB** | None (FP16) |
| FLUX.1-Schnell BnB NF4 | 10.69s | 13.73 GB | NF4 4-bit |
| FLUX.1-Schnell TorchAO INT4 | 14.31s | 8.43 GB | INT4 |

> PixArt-Sigma achieves the fastest inference and lowest VRAM of all three — without any quantization — due to its smaller and more efficient transformer architecture.

---

## References

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [PixArt-Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS)
- [DPMSolverMultistepScheduler](https://huggingface.co/docs/diffusers/api/schedulers/multistep_dpm_solver)
- [torch.compile Documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch SDP Attention](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_mem_efficient_sdp)
- [ftfy — Fix Text For You](https://github.com/rspeer/python-ftfy)