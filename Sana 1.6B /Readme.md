# SANA Optimization Suite for Tesla T4

This repository contains a highly optimized, native implementation for executing NVIDIA's ultra-fast **SANA (1.6 Billion parameter)** text-to-image model directly on a **Tesla T4 GPU** (16GB VRAM). 

By leveraging native FP16 execution, custom VRAM locking strategies, and aggressive compilation optimizations, this suite achieves blazing-fast inference speeds without relying on heavy quantization or CPU offloading.

---

## 📌 Model Information

| Property | Value |
| :--- | :--- |
| **Base Model** | `Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers` |
| **Framework** | Hugging Face Diffusers |
| **Target Hardware** | NVIDIA Tesla T4 (16GB VRAM) |
| **Native Precision** | `torch.float16` (Adapted from BF16 for Turing Compatibility) |
| **Optimization Runtime** | PyTorch 2.x Inductor / CUDA Graphs (`torch.compile`) |

---

## 🚀 Features & Optimization Techniques

### 1. PyTorch 2.0 `torch.compile` (max-autotune)
The core diffusion transformer is fully compiled using PyTorch's Inductor backend with heavy micro-optimizations (`conv_1x1_as_mm`, `coordinate_descent_tuning`). By capturing the model execution graph via **CUDA Graphs**, it drastically eliminates CPU launch overhead during inference.

### 2. VRAM Locking (Zero CPU Offloading)
Unlike typical setups that constantly swap model layers between System RAM and GPU memory, this pipeline locks the entire model (Transformer, Text Encoder, and VAE) straight into CUDA memory. This completely removes the bottleneck caused by moving layers across the slow PCIe bus.

### 3. Native FP16 Type Adaptation
SANA's original weights are distributed in `bfloat16`. Because the Tesla T4 architecture lacks hardware acceleration for BF16, trying to execute it natively forces the GPU into slow software emulation. The script explicitly casts everything to `torch.float16` to unlock native Tensor Core acceleration.

### 4. VAE Slicing & Tiling
To support a zero-offload strategy without triggering Out-Of-Memory (OOM) errors, the script uses VAE slicing (decoding latents in smaller batches) and VAE tiling (splitting the image into overlapping chunks during decoding) to flatten memory spikes.

---

## 📊 Experimental Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Resolution** | 512 × 512 | Adjusted down from 1024 to fit completely into native VRAM |
| **Sampling Steps** | 7 | Sprint-generation pipeline for rapid inference |
| **Guidance Scale** | 5.0 | Standard Classifier-Free Guidance (CFG) for SANA alignment |
| **Memory Format** | Contiguous | `channels_last` intentionally skipped to avoid T4 bottlenecks |
| **VAE Slicing / Tiling** | Enabled | Prevent memory spikes during the final image decoding phase |

---

## 📈 Benchmark Results

| Metric | Value |
| :--- | :--- |
| **Warmup & Compilation Time** | ~15 - 20 minutes *(First execution only)* |
| **Optimized Inference Time** | **Fast Sprint** ~ 3.5 seconds |
| **VRAM Profile** | Fits perfectly within the T4's 16GB limit |
| **Output Resolution** | 512 × 512 *(Saved to `cyberpunk_sana.png`)* |

> [!IMPORTANT]
> **The Compilation Tax:** Because the script uses `mode="max-autotune"`, the very first run will take an extended period to complete the warmup phase while PyTorch profiles your hardware. Subsequent generations bypass this completely and execute instantly.
