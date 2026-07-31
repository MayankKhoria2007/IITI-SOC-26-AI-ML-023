# Interior AI Studio — Model & Inference Optimization

**Intelligent Interior Design Platform — see the change before you make it.**

This repo is of **Interior AI Studio**, a virtual try-on tool for interior design (upload a room photo → describe a change → get a photorealistic edit). It holds the experiments used to find the fastest, lowest-VRAM way to run each diffusion model on constrained GPUs (Tesla T4 / L4), plus the production inference services that were shipped to Modal.

The full web app (frontend + backend + architecture docs) lives in a separate repo: [IITISOC-InteriorAI-Studio](https://github.com/Prathamesh-Hingol/IITISOC-InteriorAI-Studio).

Core modules of the product:

| Module | What it does |
|---|---|
| A. Furniture Inpainting | Composite a furniture photo into a room, matching light/perspective/scale |
| B. Image Editing | Text-guided edits to an uploaded room photo (FLUX.1-Kontext) |
| C. Text-to-Interior | Generate a room from a text description |
| D. Object Repositioning | Segment an object, drag it, auto-rescale by depth, inpaint the gap |
| E. 3D View Synthesis | Reconstruct an approximate 3D view of a room from one photo |

---

## Repository Map

```
├── README.md                              ← this file
├── FLUX.2-Klein 4B Inference Optimization/
│   ├── Readme.md
│   ├── code/main.py, requirement.txt
│   └── outputs/output.png, ss_output.jpeg
├── Fill-INT4/          Readme.md, int4_fill.py, int4_fill.png
├── Kontext-INT8/       Readme.md, generate.py, result_001.png
├── Kontext-NF4/        Readme.md, new.py, output.jpeg, output1.jpg, prompt.jpg
├── Kontext-compile/    Readme.md, nf4_kontext_compile.py, nf4_kontext_compile.png
├── Pixart/             Readme.md, pixart.py, output.png
├── Sana 1.6B/          Readme.md, main.py, output.jpg
├── Schnell-Compile/    Readme.md, README_AOT_FLUX_Schnell.md, schnell.py, output.jpg
├── Schnell-NF4/        Readme.md, nf4_schnell.py, nf4_schnell.jpeg
├── Schnell-TorchAO/    Readme.md, main.py, output.jpg
├── fill-redux-nf4-teacache-compile/   Readme.md, main.py, bed.jpg, bed1.jpg, initial.jpg, output.jpg, output1.jpg
├── kontext_simple_nf4/ Readme.md, nf4_kontext.py, nf4_kontext.png
├── modal.py                # Kontext production service (Cloudinary + TeaCache, bnb NF4)
├── Kontext-final           # Kontext production service v2 (Nunchaku SVDQuant INT4, TeaCache)
├── Schnell-final           # Schnell + LoRA production service (Modal, FastAPI)
├── schnell_finetuning.py   # LoRA fine-tuning trainer for FLUX.1-Schnell on Modal
├── prompt-enhance-final    # Qwen2.5-7B prompt-safety/enhancement microservice (vLLM on Modal)
├── segmentation-final      # Interactive SAM2 object segmentation CLI
├── drag-drop -final        # Segment → move → depth-rescale → inpaint pipeline (SAM2 + LaMa + Depth-Anything + FLUX.1-Fill)
├── fill-combined-final     # All-in-one Modal service: SAM2 + T5(NF4) + Nunchaku FLUX.1-Fill + Redux + Depth-Anything + TeaCache
└── teacache.py             # Standalone FLUX.1-Kontext + TeaCache reference script
```

---

## Experiment Folders (benchmarked notebooks/scripts)

Each folder below is a self-contained benchmark of one model + one optimization stack, run on a single GPU, with its own README, script, and sample output image.

| Folder | Model | Key Optimizations | Precision | GPU | Inference Time | Peak VRAM |
|---|---|---|---|---|---|---|
| **FLUX.2-Klein 4B Inference Optimization** | FLUX.2-Klein 4B | TorchAO INT8 (transformer) + BitsAndBytes INT8 (text encoder) | BF16 | T4 (16GB) | 10.51s | 9.38 GB |
| **Fill-INT4** | FLUX.1-Fill-dev (inpainting) | TorchAO INT4 weight-only + `torch.compile` (reduce-overhead) + text-encoder offload + VAE slicing/tiling | BF16 | — | ~14.5s | ~6.66 GB |
| **Kontext-INT8** | FLUX.1-Kontext-dev | TorchAO INT8 weight-only + GPU-native execution override + manual text-encoder CPU offload + mem-efficient SDP | BF16 | L4 (24GB) | ~32.5s | 13.66 GB |
| **Kontext-NF4** | FLUX.1-Kontext-dev | BitsAndBytes NF4 4-bit (transformer + T5) + `torch.compile` + PIL post-sharpen (zero-VRAM, replaces AuraSR) | BF16 | L4 (22.5GB) | ~13.2s | ~14.23 GB |
| **Kontext-compile** | FLUX.1-Kontext-dev | Same as Kontext-NF4 + full-resolution warm-up pass + deterministic CUDA settings for reproducible compiled kernels | BF16 | L4 (22.5GB) | ~13.5s | ~12.36 GB |
| **kontext_simple_nf4** | FLUX.1-Kontext-dev | BitsAndBytes NF4 (no `torch.compile`) — interactive CLI, low-res warm-up | BF16 | L4 (22.5GB) | ~14.9s | ~12.36 GB |
| **Pixart** | PixArt-Sigma-XL-2-1024-MS | DPM-Solver++ (Karras sigmas) + `torch.compile` on transformer **and** VAE + text-encoder deletion + mem-efficient SDP + VAE tiling/slicing + CSV ablation logging | FP16 | T4 | 9.4s | **1.8 GB** (lowest of all models) |
| **Sana 1.6B** | SANA 1.6B (`Sana_1600M_1024px_BF16_diffusers`) | `torch.compile(max-autotune)` w/ CUDA graphs + zero CPU-offload VRAM locking + BF16→FP16 cast (Turing has no native BF16) + VAE slicing/tiling | FP16 | T4 (16GB) | ~3.5s (post-warmup) | fits in 16GB |
| **Schnell-Compile** | FLUX.1-Schnell | BitsAndBytes NF4 double-quant + `torch.compile` + mem-efficient SDP + VAE tiling/slicing | FP16 | L4 | 10.69s | 13.73 GB |
| **Schnell-Compile/README_AOT_FLUX_Schnell.md** | FLUX.1-Schnell transformer | Ahead-of-Time export via PyTorch AOTInductor (forward pre-hook captures sample inputs → FX graph → compiled `model.pt2`) | — | — | ~2.5s (transformer only) | ~11.24 GB — *pipeline integration still WIP* |
| **Schnell-NF4** | FLUX.1-Schnell | BitsAndBytes NF4 (transformer + T5), no compile, VAE slicing/tiling | FP16 | — | ~13.26s | ~11.85 GB |
| **Schnell-TorchAO** | FLUX.1-Schnell | TorchAO INT4 weight-only + `torch.compile` + text-encoder offload + mem-efficient SDP + VAE tiling/slicing | BF16 | L4 (22.5GB) | 14.31s | **8.43 GB** |
| **fill-redux-nf4-teacache-compile** | FLUX.1-Fill-dev + FLUX.1-Redux-dev (image-conditioned inpainting) | BitsAndBytes NF4 + `torch.compile` (no `reduce-overhead`, conflicts with TeaCache buffers) + **TeaCache** step-skipping | — | L4 | 12.11s | 14.44 GB |


---


| File | Purpose | Model(s) | Optimizations |
|---|---|---|---|
| **modal.py** | Production Kontext image-editing service (`flux-kontext-cloudinary-teacache-service`), FastAPI + Cloudinary upload | FLUX.1-Kontext-dev | BitsAndBytes NF4 (transformer + T5) + **TeaCache** patched forward pass + block-level `torch.compile` + PIL output enhancement |
| **Kontext-final** | Newer Kontext production service, same app name, upgraded backend | FLUX.1-Kontext-dev | **Nunchaku SVDQuant INT4** transformer (replaces bnb NF4) + TeaCache + dimension snapping for FLUX-friendly resolutions; GPU snapshotting disabled (Nunchaku INT4 kernels crash on restore) |
| **Schnell-final** | Production text-to-interior service (`flux-schnell-lora`) with LoRA support and Cloudinary output | FLUX.1-Schnell + custom LoRA adapter | Modal `@app.cls` with persistent HF/LoRA volumes, FastAPI endpoint, fixed 4-step distilled sampling |
| **schnell_finetuning.py** | LoRA fine-tuning trainer for Schnell on an interior-design dataset (`victorzarzu/interior-design-prompt-editing-dataset-test`) | FLUX.1-Schnell | Runs on Modal GPU; custom `Dataset`/`DataLoader`, mock embedding path for AOT-style training loop, gradient checkpointing imports |
| **prompt-enhance-final** | Prompt safety filter + prompt-quality enhancer microservice | Qwen2.5-7B-Instruct-AWQ | Served via **vLLM** on Modal; AWQ-quantized LLM; separate system prompts for Schnell (generation) vs. Fill (inpainting) tasks; returns `SAFE|||<prompt>` / `UNSAFE` |
| **segmentation-final** | Interactive CLI for click-to-mask object segmentation (used to build masks for inpainting/repositioning) | SAM2 (`sam2.1_hiera_large`) | Session-based mask candidate generation, morphological mask cleanup (`binary_fill_holes`, `binary_closing`), multi-click merge/remove workflow |
| **drag-drop -final** | Full drag-and-drop object repositioning pipeline: segment → cut out → inpaint background → depth-rescale → recomposite | SAM2 + LaMa (background fill) + Depth-Anything-V2-Small + FLUX.1-Fill-dev | TorchAO-quantized FLUX.1-Fill, depth-based auto-rescaling on reposition, mem-efficient SDP |
| **fill-combined-final** | All-in-one Modal microservice (`herin_final`) exposing segmentation + inpainting + object-move as REST endpoints | SAM2 + T5-XXL (NF4) + **Nunchaku INT4/FP4 FLUX.1-Fill** + FLUX.1-Redux prior + Depth-Anything-V2-Small | NF4 T5 (9.4GB→2.4GB), Nunchaku quantized transformer, **TeaCache**, square-padding helpers for non-square inputs, per-endpoint FastAPI routes for click/choose/remove/generate/extract |
| **teacache.py** | Standalone reference implementation of the **TeaCache** step-skipping patch for FLUX.1-Kontext | FLUX.1-Kontext-dev | BitsAndBytes NF4, TeaCache (polynomial-fitted rel-L1 skip threshold), deterministic CUDA config, PIL output sharpening |

---

## References

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [TorchAO](https://github.com/pytorch/ao)
- [BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
- [Nunchaku (SVDQuant)](https://github.com/nunchaku-tech/nunchaku)
- [Modal](https://modal.com)
- [SAM2](https://github.com/facebookresearch/sam2)
- FLUX.1 / FLUX.2 model family — [Black Forest Labs](https://huggingface.co/black-forest-labs)
- [PixArt-Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS)
- [SANA](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers)
