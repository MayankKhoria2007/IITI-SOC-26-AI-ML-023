# FLUX.1 Fill + Redux — Fast Inpainting Pipeline

--12.11 second latency and 14.44 GB VRAM

Inpaints a masked region of a photo (e.g. "replace this bed") using
**FLUX.1-Fill-dev**, conditioned on a **reference/style image** via
**FLUX.1-Redux-dev** instead of a plain text prompt. Optimized for L4 GPUs
with 4-bit (NF4) quantization, `torch.compile`, and TeaCache step-skipping
to hit sub-15s latency at 1024×1024.

## What it does

Given three inputs — a **reference image**, a **photo to edit**, and a
**mask** — it fills the masked region of the photo with content that
matches the style/appearance of the reference image (optionally nudged by
a short text prompt):

```
reference_image  ──► Redux ──► prompt_embeds, pooled_prompt_embeds ──┐
                                                                       ├──► FLUX Fill ──► result
photo + mask     ─────────────────────────────────────────────────────┘
```

The reference image is **not** pasted or copied in — Redux encodes its
style/content into embeddings that guide generation, the same way a text
prompt would, just image-conditioned instead of word-conditioned.

## Requirements

- NVIDIA GPU with ~14 GB+ free VRAM (tested on L4)
- Hugging Face account with access to:
  - [`black-forest-labs/FLUX.1-Fill-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
  - [`black-forest-labs/FLUX.1-Redux-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev)
- Python packages: `torch`, `diffusers`, `transformers`, `bitsandbytes`,
  `huggingface_hub`, `opencv-python`, `numpy`, `Pillow`

```bash
pip install torch diffusers transformers bitsandbytes huggingface_hub opencv-python numpy pillow
```

Set your HF token in the script (`login(token="hf_...")`) or via
`huggingface-cli login` beforehand — the script needs read access to both
gated Fill and Redux repos.


## Key config knobs 

| Constant | Default | What it controls |
|---|---|---|
| `DEFAULT_STEPS` | `8` | Total denoising steps. Fewer = faster but risk of under-converged detail. |
| `DEFAULT_GUIDANCE` | `30.0` | Classifier-free guidance scale. |
| `DEFAULT_HEIGHT` / `DEFAULT_WIDTH` | `1024` / `1024` | Fixed generation resolution. |
| `SEED` | `42` | Base seed (actual seed = `SEED + job_index`). Fixed so runs are reproducible when comparing settings. |
| `COMPILE_MODE` | `None` | `torch.compile` mode. `None` = default mode. **Do not set to `"reduce-overhead"`** — see caveat below. |
| `REL_L1_THRESH` | `0.30` | TeaCache skip threshold — higher skips more steps (faster, more quality risk). |
| `MAX_CONSECUTIVE_SKIPS` | `2` | Hard cap on back-to-back skipped steps, to bound color/texture drift from long skip streaks. |


## How the speed optimizations work

- **NF4 4-bit quantization** (via `bitsandbytes`) on the transformer and T5
  text encoder — cuts VRAM and speeds up matmuls.
- **`torch.compile`** on the transformer — kernel fusion for the fixed
  1024×1024 shape.
  - ⚠️ **`reduce-overhead` mode is intentionally avoided.** It captures CUDA
    graphs, which conflicts with TeaCache holding a live tensor reference
    across steps (`tc_previous_modulated_input`) — CUDA graph replay
    overwrites that buffer in place, causing either silently wrong caching
    or a hard crash (`RuntimeError: accessing tensor output of CUDAGraphs
    that has been overwritten...`). Stick to `None` (default) or
    `"max-autotune-no-cudagraphs"`.
- **TeaCache** — skips full transformer forward passes on steps where the
  model's input barely changed from the last computed step, reusing a
  cached residual instead. The noise schedule still runs through all
  `DEFAULT_STEPS` steps; only the expensive model computation is skipped
  on some of them. First and last steps are always computed.


## Known caveats

- **`POLY_COEFFS` / `RESCALE_FN`** (TeaCache's rel-L1 rescaling polynomial)
  were fit against FLUX.1 **Kontext**, not Fill. Fill has a different
  conditioning path (masked-image + mask channels concatenated before
  `x_embedder`), so the rel-L1 distribution differs somewhat. It works as a
  starting point but may need re-fitting against Fill outputs for best
  accuracy.
- **Reference image background matters.** A product photo on a plain white
  background (vs. a photo shot in a real room) gets encoded whole by
  Redux, including the background — this can occasionally bleed into
  fill-region lighting/flatness near mask edges.
- **Mask coverage determines the replaced region exactly.** If a mask only
  partially covers an object (e.g. headboard but not the frame), only the
  covered part will change, regardless of prompt wording.
- The `text_encoder` / `text_encoder_2` (T5) loaded as part of the Fill
  pipeline are no longer used for prompt encoding now that Redux handles
  it — they're currently left loaded but unused, at some VRAM/load-time
  cost. They can be removed from `build_pipeline()` if that overhead
  matters for your setup.
