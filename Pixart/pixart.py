import torch
import time
import csv
import os
import gc
import subprocess

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

subprocess.run(["pip", "install", "ftfy", "-q"], check=True)

import ftfy  # noqa
from diffusers import PixArtSigmaPipeline, DPMSolverMultistepScheduler

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)

t_cold = time.time()

pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True,
    algorithm_type="dpmsolver++"
)

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

pipe.text_encoder.to("cuda")
pipe.transformer.to("cuda")
pipe.vae.to("cuda")

# ── compile transformer ───────────────────────────────────────────────────────
pipe.transformer = torch.compile(
    pipe.transformer,
    mode="reduce-overhead",
    fullgraph=False,
)
pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=False)

print(f"Cold start: {time.time()-t_cold:.1f}s")

PROMPT = (
    "A professional portrait photograph of a young woman with a modern layered "
    "wolf cut hairstyle, vivid copper and auburn hair color with face-framing highlights, "
    "a small elegant fine-line floral tattoo visible on the collarbone, natural makeup, "
    "soft studio lighting, shallow depth of field, photorealistic, highly detailed, 8k resolution"
)

# ── encode on GPU ─────────────────────────────────────────────────────────────
t_encode = time.time()
with torch.no_grad():
    prompt_embeds, prompt_attention_mask, neg_embeds, neg_mask = pipe.encode_prompt(
        PROMPT,
        device="cuda",
        dtype=torch.float16,
    )
encode_time = time.time() - t_encode
print(f"Text encode: {encode_time:.1f}s")

# ── free T5 ───────────────────────────────────────────────────────────────────
pipe.text_encoder = None
pipe.tokenizer = None
gc.collect()
torch.cuda.empty_cache()

# ── warmup ────────────────────────────────────────────────────────────────────
print("Warming up (compile happens here, takes 3-5 min)...")
with torch.no_grad():
    _ = pipe(
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_embeds=neg_embeds,
        negative_prompt_attention_mask=neg_mask,
        negative_prompt=None,
        num_inference_steps=1,
        height=512,
        width=512,
        guidance_scale=4.5,
    ).images[0]
print("Warmup done")

# ── timed runs ────────────────────────────────────────────────────────────────
configs = [
    {"steps": 8},
    {"steps": 10},
]

for cfg in configs:
    torch.cuda.reset_peak_memory_stats()

    t_gen = time.time()
    image = pipe(
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_embeds=neg_embeds,
        negative_prompt_attention_mask=neg_mask,
        negative_prompt=None,
        num_inference_steps=cfg["steps"],
        height=512,
        width=512,
        guidance_scale=4.5,
    ).images[0]
    gen_time  = time.time() - t_gen
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3

    fname = f"output_pixart_compiled1_{cfg['steps']}steps.png"
    image.save(fname)

    row = {
        "model": "PixArt-Sigma-1024-compiled",
        "steps": cfg["steps"], "height": 512, "width": 512,
        "scheduler": "DPM++", "compiled": True,
        "encode_time_sec": round(encode_time, 2),
        "gen_time_sec": round(gen_time, 2),
        "total_time_sec": round(encode_time + gen_time, 2),
        "peak_vram_gb": round(peak_vram, 3),
    }
    with open("ablation_log.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        f.seek(0, 2)
        if f.tell() == 0: writer.writeheader()
        writer.writerow(row)

    print(f"[{cfg['steps']} steps] encode: {encode_time:.1f}s | gen: {gen_time:.1f}s | total: {encode_time+gen_time:.1f}s | VRAM: {peak_vram:.3f} GB")