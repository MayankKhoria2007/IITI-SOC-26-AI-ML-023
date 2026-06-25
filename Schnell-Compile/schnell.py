import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time
from pathlib import Path

import torch
from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    BitsAndBytesConfig as DiffusersBnbConfig
)
from transformers import (
    T5EncoderModel,
    BitsAndBytesConfig as TransformersBnbConfig
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)

dtype = torch.float16
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
PROMPT = (
    "a futuristic city at sunset, cinematic lighting, "
    "ultra detailed"
)
OUTPUT_DIR = Path("/teamspace/studios/this_studio/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run(pipe, prompt_embeds, pooled_prompt_embeds, steps=2, height=1024, width=1024):  
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=steps,
            guidance_scale=0.0,
            height=height,
            width=width,
            output_type="pil",
        )
    torch.cuda.synchronize()
    latency = time.perf_counter() - start
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\nInference Time : {latency:.2f}s")
    print(f"Peak VRAM      : {peak_vram:.2f} GB")
    image = result.images[0]
    save_path = OUTPUT_DIR / "image.png"
    image.save(save_path)
    print(f"Saved image to: {save_path}")


load_start = time.perf_counter()

bnb_transformer = DiffusersBnbConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
    bnb_4bit_use_double_quant=True,
)
bnb_text = TransformersBnbConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
    bnb_4bit_use_double_quant=True,
)

transformer = FluxTransformer2DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    quantization_config=bnb_transformer,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)

text_encoder_2 = T5EncoderModel.from_pretrained(
    MODEL_ID,
    subfolder="text_encoder_2",
    quantization_config=bnb_text,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    transformer=transformer,
    text_encoder_2=text_encoder_2,
    torch_dtype=dtype,
)

pipe.to("cuda")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
pipe.set_progress_bar_config(disable=True)

load_time = time.perf_counter() - load_start
print(f"\nModel Load Time: {load_time:.2f}s")

pipe.transformer = torch.compile(
    pipe.transformer,
    mode="reduce-overhead",
    fullgraph=False,
)

with torch.inference_mode():
    prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=PROMPT,
        max_sequence_length=32
    )

with torch.inference_mode():
    _ = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=1,
        guidance_scale=0.0,
        height=1024, width=1024,
    )
print("Warmup done")

print("\nGenerating image")
run(pipe, prompt_embeds, pooled_prompt_embeds)