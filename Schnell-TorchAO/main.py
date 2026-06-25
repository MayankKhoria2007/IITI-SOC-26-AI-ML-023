import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, TorchAoConfig

torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)

dtype=torch.bfloat16
MODEL_ID="black-forest-labs/FLUX.1-schnell"
OUTPUT_DIR=Path("/teamspace/studios/this_studio/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPT="a futuristic city at sunset, cinematic lighting, ultra detailed"

load_start=time.perf_counter()

quantization_config=TorchAoConfig(quant_type="int4_weight_only")

transformer=FluxTransformer2DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,   
)

pipe=FluxPipeline.from_pretrained(
    MODEL_ID,
    transformer=transformer,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,  
)

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
pipe.set_progress_bar_config(disable=True)
pipe.to("cuda")

print(f"Load time: {time.perf_counter()-load_start:.2f}s")

pipe.transformer=torch.compile(
    pipe.transformer,
    mode="reduce-overhead",
    fullgraph=False,          
)

with torch.inference_mode():
    prompt_embeds, pooled_prompt_embeds, _=pipe.encode_prompt(
        prompt=PROMPT,
        prompt_2=None,
        max_sequence_length=32,
    )
pipe.text_encoder=None
pipe.text_encoder_2=None
import gc; gc.collect()
torch.cuda.empty_cache()    

with torch.inference_mode():
    _ = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=2,
        guidance_scale=0.0,
        height=1024, width=1024,
    )
torch.cuda.synchronize()
print("Warmup done")

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
start=time.perf_counter()

with torch.inference_mode():
    result=pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=2,
        guidance_scale=0.0,
        height=1024, width=1024,
    )

torch.cuda.synchronize()
latency=time.perf_counter()-start
peak_vram=torch.cuda.max_memory_allocated() / (1024 ** 3)

result.images[0].save(OUTPUT_DIR / "image_torchao2.png")
print(f"\nInference Time : {latency:.2f}s")
print(f"Peak VRAM      : {peak_vram:.2f} GB")