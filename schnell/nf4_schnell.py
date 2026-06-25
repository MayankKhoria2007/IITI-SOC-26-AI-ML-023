import os

HF_CACHE_DIR = os.environ.get("HF_HOME", "/teamspace/studios/this_studio/hf_cache")
os.environ["HF_HOME"]      = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import time
from pathlib import Path
from huggingface_hub import cached_assets_path
from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
from transformers import BitsAndBytesConfig as TransformersBnbConfig
from transformers import T5EncoderModel

gpu_name=torch.cuda.get_device_name(0)
total_vram=torch.cuda.get_device_properties(0).total_memory/1024**3
major=torch.cuda.get_device_properties(0).major
dtype=torch.float16

MODEL="black-forest-labs/FLUX.1-schnell"
PROMPT="a futuristic city at sunset, cinematic lighting, ultra detailed"

OUTPUT_DIR=Path("/teamspace/studios/this_studio/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run(pipe,steps=2,h=1024,w=1024,max_sequence=128):
    t0=time.perf_counter()
    with torch.no_grad():
        out = pipe(PROMPT, num_inference_steps=steps,guidance_scale=0.0, height=h, width=w,output_type="pil",
                       max_sequence_length=max_sequence)

    lat=(time.perf_counter()-t0)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    vram=torch.cuda.max_memory_allocated()/1024**3
    print(f"{lat:.2f}s,{vram:.2f}GB")
    img = out.images[0]
    img_path = OUTPUT_DIR/f"image.png"
    img.save(img_path)

t0 = time.perf_counter()
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
    MODEL,
    subfolder="transformer",
    quantization_config=bnb_transformer,
    torch_dtype=dtype,
)
text_encoder_2 = T5EncoderModel.from_pretrained(
    MODEL,
    subfolder="text_encoder_2",
    quantization_config=bnb_text,
    torch_dtype=dtype,
)
pipe = FluxPipeline.from_pretrained(
    MODEL,
    transformer=transformer,
    text_encoder_2=text_encoder_2,
    torch_dtype=dtype,
)

pipe.vae.to("cuda")
pipe.text_encoder.to("cuda")
pipe.set_progress_bar_config(disable=True)
print(f"{time.perf_counter()-t0:.1f}s\n")

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
run(pipe)

# model load time=62 sec
# inference time=13.26sec
# vram=11.85GB