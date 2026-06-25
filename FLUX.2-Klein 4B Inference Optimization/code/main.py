import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, TorchAoConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import hf_hub_download
from torchao.quantization import Int8WeightOnlyConfig
from PIL import Image
import time
from pathlib import Path

dtype = torch.bfloat16
Model_Id="black-forest-labs/FLUX.2-klein-4B"

device = "cuda"
flux_transformer = Flux2Transformer2DModel.from_pretrained(
    Model_Id,
    subfolder="transformer",
    torch_dtype=dtype,
    quantization_config=TorchAoConfig(Int8WeightOnlyConfig()),
    device_map=device,
)
text_encoder = AutoModelForCausalLM.from_pretrained(
    Model_Id,
    subfolder="text_encoder",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    torch_dtype=dtype,
    device_map=device,
)
pipe = Flux2KleinPipeline.from_pretrained(
    Model_Id,
    transformer=flux_transformer,
    text_encoder=text_encoder,
    torch_dtype=dtype,
)
pipe.to(device)

# --- Config ---
prompt = "A futuristic city at sunset, cyberpunk style"
# --- Warm-up stage ---
with torch.no_grad():
    _ = pipe(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=2,
        guidance_scale=0.0
    )

# --- Inference with timing + VRAM ---
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

t0 = time.perf_counter()
with torch.no_grad():
    image = pipe(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=2,
        guidance_scale=0.0
    ).images[0]
lat = time.perf_counter() - t0

vram = torch.cuda.max_memory_allocated() / 1024**3
print(f"Inference time: {lat:.2f}s | Peak VRAM: {vram:.2f} GB")

# --- Save output ---
image.save("output.png")
print("Image saved to output.png")