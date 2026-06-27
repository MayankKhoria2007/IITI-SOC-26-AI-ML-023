import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import torch
import requests
import gc
import warnings
from PIL import Image, ImageDraw
from huggingface_hub import login
from diffusers import FluxFillPipeline, FluxTransformer2DModel, TorchAoConfig

warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)

HF_TOKEN = "YOUR_TOKEN"
login(token=HF_TOKEN)

MODEL_ID="black-forest-labs/FLUX.1-Fill-dev"
link="https://images.unsplash.com/photo-1616594039964-ae9021a400a0?q=80&w=1000&auto=format&fit=crop"
image=Image.open(requests.get(link,stream=True).raw).convert("RGB")
image=image.resize((512, 512))

mask=Image.new("L",(512, 512),color=0)
draw=ImageDraw.Draw(mask)
draw.rectangle([100, 100, 412, 412],fill=255)

EDIT_TASK = (
    "Change the color of the white bed bedding sheets to a rich, elegant midnight blue color."
)

quantization_config = TorchAoConfig(quant_type="int4_weight_only")

transformer_load_start = time.perf_counter()
transformer = FluxTransformer2DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    low_cpu_mem_usage=True,
)
transformer_load_time = time.perf_counter() - transformer_load_start
print(f"Transformer load time: {transformer_load_time:.2f} seconds")

pipe_load_start = time.perf_counter()
pipe = FluxFillPipeline.from_pretrained(
    MODEL_ID,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).to("cuda")
pipe_load_time = time.perf_counter() - pipe_load_start
print(f"Pipeline load time:    {pipe_load_time:.2f} seconds")
print(f"Total model load time: {transformer_load_time + pipe_load_time:.2f} seconds")

pipe.enable_vae_slicing()            
pipe.enable_vae_tiling() 

compile_start=time.perf_counter()
pipe.transformer = torch.compile(
    pipe.transformer,
    mode="reduce-overhead",
    fullgraph=False,          
)
compile_load_time=time.perf_counter()-compile_start
print(f"Compilation time: {compile_load_time} seconds")

with torch.inference_mode():
    prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=EDIT_TASK,
        prompt_2=EDIT_TASK,
        max_sequence_length=128
    )

pipe.text_encoder=None
pipe.text_encoder_2=None
gc.collect()
torch.cuda.empty_cache()    

with torch.inference_mode():
    _ = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        image=image,
        mask_image=mask,
        num_inference_steps=2,
        guidance_scale=30.0,
        height=512, 
        width=512,
    )
torch.cuda.synchronize()
print("Warmup done")

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start_time = time.perf_counter()

with torch.inference_mode():
    final_output = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        image=image,
        mask_image=mask,
        num_inference_steps=8,       
        guidance_scale=30.0,
        height=512,
        width=512,
    ).images[0]

torch.cuda.synchronize()
end_time=time.perf_counter()

latency=end_time-start_time
peak_vram=torch.cuda.max_memory_allocated()/(1024 ** 3)

print("\nINFERENCE COMPLETION METRICS")
print(f" Total Generation Time: {latency:.2f} seconds")
print(f" Peak VRAM Utilized: {peak_vram:.2f} GB")

output_filename = "flux_fill.png"
final_output.save(output_filename)
print(f"Image successfully processed and saved to: {output_filename}")


# load time=167 sec
# compile=5 min 30 sec
# infernce time=14.54 sec
# vram=15.1GB
