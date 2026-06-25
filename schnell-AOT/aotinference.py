from huggingface_hub import login
login("HF_TOKEN")
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from pathlib import Path
import torch
from diffusers import FluxPipeline

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()
dtype = torch.bfloat16
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
OUTPUT_DIR = Path("/teamspace/studios/this_studio/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


pipe = FluxPipeline.from_pretrained(
    MODEL_ID, 
    transformer=None,
    torch_dtype=dtype, 
    low_cpu_mem_usage=True
)
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
pipe.set_progress_bar_config(disable=True)
pipe.to("cuda")

# Reload saved input tensors and compiled package
inputs = torch.load("captured_inputs.pt", weights_only=False)
compiled_transformer = torch._inductor.aoti_load_package(os.path.join(os.getcwd(), "model.pt2"))


torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()

with torch.inference_mode():
    # Execute native Triton compiled hardware graph
    outputs = compiled_transformer(*inputs["args"], **inputs["kwargs"])
   #INTEGRATING COMPILED TRANSFORMER IN PIPELINE FOR IMAGE OUTPUTS IS TO BE CONTINUED 

   
   

torch.cuda.synchronize()
print(f"Inference Time : {time.perf_counter() - start:.2f}s")
print(f"Peak VRAM      : {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")

#vram-11.24 gb
#time for running transformer on raw input:2.5s