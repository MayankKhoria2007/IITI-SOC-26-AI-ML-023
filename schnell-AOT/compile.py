
from huggingface_hub import login
login("HF_token")
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers import TorchAoConfig as TransformerConfig
from torchao.quantization import Int8WeightOnlyConfig

from transformers import  T5EncoderModel

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

transformer_quantization_config=TransformerConfig(quant_type=Int8WeightOnlyConfig())


transformer=FluxTransformer2DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    quantization_config=transformer_quantization_config,
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
    captured = {}

    def hook(module, args, kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    handle = pipe.transformer.register_forward_pre_hook(
        hook,
    with_kwargs=True
    )

    _ = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=2,
        guidance_scale=0.0,
        height=1024, width=1024,
    )
    from torchao.utils import unwrap_tensor_subclass
    handle.remove()
    unwrap_tensor_subclass(pipe.transformer)
    torch.cuda.synchronize()
    import gc; gc.collect()
    torch.cuda.empty_cache()
   
    
    exported = torch.export.export(
        pipe.transformer, 
        args=captured["args"], 
        kwargs=captured["kwargs"],
        strict=False  
    )
    output_path = torch._inductor.aoti_compile_and_package(
    exported,
    package_path=os.path.join(os.getcwd(), "model.pt2")
    )
    torch.save({"args": captured["args"], "kwargs": captured["kwargs"]}, "captured_inputs.pt")

torch.cuda.synchronize()
print("Warmup done")

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
start=time.perf_counter()
del pipe.transformer
del exported
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

#with torch.inference_mode():
#    start=time.perf_counter()
#    compiled_transformer = torch._inductor.aoti_load_package(os.path.join(os.getcwd(), "model.pt2"))
#
#    outputs = compiled_transformer(*captured["args"], **captured["kwargs"])
#    latents = outputs[0]
#    
#   
#    

#torch.cuda.synchronize()
#latency=time.perf_counter()-start
#peak_vram=torch.cuda.max_memory_allocated() / (1024 ** 3)
#
#post_processed_image.save(OUTPUT_DIR / "image_torchao2.png")
#print(f"\nInference Time : {latency:.2f}s")
#print(f"Peak VRAM      : {peak_vram:.2f} GB")
