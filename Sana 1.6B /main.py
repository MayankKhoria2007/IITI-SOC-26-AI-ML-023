# ==============================================================================
# SANA Optimization Suite
# ==============================================================================
# This script executes NVIDIA's ultra-fast SANA model natively on a Tesla T4.
# Due to its extremely lightweight 1.6 Billion parameter size, we do not need
# to rely on heavy quantizers. It will load purely in bfloat16 and execute 
# blazingly fast.
# ==============================================================================

import os
import time
import gc
import torch
import argparse
from huggingface_hub import login
from diffusers import SanaPipeline

# ------------------------------------------------------------------------------
# 1. Environment Profiling
# ------------------------------------------------------------------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please enable a GPU runtime.")

gpu_name = torch.cuda.get_device_name(0)
print("=" * 80)
print(f"HARDWARE PROFILE: {gpu_name}")
print("=" * 80)

# ------------------------------------------------------------------------------
# 2. Authentication
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--token", type=str, default=None)
args, unknown = parser.parse_known_args()

hf_token = args.token or os.environ.get("HF_TOKEN")
if hf_token:
    print("Logging into Hugging Face Hub using provided token...")
    login(token=hf_token)

# ------------------------------------------------------------------------------
# 3. Model Loading
# ------------------------------------------------------------------------------
print("\nLoading NVIDIA SANA (1.6B parameters)...")
print("This will fit perfectly into the T4's 16GB VRAM in native bfloat16.")

# Load the pipeline in fp16 (T4 GPUs do not support bfloat16 natively)
pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    variant="bf16",
    torch_dtype=torch.float16,
)

# Ensure text encoder and VAE stay in float16
pipe.text_encoder.to(torch.float16)
pipe.vae.to(torch.float16)

# Lock everything completely into VRAM for raw speed (NO CPU OFFLOADING)
print("Locking all models directly into VRAM for maximum PCIe speed...")
pipe.to("cuda")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# ------------------------------------------------------------------------------
# 4. Micro-Optimizations
# ------------------------------------------------------------------------------
# We skip channels_last memory formatting as it proved to bottleneck the older T4 GPU.

print("\nEnabling CUDA Graphs via torch.compile (max-autotune)...")
print("Note: This will take ~15-20 minutes to compile on the first run.")
import torch._inductor.config
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
# ------------------------------------------------------------------------------
# 5. Execution & Benchmarking
# ------------------------------------------------------------------------------
test_prompt = "A highly detailed, candid close-up portrait of a woman in her late 20s with freckles, soft curly auburn hair, and warm amber eyes. She is wearing a cozy, cream-colored knit sweater, looking slightly away from the camera with a gentle, thoughtful expression. The lighting is soft and natural, coming from a nearby window during the golden hour, creating a warm glow. The background is a blurred, cozy living room. Photographed on a 50mm lens with a shallow depth of field, sharp focus on the eyes, cinematic, photorealistic."

print("\nExecuting Warmup Generation (Step 1/2)...")
warmup_start = time.time()
_ = pipe(
    prompt=test_prompt,
    width=512, # Dropped from 1024 to prevent OOM without CPU offloading
    height=512,
    guidance_scale=5.0, # SANA uses CFG
    num_inference_steps=7, # Sprint generation (1-4 steps) for ultra-fast speed
    generator=torch.Generator("cuda").manual_seed(42)
)
warmup_time = time.time() - warmup_start
print(f"Warmup Run Completed in: {warmup_time:.2f} seconds.")

print("\nExecuting Optimized Benchmark Generation (Step 2/2)...")
gc.collect()
torch.cuda.empty_cache()

benchmark_start = time.time()
output = pipe(
    prompt=test_prompt,
    width=512, # 512x512 natively in VRAM
    height=512,
    guidance_scale=5.0,
    num_inference_steps=7, 
    generator=torch.Generator("cuda").manual_seed(42)
)
benchmark_time = time.time() - benchmark_start

print("=" * 80)
print("BENCHMARK COMPLETE")
print(f"SANA Inference Time: {benchmark_time:.2f} seconds")
print("=" * 80)

output_filename = "cyberpunk_sana.png"
output.images[0].save(output_filename)
print(f"\nSaved beautiful generated image to: '{output_filename}'")
