import os
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from PIL import Image
import gc
import torch
from torchao.quantization import Int8WeightOnlyConfig
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, TorchAoConfig
from diffusers.utils import load_image

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_ID         = "black-forest-labs/FLUX.1-Kontext-dev"
OUTPUT_DIR       = Path("./output")
DTYPE            = torch.bfloat16
DEFAULT_STEPS    = 6
DEFAULT_GUIDANCE = 4.5
DEFAULT_HEIGHT   = 1024
DEFAULT_WIDTH    = 1024
WARMUP_STEPS     = 2
# ─────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)


# Override _execution_device so the pipeline always routes tensors to CUDA
class CudaFluxKontextPipeline(FluxKontextPipeline):
    @property
    def _execution_device(self):
        return torch.device("cuda:0")


def build_pipeline():
    print("Initializing FLUX.1 Kontext Pipeline (INT8 GPU Native)...")
    t0 = time.perf_counter()

    # INT8 quantization via TorchAoConfig — works without fbgemm
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        quantization_config=TorchAoConfig(quant_type=Int8WeightOnlyConfig()),
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )

    pipe = CudaFluxKontextPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to("cuda")

    pipe.set_progress_bar_config(disable=True)

    print(f"Pipeline loaded in {time.perf_counter() - t0:.2f}s")

    print("Running warm-up pass...")
    dummy = Image.new("RGB", (DEFAULT_WIDTH, DEFAULT_HEIGHT), color=(128, 128, 128))
    with torch.inference_mode():
        embeds, pooled, _ = pipe.encode_prompt(
            prompt="warm-up", prompt_2=None, max_sequence_length=256
        )
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        
        pipe(
            prompt_embeds=embeds,
            pooled_prompt_embeds=pooled,
            image=dummy,
            num_inference_steps=WARMUP_STEPS,
            guidance_scale=DEFAULT_GUIDANCE,
            max_sequence_length=256,
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
        )
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    free_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
    print(f"Warm-up complete. Free VRAM: {free_vram:.2f} GB\n")
    return pipe


def encode_prompt(pipe, prompt: str):
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    
    with torch.inference_mode():
        embeds, pooled, _ = pipe.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=256
        )
        embeds = embeds.to(dtype=DTYPE).clone()
        pooled = pooled.to(dtype=DTYPE).clone()

    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    return embeds, pooled


def load_input_image(source: str) -> Image.Image:
    try:
        if source.startswith("http://") or source.startswith("https://"):
            img = load_image(source).convert("RGB")
        else:
            img = Image.open(source).convert("RGB")
        img = img.resize((DEFAULT_WIDTH, DEFAULT_HEIGHT), Image.LANCZOS)
        return img
    except Exception as e:
        print(f"  Warning: Could not load image ({e}). Using grey placeholder.")
        return Image.new("RGB", (DEFAULT_WIDTH, DEFAULT_HEIGHT), color=(128, 128, 128))


def generate(pipe, prompt: str, image_source: str, job_index: int) -> Path:
    input_image  = load_input_image(image_source)
    prompt_embeds, pooled_prompt_embeds = encode_prompt(pipe, prompt)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    with torch.inference_mode():
        result = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            image=input_image,
            num_inference_steps=DEFAULT_STEPS,
            guidance_scale=DEFAULT_GUIDANCE,
            max_sequence_length=256,
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
        )

    torch.cuda.synchronize()
    latency   = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    out_path = OUTPUT_DIR / f"result_{job_index:03d}.png"
    result.images[0].save(out_path)

    print(f"  Saved   : {out_path}")
    print(f"  Latency : {latency:.2f}s  |  Peak VRAM: {peak_vram:.2f} GB")
    return out_path


# ── Interactive loop ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipe      = build_pipeline()
    job_index = 1

    print("=" * 52)
    print("  FLUX.1 Kontext — Interactive Generation Loop")
    print("  INT8 GPU Native edition")
    print("  Type 'quit' or 'exit' at any prompt to stop.")
    print("=" * 52)

    while True:
        print(f"\n[ Job #{job_index} ]")

        image_source = input("  Image (URL or local path): ").strip()
        if image_source.lower() in ("quit", "exit"):
            break

        prompt = input("  Prompt: ").strip()
        if prompt.lower() in ("quit", "exit"):
            break

        if not prompt:
            print("  Prompt cannot be empty. Try again.")
            continue

        print("  Rendering...")
        generate(pipe, prompt, image_source, job_index)
        job_index += 1

    print("\nSession ended. All outputs saved to ./output/")
