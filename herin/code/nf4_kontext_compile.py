import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch, gc, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from PIL import Image
from diffusers import FluxKontextPipeline, FluxTransformer2DModel
from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
from transformers import BitsAndBytesConfig as TransformersBnbConfig
from transformers import T5EncoderModel
from diffusers.utils import load_image
from huggingface_hub import login

login(token="YOUR_TOKEN")

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
OUTPUT_DIR = Path("/teamspace/studios/this_studio/output")
DTYPE = torch.bfloat16
DEFAULT_STEPS = 4
DEFAULT_GUIDANCE = 2.5
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
MAX_SEQ = 64
GPU_MAIN = "cuda:0"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def build_pipeline():
    bnb_transformer = DiffusersBnbConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
        bnb_4bit_use_double_quant=True,
    )
    bnb_t5 = TransformersBnbConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
        bnb_4bit_use_double_quant=True,
    )

    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        quantization_config=bnb_transformer,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        device_map=GPU_MAIN,
        use_safetensors=True,         
    )

    text_encoder_2 = T5EncoderModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder_2",
        quantization_config=bnb_t5,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        device_map=GPU_MAIN,
        use_safetensors=True,          
    )

    pipe = FluxKontextPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True,          
    )

    pipe.text_encoder = pipe.text_encoder.to(GPU_MAIN)
    pipe.vae = pipe.vae.to(GPU_MAIN)
    pipe.vae.use_tiling = False
    pipe.vae.use_slicing = False
    pipe.set_progress_bar_config(disable=True)

    pipe.transformer = torch.compile(
        pipe.transformer,
        mode="reduce-overhead",
        fullgraph=False,
    )

    gc.collect()
    torch.cuda.empty_cache()

    used = torch.cuda.memory_allocated(0) / 1024**3
    free = torch.cuda.mem_get_info(0)[0] / 1024**3
    print(f"  GPU 0: {used:.2f} GB used | {free:.2f} GB free")

    run_warmup(pipe)
    return pipe


def run_warmup(pipe):
    dummy = Image.new("RGB", (DEFAULT_WIDTH, DEFAULT_HEIGHT), color=(128, 128, 128))
    pe, ppe = encode_prompt(pipe, "warmup prompt")

    t0 = time.perf_counter()
    with torch.inference_mode():
        pipe(
            prompt_embeds=pe,
            pooled_prompt_embeds=ppe,
            image=dummy,
            num_inference_steps=DEFAULT_STEPS,
            guidance_scale=DEFAULT_GUIDANCE,
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
            max_sequence_length=MAX_SEQ,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    used = torch.cuda.memory_allocated(0) / 1024**3
    free = torch.cuda.mem_get_info(0)[0] / 1024**3
    print(f"  Warmup finished in {elapsed:.1f}s  |  GPU 0: {used:.2f} GB used | {free:.2f} GB free")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def encode_prompt(pipe, prompt: str):
    with torch.inference_mode():
        pe, ppe, _ = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=GPU_MAIN,
            num_images_per_prompt=1,
            max_sequence_length=MAX_SEQ,
        )
    pe  = pe.to(GPU_MAIN,  dtype=DTYPE)
    ppe = ppe.to(GPU_MAIN, dtype=DTYPE)
    return pe, ppe


def load_input_image(source: str) -> Image.Image:
    if source.startswith("http"):
        img = load_image(source).convert("RGB")
    else:
        img = Image.open(source).convert("RGB")
    return img.resize((DEFAULT_WIDTH, DEFAULT_HEIGHT), Image.LANCZOS)


def generate(pipe, prompt: str, image_source: str, job_index: int) -> Path:
    t_load = time.perf_counter()
    img = load_input_image(image_source)
    print(f"Image loaded ({time.perf_counter()-t_load:.1f}s)")

    t_txt = time.perf_counter()
    pe, ppe = encode_prompt(pipe, prompt)
    print(f"Prompt encoded ({time.perf_counter()-t_txt:.2f}s)")

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.perf_counter()

    with torch.inference_mode():
        result = pipe(
            prompt_embeds=pe,
            pooled_prompt_embeds=ppe,
            image=img,
            num_inference_steps=DEFAULT_STEPS,
            guidance_scale=DEFAULT_GUIDANCE,
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
            max_sequence_length=MAX_SEQ,
        )

    torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    peak_0  = torch.cuda.max_memory_allocated(0) / 1024**3

    out_path = OUTPUT_DIR / f"result_{job_index:03d}.png"
    result.images[0].save(out_path)

    print(f"Saved : {out_path}")
    print(f"Latency : {latency:.2f}s | Peak VRAM: {peak_0:.2f} GB\n")
    return out_path


if __name__ == "__main__":
    pipe = build_pipeline()
    job_index = 1

    while True:
        print(f"\n[ Job #{job_index} ]")
        image_source = input("Image (URL or local path): ").strip()
        if image_source.lower() in ("quit", "exit"):
            break
        prompt = input("Prompt: ").strip()
        if not prompt or prompt.lower() in ("quit", "exit"):
            break
        generate(pipe, prompt, image_source, job_index)
        job_index += 1

    print("Session ended. All outputs saved to", OUTPUT_DIR)

# compile time=160sec
# inference time=13.54 sec
# vram=12.36GB