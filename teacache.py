import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import cv2
import numpy as np
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

login(token="HFTOKEN")

MODEL_ID         = "black-forest-labs/FLUX.1-Kontext-dev"
OUTPUT_DIR       = Path("/teamspace/studios/this_studio/output")
DTYPE            = torch.bfloat16
DEFAULT_STEPS    = 4
DEFAULT_GUIDANCE = 3.5
DEFAULT_HEIGHT   = 1024
DEFAULT_WIDTH    = 1024
MAX_SEQ          = 256
GPU_MAIN         = "cuda:0"

# Senior's polynomial coefficients — fitted on modulated norm1 input difference
POLY_COEFFS = np.array([
    4.98651651e+02,
   -2.83781631e+02,
    5.58554382e+01,
   -3.82021401e+00,
    2.64230861e-01,
])
RESCALE_FN = np.poly1d(POLY_COEFFS)

REL_L1_THRESH = 0.25

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark        = True
torch.backends.cudnn.deterministic    = False
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)


# ── TeaCache patch ─────────────────────────────────────────────────────────────
def apply_teacache(pipe, num_steps: int, rel_l1_thresh: float):
    transformer = getattr(pipe.transformer, "_orig_mod", pipe.transformer)

    transformer.tc_cnt                       = 0
    transformer.tc_num_steps                 = num_steps
    transformer.tc_rel_l1_thresh             = rel_l1_thresh
    transformer.tc_accumulated_rel_l1        = 0.0
    transformer.tc_previous_modulated_input  = None
    transformer.tc_previous_residual         = None
    transformer.tc_skip_count                = 0
    transformer.tc_enabled                   = True

    original_forward = transformer.forward

    def teacache_forward(
        hidden_states,
        encoder_hidden_states=None,
        pooled_projections=None,
        timestep=None,
        img_ids=None,
        txt_ids=None,
        guidance=None,
        joint_attention_kwargs=None,
        return_dict=True,
        **kwargs,
    ):
        temb = transformer.time_text_embed(timestep, guidance, pooled_projections)

        inp          = transformer.x_embedder(hidden_states)
        temb_        = temb.clone()

        modulated_inp, _, _, _, _ = transformer.transformer_blocks[0].norm1(
            inp, emb=temb_
        )

        cnt       = transformer.tc_cnt
        num_steps = transformer.tc_num_steps

        if cnt == 0 or cnt == num_steps - 1:
            should_calc = True
            transformer.tc_accumulated_rel_l1 = 0.0
        else:
            rel_diff = (
                (modulated_inp - transformer.tc_previous_modulated_input).abs().mean()
                / transformer.tc_previous_modulated_input.abs().mean()
            ).cpu().item()

            transformer.tc_accumulated_rel_l1 += float(RESCALE_FN(rel_diff))

            if transformer.tc_accumulated_rel_l1 < transformer.tc_rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                transformer.tc_accumulated_rel_l1 = 0.0

        transformer.tc_previous_modulated_input = modulated_inp
        transformer.tc_cnt += 1

        if not should_calc:
            transformer.tc_skip_count += 1
            hidden_states = hidden_states + transformer.tc_previous_residual
            if return_dict:
                from diffusers.models.modeling_outputs import Transformer2DModelOutput
                return Transformer2DModelOutput(sample=hidden_states)
            return (hidden_states,)

        input_hidden_states = hidden_states.clone()

        output = original_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=return_dict,
            **kwargs,
        )

        out_tensor = output.sample if return_dict else output[0]
        transformer.tc_previous_residual = (
            out_tensor.detach().clone() - input_hidden_states.detach().clone()
        )

        return output

    transformer.forward = teacache_forward
    return original_forward


def remove_teacache(pipe, original_forward):
    transformer = getattr(pipe.transformer, "_orig_mod", pipe.transformer)
    transformer.forward          = original_forward
    transformer.tc_enabled       = False
    transformer.tc_cnt           = 0
    transformer.tc_skip_count    = 0
# ──────────────────────────────────────────────────────────────────────────────


def enhance_output(img: Image.Image) -> Image.Image:
    arr       = np.array(img)
    gaussian  = cv2.GaussianBlur(arr, (0, 0), 2.0)
    sharpened = cv2.addWeighted(arr, 1.8, gaussian, -0.8, 0)
    lab       = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
    l, a, b   = cv2.split(lab)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l         = clahe.apply(l)
    enhanced  = cv2.merge([l, a, b])
    enhanced  = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)


def build_pipeline():
    print("Initializing FLUX.1 Kontext  |  NF4 4-bit  |  4 steps  |  TeaCache...")

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

    print(f"  Loading transformer (NF4) → {GPU_MAIN}...")
    initial=time.perf_counter()
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        quantization_config=bnb_transformer,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        device_map=GPU_MAIN,
        use_safetensors=True,
    )

    print(f"  Loading T5 (NF4) → {GPU_MAIN}...")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder_2",
        quantization_config=bnb_t5,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        device_map=GPU_MAIN,
        use_safetensors=True,
    )

    print("  Building pipeline...")
    pipe = FluxKontextPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    print(f"model load time: {time.perf_counter()-initial}")
    pipe.text_encoder    = pipe.text_encoder.to(GPU_MAIN)
    pipe.vae             = pipe.vae.to(GPU_MAIN)
    pipe.vae.use_tiling  = False
    pipe.vae.use_slicing = False
    pipe.set_progress_bar_config(disable=True)

    print("  Compiling transformer...")
    pipe.transformer = torch.compile(
        pipe.transformer,
        fullgraph=False,
    )

    gc.collect()
    torch.cuda.empty_cache()

    print(f"  VAE        : {next(pipe.vae.parameters()).device}")
    print(f"  CLIP       : {next(pipe.text_encoder.parameters()).device}")
    print(f"  Transformer: {next(pipe.transformer.parameters()).device}")
    used = torch.cuda.memory_allocated(0) / 1024**3
    free = torch.cuda.mem_get_info(0)[0] / 1024**3
    print(f"  GPU 0      : {used:.2f} GB used | {free:.2f} GB free")

    run_warmup(pipe)
    return pipe


def run_warmup(pipe):
    dummy   = Image.new("RGB", (DEFAULT_WIDTH, DEFAULT_HEIGHT), color=(128, 128, 128))
    pe, ppe = encode_prompt(pipe, "warmup prompt")

    orig = apply_teacache(pipe, DEFAULT_STEPS, REL_L1_THRESH)

    print("  Warming up...")
    t0 = time.perf_counter()

    with torch.inference_mode():
        result = pipe(
            prompt_embeds=pe,
            pooled_prompt_embeds=ppe,
            image=dummy,
            num_inference_steps=DEFAULT_STEPS,
            guidance_scale=DEFAULT_GUIDANCE,
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
            max_sequence_length=MAX_SEQ,
        )
        _ = enhance_output(result.images[0])

    remove_teacache(pipe, orig)
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    used = torch.cuda.memory_allocated(0) / 1024**3
    free = torch.cuda.mem_get_info(0)[0] / 1024**3
    print(f"  ✓ Warmup done in {elapsed:.1f}s  |  {used:.2f} GB used | {free:.2f} GB free")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("  Ready.\n")


def encode_prompt(pipe, prompt: str):
    with torch.inference_mode():
        pe, ppe, _ = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=GPU_MAIN,
            num_images_per_prompt=1,
            max_sequence_length=MAX_SEQ,
        )
    pe  = pe.to(GPU_MAIN, dtype=DTYPE)
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
    img    = load_input_image(image_source)
    print(f"  ✓ Image loaded     ({time.perf_counter()-t_load:.1f}s)")

    t_txt   = time.perf_counter()
    pe, ppe = encode_prompt(pipe, prompt)
    print(f"  ✓ Prompt encoded   ({time.perf_counter()-t_txt:.2f}s)")

    orig = apply_teacache(pipe, DEFAULT_STEPS, REL_L1_THRESH)

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.perf_counter()
    print(f"  Rendering {DEFAULT_STEPS} steps @ {DEFAULT_WIDTH}×{DEFAULT_HEIGHT}  "
          f"thresh={REL_L1_THRESH}...")

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

    skipped = getattr(pipe.transformer, "_orig_mod", pipe.transformer).tc_skip_count
    remove_teacache(pipe, orig)

    t_enh       = time.perf_counter()
    final_image = enhance_output(result.images[0])
    print(f"  ✓ Enhancement done ({time.perf_counter()-t_enh:.3f}s)")

    torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    peak_0  = torch.cuda.max_memory_allocated(0) / 1024**3

    out_path = OUTPUT_DIR / f"result_{job_index:03d}.png"
    final_image.save(out_path)

    print(f"  Skipped  : {skipped}/{DEFAULT_STEPS} steps ({100*skipped//DEFAULT_STEPS}%)")
    print(f"  Saved    : {out_path}")
    print(f"  Latency  : {latency:.2f}s  |  Peak VRAM: {peak_0:.2f} GB\n")
    return out_path


if __name__ == "__main__":
    pipe      = build_pipeline()
    job_index = 1

    print("\n" + "=" * 60)
    print("  FLUX.1 Kontext  |  L4  |  NF4  |  4 steps  |  TeaCache")
    print(f"  Guidance={DEFAULT_GUIDANCE}  Res={DEFAULT_WIDTH}×{DEFAULT_HEIGHT}  "
          f"thresh={REL_L1_THRESH}  MaxSeq={MAX_SEQ}")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        print(f"\n[ Job #{job_index} ]")

        image_source = input("  Image (URL or local path): ").strip()
        if image_source.lower() in ("quit", "exit"):
            break

        prompt = input("  Prompt: ").strip()
        if not prompt or prompt.lower() in ("quit", "exit"):
            break

        generate(pipe, prompt, image_source, job_index)
        job_index += 1

    print("Session ended. All outputs saved to", OUTPUT_DIR)