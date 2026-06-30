from __future__ import annotations

import os
# Configure global execution environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import logging
import gc
from io import BytesIO
from pathlib import Path
import numpy as np
import modal

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

cuda_env = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TORCH_BLAS_PREFER_HIP": "0",
    "HF_HOME": "/cache/huggingface",  # Directs Hugging Face to use our persistent volume storage
}

app = modal.App("flux-kontext-cloudinary-teacache-service")

# Look up or create the persistent cache storage volume
volume = modal.Volume.from_name("flux-weights-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "fastapi[standard]", 
        "torch",
        "transformers",
        "diffusers",
        "accelerate",
        "bitsandbytes",
        "sentencepiece",
        "protobuf",
        "peft",
        "huggingface_hub",
        "pillow",
        "opencv-python",
        "numpy",
        "cloudinary"
    )
)

# ---------------------------------------------------------
# SERVERLESS ENGINE CLASS WITH INTEGRATED OPTIMIZATIONS
# ---------------------------------------------------------
@app.cls(
    image=image,
    gpu="L4",             
    startup_timeout=1500,          
    env=cuda_env,         
    secrets=[modal.Secret.from_name("flux-kontext-secrets")],
    volumes={"/cache": volume},  
    min_containers=0,
    scaledown_window=100,
    max_containers=1
)
class FluxKontextEngine:
    @modal.enter()
    def initialize_pipeline(self):
        import torch
        from diffusers import FluxKontextPipeline, FluxTransformer2DModel
        from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
        from transformers import BitsAndBytesConfig as TransformersBnbConfig
        from transformers import T5EncoderModel
        import cloudinary
        from huggingface_hub import login

        hf_token = os.environ.get("HF_TOKEN", "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")
        login(token=hf_token)

        cloudinary.config(
            cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
            api_key=os.environ.get("CLOUDINARY_API_KEY"),
            api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
            secure=True
        )

        self.MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
        self.DTYPE = torch.bfloat16
        self.GPU_MAIN = "cuda:0"
        self.MAX_SEQ = 256  
        self.DEFAULT_WIDTH = 1024
        self.DEFAULT_HEIGHT = 1024
        self.DEFAULT_STEPS = 4
        self.DEFAULT_GUIDANCE = 3.5
        self.REL_L1_THRESH = 0.25
        
        # Guard flag for runtime compilation execution 
        self.is_compiled = False

        self.POLY_COEFFS = np.array([
            4.98651651e+02,
           -2.83781631e+02,
            5.58554382e+01,
           -3.82021401e+00,
            2.64230861e-01,
        ])
        self.RESCALE_FN = np.poly1d(self.POLY_COEFFS)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

        print("⚡ Loading quantized models layers (Reading from persistent Volume Cache)...")
        bnb_transformer = DiffusersBnbConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.DTYPE,
            bnb_4bit_use_double_quant=True,
        )
        bnb_t5 = TransformersBnbConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.DTYPE,
            bnb_4bit_use_double_quant=True,
        )

        transformer = FluxTransformer2DModel.from_pretrained(
            self.MODEL_ID, subfolder="transformer", quantization_config=bnb_transformer,
            torch_dtype=self.DTYPE, low_cpu_mem_usage=True, device_map=self.GPU_MAIN, use_safetensors=True,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            self.MODEL_ID, subfolder="text_encoder_2", quantization_config=bnb_t5,
            torch_dtype=self.DTYPE, low_cpu_mem_usage=True, device_map=self.GPU_MAIN, use_safetensors=True,
        )

        print("Building full pipeline...")
        self.pipe = FluxKontextPipeline.from_pretrained(
            self.MODEL_ID, transformer=transformer, text_encoder_2=text_encoder_2,
            torch_dtype=self.DTYPE, low_cpu_mem_usage=True, use_safetensors=True,
        )

        volume.commit()

        self.pipe.text_encoder = self.pipe.text_encoder.to(self.GPU_MAIN)
        self.pipe.vae = self.pipe.vae.to(self.GPU_MAIN)
        self.pipe.vae.use_tiling = False
        self.pipe.vae.use_slicing = False
        self.pipe.set_progress_bar_config(disable=True)

        gc.collect()
        torch.cuda.empty_cache()

    def _apply_teacache(self):
        """Patches the model forward loop to activate TeaCache computation reduction."""
        transformer = getattr(self.pipe.transformer, "_orig_mod", self.pipe.transformer)

        transformer.tc_cnt = 0
        transformer.tc_num_steps = self.DEFAULT_STEPS
        transformer.tc_rel_l1_thresh = self.REL_L1_THRESH
        transformer.tc_accumulated_rel_l1 = 0.0
        transformer.tc_previous_modulated_input = None
        transformer.tc_previous_residual = None
        transformer.tc_skip_count = 0
        transformer.tc_enabled = True

        rescale_fn = self.RESCALE_FN

        def check_teacache_skip(tf_mod, modulated_inp):
            cnt = tf_mod.tc_cnt
            num_steps = tf_mod.tc_num_steps

            if cnt == 0 or cnt == num_steps - 1:
                tf_mod.tc_accumulated_rel_l1 = 0.0
                return True  # should_calc

            rel_diff = (
                (modulated_inp - tf_mod.tc_previous_modulated_input).abs().mean()
                / tf_mod.tc_previous_modulated_input.abs().mean()
            ).cpu().item()

            tf_mod.tc_accumulated_rel_l1 += float(rescale_fn(rel_diff))

            if tf_mod.tc_accumulated_rel_l1 < tf_mod.tc_rel_l1_thresh:
                return False  # should_calc
            else:
                tf_mod.tc_accumulated_rel_l1 = 0.0
                return True  # should_calc

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
            # 🔥 FIX: Instruct the Inductor compilation backend that a fresh graph iteration step is starting
            import torch
            torch.compiler.cudagraph_mark_step_begin()

            temb = transformer.time_text_embed(timestep, guidance, pooled_projections)
            inp = transformer.x_embedder(hidden_states)
            temb_ = temb.clone()

            modulated_inp, _, _, _, _ = transformer.transformer_blocks[0].norm1(
                inp, emb=temb_
            )

            should_calc = check_teacache_skip(transformer, modulated_inp)

            transformer.tc_previous_modulated_input = modulated_inp.detach().clone()
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

    def _remove_teacache(self, original_forward):
        """Unpatches the forward pass logic to safely restore clean model state."""
        transformer = getattr(self.pipe.transformer, "_orig_mod", self.pipe.transformer)
        transformer.forward = original_forward
        transformer.tc_enabled = False
        transformer.tc_cnt = 0
        transformer.tc_skip_count = 0

    def _compile_and_warmup(self):
        """Compiles the individual block subcomponents to avoid top-level layout crashes."""
        import torch
        from PIL import Image
        
        print("⚙️ Compiling underlying transformer blocks via Inductor...")
        transformer = getattr(self.pipe.transformer, "_orig_mod", self.pipe.transformer)
        for i in range(len(transformer.transformer_blocks)):
            transformer.transformer_blocks[i] = torch.compile(
                transformer.transformer_blocks[i], 
                mode="reduce-overhead", 
                fullgraph=False
            )
        
        dummy = Image.new("RGB", (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT), color=(128, 128, 128))
        pe, ppe = self._encode_prompt("warmup prompt")

        orig = self._apply_teacache()

        print("🔥 Running live isolated block kernel warmup pass...")
        with torch.inference_mode():
            result = self.pipe(
                prompt_embeds=pe, pooled_prompt_embeds=ppe, image=dummy,
                num_inference_steps=self.DEFAULT_STEPS, guidance_scale=self.DEFAULT_GUIDANCE,
                height=self.DEFAULT_HEIGHT, width=self.DEFAULT_WIDTH, max_sequence_length=self.MAX_SEQ,
            )
            _ = self._enhance_output(result.images[0])
            
        self._remove_teacache(orig)
        torch.cuda.synchronize()
        
        gc.collect()
        torch.cuda.empty_cache()
        self.is_compiled = True
        print("🎉 SERVERLESS ENGINE RUNNING WITH COMPILED BLOCKS - HOT AND READY!\n")

    def _encode_prompt(self, prompt: str):
        import torch
        with torch.inference_mode():
            pe, ppe, _ = self.pipe.encode_prompt(
                prompt=prompt, prompt_2=None, device=self.GPU_MAIN,
                num_images_per_prompt=1, max_sequence_length=self.MAX_SEQ,
            )
        return pe.to(self.GPU_MAIN, dtype=self.DTYPE), ppe.to(self.GPU_MAIN, dtype=self.DTYPE)

    def _enhance_output(self, img) -> Image.Image:
        import cv2
        import numpy as np
        from PIL import Image
        arr = np.array(img)
        gaussian = cv2.GaussianBlur(arr, (0, 0), 2.0)
        sharpened = cv2.addWeighted(arr, 1.8, gaussian, -0.8, 0)
        lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)

    @modal.method()
    def generate(self, prompt: str, image_url: str) -> dict:
        import torch
        from diffusers.utils import load_image
        import cloudinary.uploader

        # Trigger localized compilation lazily on first call
        if not self.is_compiled:
            self._compile_and_warmup()

        print(f"🎬 Active Generation Request Received: '{prompt}'")
        t_load = time.perf_counter()
        img = load_image(image_url).convert("RGB").resize((self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT))
        print(f"  ✓ Image resolved ({time.perf_counter()-t_load:.2f}s)")

        t_txt = time.perf_counter()
        pe, ppe = self._encode_prompt(prompt)
        print(f"  ✓ Prompt encoded  ({time.perf_counter()-t_txt:.2f}s)")

        orig = self._apply_teacache()
        torch.cuda.reset_peak_memory_stats(0)

        with torch.inference_mode():
            result = self.pipe(
                prompt_embeds=pe, pooled_prompt_embeds=ppe, image=img,
                num_inference_steps=self.DEFAULT_STEPS, guidance_scale=self.DEFAULT_GUIDANCE,
                height=self.DEFAULT_HEIGHT, width=self.DEFAULT_WIDTH, max_sequence_length=self.MAX_SEQ,
            )

        transformer_ctx = getattr(self.pipe.transformer, "_orig_mod", self.pipe.transformer)
        skipped = transformer_ctx.tc_skip_count
        self._remove_teacache(orig)

        final_image = self._enhance_output(result.images[0])
        buffer = BytesIO()
        final_image.save(buffer, format="PNG")
        buffer.seek(0)

        print(f"  ✓ Skipped: {skipped}/{self.DEFAULT_STEPS} steps ({100*skipped//self.DEFAULT_STEPS}%)")
        print("📤 Uploading assets directly to Cloudinary...")
        upload_result = cloudinary.uploader.upload(buffer, folder="flux-kontext-outputs", resource_type="image")

        return {
            "status": "success",
            "cloudinary_url": upload_result.get("secure_url"),
            "public_id": upload_result.get("public_id"),
            "steps_skipped": skipped
        }

# ---------------------------------------------------------
# FASTAPI ROUTING INTERFACE WITH LIVE /DOCS
# ---------------------------------------------------------
from fastapi import FastAPI
from pydantic import BaseModel

class GenerationRequest(BaseModel):
    prompt: str
    image_url: str

web_app = FastAPI(title="FLUX Kontext Image-to-Image Service Docs")

@web_app.post("/")
def run_generation(payload: GenerationRequest):
    """Trigger image editing via FLUX Kontext with TeaCache acceleration"""
    engine = FluxKontextEngine()
    return engine.generate.remote(prompt=payload.prompt, image_url=payload.image_url)

@app.function(image=image, timeout=1500)  
@modal.asgi_app()                        
def api_generate():
    return web_app
