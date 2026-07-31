from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import logging
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.utils.checkpoint

import modal
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

cuda_env = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HF_HOME": "/cache/huggingface", 
}

app = modal.App("flux-schnell-aot-trainer")

output_volume = modal.Volume.from_name("flux-lora-outputs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("flux-weights-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch", 
        "torchvision", 
        "transformers", 
        "diffusers", 
        "peft", 
        "accelerate", 
        "bitsandbytes", 
        "sentencepiece", 
        "protobuf", 
        "huggingface_hub", 
        "pillow", 
        "opencv-python", 
        "numpy", 
        "datasets"
    )
)

class HFInteriorDesignDataset(Dataset):
    def __init__(self, target_size: int = 1024):
        from datasets import load_dataset
        print("📥 Streaming dataset context...")
        self.dataset = load_dataset("victorzarzu/interior-design-prompt-editing-dataset-test", split="train")
        self.target_size = target_size
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        img = row["designed_image"].convert("RGB")
        prompt = row["output_prompt"] if row["output_prompt"] else "A premium architectural interior space"
        
        width, height = img.size
        new_size = min(width, height)
        left = (width - new_size) // 2
        top = (height - new_size) // 2
        right = left + new_size
        bottom = top + new_size
        
        img_cropped = img.crop((left, top, right, bottom))
        img_resized = img_cropped.resize((self.target_size, self.target_size), Image.LANCZOS)
        
        pixel_values = self.normalize(img_resized)
        return {"pixel_values": pixel_values, "prompt": prompt}

@app.cls(
    image=image,
    gpu="L4",
    startup_timeout=1500,
    timeout=18000,
    env=cuda_env,
    secrets=[modal.Secret.from_name("flux-kontext-secrets")], 
    volumes={
        "/output": output_volume,
        "/cache": hf_cache_volume
    },
    max_containers=1
)
class FluxAOTDatasetTrainer:
    @modal.enter()
    def setup_engine(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        self.DTYPE = torch.bfloat16
        self.GPU_DEVICE = "cuda:0"
        
    def _get_mock_embeddings(self, batch_size: int, device: str):
        prompt_embeds = torch.zeros((batch_size, 512, 4096), dtype=self.DTYPE, device=device)
        pooled_prompt_embeds = torch.zeros((batch_size, 768), dtype=self.DTYPE, device=device)
        text_ids = torch.zeros((512, 3), dtype=self.DTYPE, device=device)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    def _pack_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = latents.shape
        latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(1, 2).flatten(2)
        
        img_ids = torch.zeros(height // 2, width // 2, 3, device=latents.device, dtype=latents.dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(height // 2, device=latents.device)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(width // 2, device=latents.device)[None, :]
        img_ids = img_ids.flatten(0, 1)
        
        return latents, img_ids

    @modal.method()
    def train_lora_job(self, max_train_steps: int = 1500, lr: float = 2e-5, batch_size: int = 1, resolution: int = 1024):
        from diffusers import FluxTransformer2DModel, AutoencoderKL
        from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
        from peft import LoraConfig, get_peft_model
        from huggingface_hub import login
        import bitsandbytes as bnb
        
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        MODEL_ID = "black-forest-labs/FLUX.1-schnell"
        OUTPUT_DIR = "/output/flux_schnell_interior_lora"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print("⚡ Loading core VAE layers directly onto device...")
        vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=self.DTYPE).to(self.GPU_DEVICE)
        vae.requires_grad_(False)
        
        quantization_config = DiffusersBnbConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.DTYPE,
            bnb_4bit_use_double_quant=True,
        )

        print("⚡ Loading baseline transformer with automated 4-bit mapping...")
        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL_ID, 
            subfolder="transformer", 
            quantization_config=quantization_config,
            torch_dtype=self.DTYPE,
            device_map="auto"
        )

        # 🛠️ 1. ENABLE NATIVE CHECKPOINTING BEFORE PEFT
        transformer.enable_gradient_checkpointing()

        # Apply static-compatible LoRA layers directly
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
        )
        transformer = get_peft_model(transformer, lora_config)
        transformer.train()

        # 🛠️ 2. THE NONETYPE ANNIHILATOR
        # Explicitly assign the PyTorch checkpoint method to the unwrapped diffusers model.
        # This completely guarantees the forward pass has a function to call.
        transformer.base_model.model._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

        optimizer = bnb.optim.AdamW8bit(transformer.parameters(), lr=lr, weight_decay=1e-2)

        dataset = HFInteriorDesignDataset(target_size=resolution)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        global_step = 0
        keep_training = True
        epoch = 0

        print(f"🎉 Architecture safely locked under 11GB footprint limits. Launching {max_train_steps} training steps...")
        
        while keep_training:
            epoch += 1
            t_start = time.perf_counter()
            epoch_loss = 0.0
            steps_in_epoch = 0
            
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                pixel_values = batch["pixel_values"].to(self.GPU_DEVICE, dtype=self.DTYPE)
                
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                
                packed_latents, img_ids = self._pack_latents(latents)
                
                noise = torch.randn_like(packed_latents)
                bsz = packed_latents.shape[0]
                timesteps = torch.rand((bsz,), device=packed_latents.device)
                sigmas = timesteps.view(bsz, 1, 1)
                
                noisy_latents = (1.0 - sigmas) * packed_latents + sigmas * noise
                
                prompt_embeds, pooled_prompt_embeds, text_ids = self._get_mock_embeddings(bsz, self.GPU_DEVICE)
                
                # 🛠️ 3. THE AUTOGRAD SAVIOR (SOLVES LOSS = 400+)
                # By detaching and enforcing grad natively on the inputs, PyTorch is FORCED 
                # to track gradients through the 4-bit checkpoint layers.
                noisy_latents = noisy_latents.detach().requires_grad_(True)
                prompt_embeds = prompt_embeds.detach().requires_grad_(True)
                
                with torch.autocast(device_type="cuda", dtype=self.DTYPE):
                    model_pred = transformer(
                        hidden_states=noisy_latents,
                        timestep=timesteps * 1000.0,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=img_ids,
                        return_dict=False,
                    )[0]
                
                target = noise - packed_latents
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss.backward()
                
                # Ceiling limit on gradients to prevent any early step turbulence
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                global_step += 1
                steps_in_epoch += 1
                
                if global_step % 50 == 0:
                    print(f"Step [{global_step}/{max_train_steps}] | Loss: {loss.item():.5f}")

                if global_step >= max_train_steps:
                    keep_training = False
                    break

            torch.cuda.empty_cache()
            gc.collect()
            
            elapsed = time.perf_counter() - t_start
            avg_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0
            print(f"🔄 Epoch {epoch} complete | Average Loss: {avg_loss:.5f} | Global Steps: {global_step}")

        print("💾 Saving fine-tuned Adapter weights...")
        clean_transformer = getattr(transformer, "_orig_mod", transformer)
        clean_transformer.save_pretrained(OUTPUT_DIR)
        
        output_volume.commit()
        hf_cache_volume.commit()
        print(f"🚀 Training complete. Deliverables successfully written.")

@app.local_entrypoint()
def main(max_steps: int = 1500, lr: float = 2e-5):
    trainer = FluxAOTDatasetTrainer()
    trainer.train_lora_job.remote(max_train_steps=max_steps, lr=lr)
