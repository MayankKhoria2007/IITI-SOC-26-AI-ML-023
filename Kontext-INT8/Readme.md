# 🚀 FLUX.1 Kontext INT8 Inference

Ultra-efficient **FLUX.1-Kontext-dev** inference using **TorchAO INT8 weight-only quantization** with GPU-native execution.

This implementation significantly reduces memory usage while maintaining high image quality and fast inference performance.

---

## ✨ Features

* ⚡ **FLUX.1-Kontext-dev** image editing/generation
* 🧠 **Native INT8 Transformer Quantization** via TorchAO
* 🚀 GPU-native inference (no CPU offloading during denoising)
* 💾 Reduced VRAM footprint
* 🔥 Optimized for NVIDIA GPUs
* 🖼️ Supports both **local images** and **image URLs**
* ♻️ Automatic text encoder CPU offloading after prompt encoding
* 📈 Interactive generation loop
* 🏎️ Flash attention optimizations enabled

---

## 📊 Performance

Tested on **NVIDIA L4 GPU**

| Metric                 | Value             |
| ---------------------- | ----------------- |
| Resolution             | 1024 × 1024       |
| Inference Steps        | 6                 |
| Guidance Scale         | 4.5               |
| Precision              | BF16              |
| Quantization           | INT8 Weight-Only  |
| Average Inference Time | **~32.5 seconds** |
| Peak VRAM Usage        | **13.66 GB**      |

---

## 🏗️ Architecture

The pipeline uses:

* **FLUX.1-Kontext-dev**
* **TorchAO INT8 Weight-Only Quantization**
* **BF16 activations**
* **CUDA execution override**
* **Manual text encoder offloading**
* **Memory-efficient attention**

```text
Prompt
   ↓
Text Encoding (GPU)
   ↓
Move Encoders → CPU
   ↓
INT8 FLUX Transformer (GPU)
   ↓
Image Generation
   ↓
Save Output
```

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/flux-kontext-int8.git
cd flux-kontext-int8
```

Create environment:

```bash
conda create -n flux python=3.12 -y
conda activate flux
```

Install PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Install dependencies:

```bash
pip install diffusers transformers accelerate safetensors pillow
pip install torchao
pip install sentencepiece protobuf
```

---

## 🔑 Hugging Face Authentication

You must have access to:

* `black-forest-labs/FLUX.1-Kontext-dev`

Login:

```bash
huggingface-cli login
```

or

```bash
export HF_TOKEN=your_hf_token
```

---

## 🚀 Usage

Run:

```bash
python app.py
```

Interactive session:

```text
====================================================
 FLUX.1 Kontext — Interactive Generation Loop
 INT8 GPU Native edition
 Type 'quit' or 'exit' at any prompt to stop.
====================================================

[ Job #1 ]

Image (URL or local path):
Prompt:
Rendering...
```

Example:

```text
Image:
./cat.png

Prompt:
Turn the cat into a cyberpunk warrior
```

---

## 📂 Output

Generated images are automatically saved to:

```text
output/
├── result_001.png
├── result_002.png
├── result_003.png
```

---

## ⚙️ Key Optimizations

### INT8 Quantized Transformer

```python
quantization_config=TorchAoConfig(
    quant_type=Int8WeightOnlyConfig()
)
```

Reduces transformer memory consumption while preserving quality.

---

### Manual Text Encoder Offloading

Text encoders are moved to CPU immediately after prompt embedding generation:

```python
pipe.text_encoder.to("cpu")
pipe.text_encoder_2.to("cpu")
```

This frees several gigabytes of VRAM for denoising.

---

### CUDA Execution Override

A custom pipeline ensures all tensors remain on GPU during inference:

```python
class CudaFluxKontextPipeline(FluxKontextPipeline):
    @property
    def _execution_device(self):
        return torch.device("cuda:0")
```

---

### Memory Efficient Attention

```python
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

Improves memory efficiency during inference.

---

## 🖥️ Hardware Requirements

Recommended:

* NVIDIA GPU with **16 GB+ VRAM**
* CUDA 12+
* Python 3.12+

Minimum tested configuration:

* NVIDIA L4 (24 GB VRAM)

---

## 📜 License

This repository only provides inference code.

Please comply with the license terms of:

* FLUX.1-Kontext-dev
* Diffusers
* TorchAO

---

## 🙏 Acknowledgements

* Black Forest Labs
* Hugging Face Diffusers
* PyTorch
* TorchAO
