# FLUX.2-Klein 4B Inference Optimization

This repository demonstrates memory-efficient inference for **FLUX.2-Klein 4B** using a combination of:

* TorchAO INT8 Weight-Only Quantization
* BitsAndBytes INT8 Quantization
* BF16 Mixed Precision Inference

The objective is to reduce GPU memory consumption while maintaining practical image generation performance on consumer-grade GPUs.

---

## Model Information

| Property                  | Value                   |
| ------------------------- | ----------------------- |
| Model                     | FLUX.2-Klein 4B         |
| Framework                 | Hugging Face Diffusers  |
| Precision                 | BF16                    |
| Transformer Quantization  | TorchAO INT8            |
| Text Encoder Quantization | BitsAndBytes INT8       |
| GPU                       | NVIDIA Tesla T4 (16 GB) |
| Runtime                   | Google Colab            |

---

## Optimization Techniques

### 1. Transformer Quantization (TorchAO)

The diffusion transformer is quantized using TorchAO INT8 weight-only quantization.

Benefits:

* Reduced VRAM consumption
* Lower memory footprint
* Faster model loading
* Minimal quality degradation

---

### 2. Text Encoder Quantization (BitsAndBytes)

The text encoder is quantized using BitsAndBytes 8-bit quantization.

Benefits:

* Efficient prompt encoding
* Reduced memory usage
* Better deployment on constrained hardware

---

## Benchmark Methodology

The benchmark follows the steps below:

1. Load quantized model components.
2. Run a warm-up inference pass.
3. Reset CUDA memory statistics.
4. Measure end-to-end inference latency.
5. Record peak allocated GPU memory.
6. Save generated output image.

---

## Experimental Configuration

| Parameter                 | Value             |
| ------------------------- | ----------------- |
| Resolution                | 512 × 512         |
| Sampling Steps            | 2                 |
| Guidance Scale            | 0.0               |
| Precision                 | BF16              |
| Transformer Quantization  | TorchAO INT8      |
| Text Encoder Quantization | BitsAndBytes INT8 |

---

## Results

| Metric          | Value         |
| --------------- | ------------- |
| Inference Time  | **10.51 s**   |
| Peak VRAM Usage | **9.38 GB**   |
| Resolution      | **512 × 512** |
| Inference Steps | **2**         |

### Sample Prompt

```text
A futuristic city at sunset, cyberpunk style
```

---

## Output

Generated image:

![Output](outputs/output.png)
```
└── 📁outputs
    ├── output .png
    └── ss_output.jpeg
```
---

## Repository Structure

```text
FLUX2-Klein-Inference-Optimization/
│
├── README.md
├── requirements.txt
├── main.py
│
└── outputs/
    └── output.png
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd FLUX2-Klein-Inference-Optimization
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run inference:

```bash
python main.py
```

The generated image will be saved as:

```text
output.png
```

---

## Key Findings

* Successfully executed FLUX.2-Klein 4B on a Tesla T4 GPU.
* Quantization enabled deployment within 9.38 GB of VRAM.
* End-to-end image generation completed in 10.51 seconds.
* BF16 and INT8 quantization provide a practical trade-off between memory efficiency and performance.
* Demonstrates feasibility of running multi-billion parameter diffusion models on limited-memory GPUs.

---


## References

* Hugging Face Diffusers
* TorchAO
* BitsAndBytes
* FLUX.2-Klein 4B
