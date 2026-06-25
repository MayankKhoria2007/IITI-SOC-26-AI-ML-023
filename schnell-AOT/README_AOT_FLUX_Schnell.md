# Ahead-of-Time (AOT) Compilation of FLUX.1 Schnell Transformer

## Overview

The objective of this work is to optimize the inference of the **FLUX.1
Schnell** model by compiling its transformer using **PyTorch
AOTInductor**. Instead of executing the transformer through the Python
runtime, the transformer is exported into a static computation graph and
compiled into optimized native code for efficient inference.

------------------------------------------------------------------------

## AOT Compilation Workflow

``` mermaid
flowchart LR

A[Load FLUX.1 Schnell Pipeline]
--> B[Register Forward Pre-Hook]

B --> C[Run Short Pipeline Pass]

C --> D[Capture Sample Inputs<br/>(args & kwargs)]

D --> E[Export Transformer<br/>to FX Graph]

E --> F[AOTInductor Compilation]

F --> G[Generate Compiled Artifact<br/>model.pt2]

F --> H[Save Sample Inputs<br/>captured_inputs.pt]

G --> I[Load Compiled Artifact]

H --> I

I --> J[Run Compiled Transformer<br/>using Saved Sample Inputs]

J --> K[Transformer Outputs]
```

------------------------------------------------------------------------

## Workflow Description

### 1. Sample Input Capture

The main challenge during AOT compilation was obtaining suitable
**sample inputs** for exporting the transformer. Since these inputs are
generated internally by the diffusion pipeline, a **forward pre-hook**
was registered on the transformer to capture the sample inputs during a
pipeline execution. The captured inputs were then used for export.

### 2. Transformer Export

The transformer was exported using **Torch Export**, producing an **FX
graph** representing the transformer computation.

### 3. AOT Compilation

The exported FX graph was compiled using **PyTorch AOTInductor**,
generating a compiled package (`model.pt2`) containing optimized C++
code, Triton kernels, and runtime metadata. The captured sample inputs
were also saved (`captured_inputs.pt`) for inference.

### 4. Compiled Inference

During inference, the compiled artifact is loaded and executed using the
saved sample inputs instead of the original transformer.

------------------------------------------------------------------------

## Current Status

### Completed

-   Captured sample inputs using a forward pre-hook.
-   Exported the transformer to an FX graph.
-   Compiled the exported graph using AOTInductor.
-   Generated and loaded the compiled artifact.
-   Successfully executed the compiled transformer independently using
    the saved sample inputs.

### Current Challenge

The compiled transformer executes correctly in isolation; however,
integrating it back into the original Diffusers pipeline to generate the
final image is still under development.

------------------------------------------------------------------------

## Performance Summary

  Metric                                                      Value
  ---------------------------- ------------------------------------
  Model                                  FLUX.1 Schnell Transformer
  Export Method                                        Torch Export
  Compiler                                      PyTorch AOTInductor
  Compiled Artifact              `model.pt2` (C++ + Triton kernels)
  Sample Input Capture                             Forward Pre-Hook
  Transformer Inference Time                            **≈ 2.5 s**
  Peak GPU VRAM                                      **≈ 11.24 GB**
  Pipeline Integration                                  In Progress
