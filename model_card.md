---
library_name: diffusers
pipeline_tag: image-to-image
tags:
  - lora
  - hdr
  - exr
  - tone-mapping
  - logc3
  - diffusion-transformer
  - qwen-image-edit
  - flux2-klein
base_model:
  - Qwen/Qwen-Image-Edit-2511
  - black-forest-labs/FLUX.2-klein-base-4B
  - black-forest-labs/FLUX.2-klein-base-9B
license: mit
---

# LumiPic — Single-Image SDR to HDR LoRA

Converts standard dynamic range (SDR) images to high dynamic range (HDR) EXR files — float-valued, with range well beyond what an 8-bit SDR output can carry.

Based on [LumiVid](https://hdr-lumivid.github.io/) ([paper](https://arxiv.org/abs/2604.11788)) — the Lightricks research that introduced LogC3-encoded diffusion for HDR generation. LumiPic is the same technique adapted to single-image diffusion transformers; *the technique is base-model agnostic*. Two trained LoRA families are published here, on different bases.

## Examples

Same 20 HDR outputs, viewed at two extreme exposure offsets — highlights still hold structure at EV+6, shadows still hold information at EV-6.

**Exposure +6 (highlights pulled down):**

![EV+6](grid_ev+6.png)

**Exposure -6 (shadows pushed up):**

![EV-6](grid_ev-6.png)

## Weights

### Qwen-Image-Edit-2511 (mature)

5+ training iterations, 563 MB per LoRA. Best quality, larger base (~54 GB).

- `v5b_step2000.safetensors` — **Qwen default**. Most robust overall; best on stylized/AI-generated SDR inputs.
- `v9_step1500.safetensors` — alternative. LumiVid-aligned augs (joint HDR+SDR EV shifts, luminance blur p=1.0). Slightly better on natural photos.
- `hdrdit_v1_QE2511.safetensors` — original v1 release.

### FLUX.2-klein-base-4B (alpha)

Single training iteration, 88 MB per LoRA. Apache 2.0 base, 5× smaller than Qwen, fastest end-to-end.

- `klein4b_alpha_step1750.safetensors` — **klein-4B default**. Faithful HDR look; well-balanced.
- `klein4b_alpha_step1000.safetensors` — alternative. Aggressive HDR (higher p99) but tends to blow out bright highlights.

### FLUX.2-klein-base-9B (alpha)

Single training iteration, 158 MB per LoRA. Larger klein variant — more capacity, more nuanced HDR. Base model is **gated** on HF (request access at the [base model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) before first run).

- `klein9b_alpha_step2000.safetensors` — **klein-9B default**. Stable, well-behaved (mean p99 9.1, no scenes saturated in our 20-image benchmark).
- `klein9b_alpha_step1250.safetensors` — most "good" scenes (13/20 in 5-50 p99 range, 0 saturated). Practically the best by per-scene quality count.
- `klein9b_alpha_step1000.safetensors` — most aggressive (mean p99 20.6, but 4 scenes saturated). Pre-overfit peak HDR range.
- `klein9b_alpha_step{250,500,750,1500,1750}.safetensors` — intermediate snapshots, available for experimentation.

The training overfit curve was steep on klein-9B (mean p99: 1000=20.6 → 1250=11.1 → 1500=9.3 → 1750=9.6 → 2000=9.1). Step 2000 is the safest default; step 1250 is the practical sweet spot.

## Usage

**Qwen-Image-Edit-2511:**
```python
from diffusers import QwenImageEditPipeline
import torch
from PIL import Image

pipe = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("oumoumad/LumiPic", weight_name="v5b_step2000.safetensors")
pipe = pipe.to("cuda")

image = Image.open("photo.jpg").convert("RGB")
output = pipe(prompt="Convert this image to HDR", image=image,
              num_inference_steps=40, guidance_scale=3.0, output_type="pt")
# Decode LogC3 → linear HDR (see logc3.py in GitHub repo)
```

**FLUX.2-klein-base-4B / 9B** (requires bleeding-edge diffusers: `pip install "git+https://github.com/huggingface/diffusers.git"`):
```python
from diffusers import Flux2KleinPipeline
import torch
from PIL import Image

# 4B (Apache 2.0, ungated)
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("oumoumad/LumiPic", weight_name="klein4b_alpha_step1750.safetensors")

# 9B (gated — accept license on HF first)
# pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-9B", torch_dtype=torch.bfloat16)
# pipe.load_lora_weights("oumoumad/LumiPic", weight_name="klein9b_alpha_step2000.safetensors")

pipe.enable_model_cpu_offload()  # or pipe.to("cuda") if you have 32GB+

image = Image.open("photo.jpg").convert("RGB")
output = pipe(prompt="Convert this image to HDR", image=image,
              num_inference_steps=25, guidance_scale=3.0, output_type="pt")
```

See the [GitHub repo](https://github.com/oumad/LumiPic) for complete inference code with EXR output (`inference.py` for Qwen, `inference_klein.py` for klein).

## Quick Inference

```bash
git clone https://github.com/oumad/LumiPic.git && cd LumiPic
pip install -r requirements.txt
python inference.py --image photo.jpg          # Qwen path (production)
python inference_klein.py --image photo.jpg    # klein path (alpha)
```

The base model and LoRA weights download automatically on first run.

## ComfyUI

Three ready-to-use workflows:
- [`SDR_To_HDR_QE11.json`](https://huggingface.co/oumoumad/LumiPic/resolve/main/SDR_To_HDR_QE11.json?download=true) — Qwen-Image-Edit-2511
- [`SDR_To_HDR_klein4b.json`](https://huggingface.co/oumoumad/LumiPic/resolve/main/SDR_To_HDR_klein4b.json?download=true) — FLUX.2-klein-base-4B
- [`SDR_To_HDR_klein9b.json`](https://huggingface.co/oumoumad/LumiPic/resolve/main/SDR_To_HDR_klein9b.json?download=true) — FLUX.2-klein-base-9B

Both use the `Gear · LogC3 Decode + Save EXR` node from [ComfyUI_Gear](https://github.com/oumad/ComfyUI_Gear) for the LogC3 decode + EXR write. Drop the JSON onto your canvas, place the matching LoRA file in `ComfyUI/models/loras/{qwen,flux}/hdr/`, install ComfyUI_Gear, queue with prompt `"Convert this image to HDR"`.

## Training

- **Technique**: LoRA (rank 32) on the DiT transformer, trained to output ARRI LogC3-encoded HDR
- **Dataset**: ~260–606 diverse HDR pairs (Poly Haven HDRIs, RED/ARRI footage, CG renders, Blender scenes), augmented with exposure shifts, luminance blur, contrast, JPEG, white-balance jitter
- **Hyperparams**: rank 32, alpha 32, bf16, AdamW lr 1e-4, flowmatch scheduler
- **Steps**: 2000 (Qwen sweet spot ~1500; klein-4B sweet spot ~1500–1750, peak HDR at 1000)
- Trained with [Ostris AI-Toolkit](https://github.com/ostris/ai-toolkit). Qwen LoRAs use a fork with float32 `.npy` targets ([npy-float32-targets branch](https://github.com/oumad/ai-toolkit/tree/npy-float32-targets)); klein LoRAs use upstream + a 4-line VAE-on-GPU patch to `flux2_model.py`.
