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
base_model: Qwen/Qwen-Image-Edit-2511
license: mit
---

# LumiPic — Single-Image SDR to HDR LoRA

Converts standard dynamic range (SDR) images to high dynamic range (HDR) EXR files — float-valued, with range well beyond what an 8-bit SDR output can carry.

Based on [LumiVid](https://hdr-lumivid.github.io/) ([paper](https://arxiv.org/abs/2604.11788)) — the Lightricks research that introduced LogC3-encoded diffusion for HDR generation. LumiPic adapts that technique to a single-image editing DiT.

## Usage

```python
from diffusers import QwenImageEditPipeline
import torch
from PIL import Image

pipe = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
# Default: v5b_step2000 (current best). Pass weight_name="hdrdit_v1_QE2511.safetensors" for v1.
pipe.load_lora_weights("oumoumad/LumiPic", weight_name="v5b_step2000.safetensors")
pipe = pipe.to("cuda")

image = Image.open("photo.jpg").convert("RGB")
output = pipe(prompt="Convert this image to HDR", image=image,
              num_inference_steps=40, guidance_scale=3.0, output_type="pt")

# Decode LogC3 → linear HDR
pixel = output.images[0].float().permute(1, 2, 0).cpu().clamp(0, 1)
# Apply LogC3 decompression (see logc3.py in GitHub repo)
```

See [GitHub repo](https://github.com/oumad/LumiPic) for complete inference code with EXR output.

## Weights

- `v5b_step2000.safetensors` — **default** (rank 32, 563 MB). Most robust overall; best on stylized/AI-generated SDR inputs.
- `v9_step1500.safetensors` — alternative (rank 32, 563 MB). LumiVid-aligned augs (joint EV shifts, luminance blur p=1.0). Slightly better on natural photo content, worse on AI-generated inputs.
- `hdrdit_v1_QE2511.safetensors` — original v1 release (rank 32, 563 MB).

## Quick Inference

```bash
git clone https://github.com/oumad/LumiPic.git && cd LumiPic
pip install -r requirements.txt
python inference.py --image photo.jpg
```

The base model and LoRA weights download automatically on first run.

## Training

- **Base model**: Qwen-Image-Edit-2511 (frozen VAE)
- **Technique**: LoRA (rank 32) on DiT, trained to output ARRI LogC3 encoded HDR
- **Dataset**: ~260 diverse HDR pairs (Poly Haven, RED/ARRI footage, CG renders, Blender scenes)
- **SDR augmentation**: exposure shifts, luminance blur, contrast, JPEG, white-balance jitter
- **Steps**: 2000, batch 4, bf16, AdamW, lr 1e-4
