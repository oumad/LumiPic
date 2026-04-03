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

# HDRDiT — Single-Image SDR to HDR LoRA

Converts standard dynamic range (SDR) images to high dynamic range (HDR) EXR files with 10+ stops of dynamic range.

## Usage

```python
from diffusers import QwenImageEditPipeline
import torch
from PIL import Image

pipe = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("oumoumad/HDRDiT")
pipe = pipe.to("cuda")

image = Image.open("photo.jpg").convert("RGB")
output = pipe(prompt="Convert this image to HDR", image=image,
              num_inference_steps=40, guidance_scale=3.0, output_type="pt")

# Decode LogC3 → linear HDR
pixel = output.images[0].float().permute(1, 2, 0).cpu().clamp(0, 1)
# Apply LogC3 decompression (see logc3.py in GitHub repo)
```

See [GitHub repo](https://github.com/oumad/HDRDiT) for complete inference code with EXR output.

## Training

- **Base model**: Qwen-Image-Edit-2511 (frozen VAE)
- **Technique**: LoRA (rank 32) on DiT, trained to output ARRI LogC3 encoded HDR
- **Dataset**: 204 diverse HDR pairs (Poly Haven, RED/ARRI footage, CG renders)
- **SDR augmentation**: 61% degraded inputs (exposure, compression, blur, contrast)
- **Steps**: 2000, batch 4, bf16, AdamW, lr 1e-4
