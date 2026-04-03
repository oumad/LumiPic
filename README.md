# HDRDiT

**Single-Image HDR Reconstruction via LogC3-Encoded Diffusion Transformer LoRA**

Convert any standard dynamic range (SDR) image into a true high dynamic range (HDR) EXR file with 10+ stops of dynamic range — using a lightweight LoRA adapter on a frozen diffusion transformer.

## How It Works

1. **Input**: Any SDR image (JPEG, PNG, etc.)
2. **Process**: A LoRA-adapted Diffusion Transformer (DiT) generates output in [ARRI LogC3](https://www.arri.com/en/learn-help/learn-help-camera-system/technical-information/about-log-c) space
3. **Output**: Scene-linear HDR EXR file with values up to ~55x brighter than white

The key insight: HDR values are compressed into LogC3 [0, 1] range before VAE encoding. The VAE stays frozen — it treats LogC3 data as a normal image. The LoRA teaches the DiT to produce LogC3-encoded output. At inference, the VAE output is decompressed back to linear HDR.

This technique can be applied to **any** Diffusion Transformer architecture. This release uses Qwen-Image-Edit-2511 as the base model.

<p align="center">
  <img src="assets/pipeline.png" width="800">
</p>

## Quick Start

```bash
pip install -r requirements.txt

# Single image
python inference.py --image photo.jpg

# Directory of images
python inference.py --image-dir ./inputs --output-dir ./outputs

# Custom settings
python inference.py --image photo.jpg --steps 40 --guidance 3.0 --seed 42
```

The LoRA weights are automatically downloaded from [HuggingFace](https://huggingface.co/oumad/HDRDiT).

## Requirements

- **GPU**: 48GB+ VRAM recommended (unquantized bf16)
  - 32GB possible with 8-bit quantization
  - 24GB possible with 4-bit quantization (quality tradeoff)
- **Python**: 3.10+
- **CUDA**: 12.0+

## Output Format

- **EXR files** in scene-linear RGB (Rec.709 primaries)
- Values range from 0 to ~55 (LogC3 EI 800 ceiling)
- Suitable for compositing in Nuke, Blender, DaVinci Resolve, etc.
- Recommended display transform: ACES Output Transform (sRGB/Rec.709)

## Examples

| Input (SDR) | Output (HDR, tonemapped for display) | EV -3 | EV +3 |
|:-----------:|:-----------------------------------:|:-----:|:-----:|
| ![](assets/examples/market_sdr.jpg) | ![](assets/examples/market_hdr.jpg) | ![](assets/examples/market_ev-3.jpg) | ![](assets/examples/market_ev+3.jpg) |
| ![](assets/examples/car_sdr.jpg) | ![](assets/examples/car_hdr.jpg) | ![](assets/examples/car_ev-3.jpg) | ![](assets/examples/car_ev+3.jpg) |

## Technical Details

### LogC3 Encoding

[ARRI LogC3](https://www.arri.com/en/learn-help/learn-help-camera-system/technical-information/about-log-c) (Exposure Index 800) is the standard log encoding used by ARRI cinema cameras. It maps scene-linear values to a perceptually uniform [0, 1] range:

- **Mid-gray** (0.18 linear) → 0.39 LogC3
- **10 stops above mid-gray** (~55.1 linear) → 1.0 LogC3
- **Shadow detail** (0.001 linear) → 0.09 LogC3

This encoding preserves highlight detail far better than simple gamma or PQ curves, and is widely used in the VFX industry.

### Training

The LoRA was trained using [Ostris AI-Toolkit](https://github.com/oumad/ai-toolkit/tree/npy-float32-targets) with:

- **Base model**: Qwen-Image-Edit-2511
- **Dataset**: 204 diverse HDR image pairs (Poly Haven HDRIs, RED/ARRI camera footage, CG renders, Blender scenes)
- **Target encoding**: Float32 LogC3-compressed .npy files
- **SDR augmentation**: 61% of input images degraded with exposure variation, luminance-selective blur, contrast adjustments, JPEG compression, and white balance jitter — forcing robustness to real-world SDR sources
- **LoRA rank**: 32, alpha 32
- **Training**: 2000 steps, batch 4, bf16, AdamW, lr 1e-4, flowmatch scheduler
- **Key config**: `cache_latents_to_disk: true` for efficient training

### Why LogC3 over other encodings?

| Encoding | Max linear | Stops above mid-gray | Industry use |
|----------|-----------|---------------------|-------------|
| **LogC3 (ours)** | 55.1 | ~14 | ARRI cameras, VFX |
| PU21 (X2HDR) | 10,000 | ~26 | Perceptual research |
| PQ/ST.2084 | 10,000 | ~26 | HDR displays |
| sRGB gamma | 1.0 | 0 | Consumer displays |

LogC3 provides sufficient range for most real-world content while staying within a compact [0, 1] range that existing VAEs handle well.

## Citation

```bibtex
@misc{hdrdit2026,
  title={HDRDiT: Single-Image HDR Reconstruction via LogC3-Encoded Diffusion Transformer LoRA},
  author={Oumoumad},
  year={2026},
  url={https://github.com/oumad/HDRDiT}
}
```

## Acknowledgments

- [Lightricks LTX-2 HDR IC-LoRA](https://huggingface.co/Lightricks/LTX-Video) — inspiration for the LogC3 approach
- [Ostris AI-Toolkit](https://github.com/ostris/ai-toolkit) — training framework
- [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) — base model
- [Poly Haven](https://polyhaven.com/) — CC0 HDR environment maps
- [ARRI](https://www.arri.com/) — LogC3 transfer function specification

## License

MIT
