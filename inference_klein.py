"""LumiPic — FLUX.2-klein-base-4B inference path.

Same idea as inference.py but using the FLUX.2-klein-base-4B (Apache 2.0)
base model with the klein-4b LoRA. Klein output is ~5x smaller and ~2x
faster than the Qwen-Image-Edit pipeline at the cost of being newer/less
mature.

Requires:
    pip install "git+https://github.com/huggingface/diffusers.git"  # >= 0.37.0.dev0
    pip install peft>=0.17.0 transformers>=4.40 accelerate safetensors Pillow opencv-python numpy

Usage:
    # Single image
    python inference_klein.py --image photo.jpg

    # With a specific checkpoint (alpha = step 1000 or 1750)
    python inference_klein.py --image photo.jpg --weight-name klein4b_alpha_step1750.safetensors

    # Local LoRA path
    python inference_klein.py --image photo.jpg --lora ./path/to/klein4b_step1000.safetensors

    # Batch
    python inference_klein.py --image-dir ./inputs --output-dir ./outputs
"""
import argparse
import os
import sys
import time

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch
from PIL import Image

from logc3 import LogC3


DEFAULT_LORA_REPO = "oumoumad/LumiPic"
DEFAULT_LORA_WEIGHT = "klein4b_alpha_step1750.safetensors"
DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"

# klein-9B variant — pass --base black-forest-labs/FLUX.2-klein-base-9B
# along with --weight-name klein9b_alpha_step2000.safetensors. The 9B base
# is gated; accept the license at the HF page first.


def tonemap_reinhard(hdr, gamma=2.2):
    hdr = np.maximum(hdr, 0)
    return np.clip(np.power(hdr / (1 + hdr), 1 / gamma), 0, 1)


def load_pipeline(model_id=DEFAULT_BASE_MODEL,
                  lora_id=DEFAULT_LORA_REPO,
                  weight_name=DEFAULT_LORA_WEIGHT,
                  cpu_offload=True,
                  dtype=torch.bfloat16):
    """Load FLUX.2-klein-base-4B and the LumiPic klein LoRA.

    Args:
        model_id:    Base model HF ID (default: FLUX.2-klein-base-4B, Apache 2.0).
        lora_id:     HF repo ID or local .safetensors path.
        weight_name: Filename inside the HF repo. Two alphas are published:
                     - klein4b_alpha_step1000.safetensors -- aggressive HDR
                     - klein4b_alpha_step1750.safetensors -- faithful HDR (default)
        cpu_offload: If True, use enable_model_cpu_offload(). Recommended on <32GB GPUs.
                     Set False on 32GB+ for ~2x speedup.
    """
    from diffusers import Flux2KleinPipeline

    print(f"Loading base model: {model_id}")
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)

    if os.path.isfile(lora_id):
        print(f"Loading LoRA from local: {lora_id}")
        pipe.load_lora_weights(lora_id)
    else:
        print(f"Loading LoRA from HF: {lora_id} ({weight_name})")
        pipe.load_lora_weights(lora_id, weight_name=weight_name)

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    return pipe


def convert_to_hdr(pipe, image, logc3, steps=25, guidance=3.0, seed=42):
    """SDR -> HDR float32 numpy [H, W, 3] in scene-linear."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(
        prompt="Convert this image to HDR",
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        output_type="pt",
    )
    pixel = out.images[0].float().clamp(0, 1)
    logc_data = pixel.permute(1, 2, 0).cpu()
    return logc3.decompress(logc_data).numpy().astype(np.float32)


def save_exr(hdr, path):
    bgr = hdr[:, :, ::-1].copy().astype(np.float32)
    cv2.imwrite(path, bgr,
                [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
                 cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP])


def save_preview(hdr, path):
    Image.fromarray((tonemap_reinhard(hdr) * 255).clip(0, 255).astype(np.uint8)).save(path)


def process_image(pipe, logc3, image_path, output_path,
                  steps=25, guidance=3.0, seed=42, save_preview_png=True):
    image = Image.open(image_path).convert("RGB")
    t0 = time.time()
    hdr = convert_to_hdr(pipe, image, logc3, steps, guidance, seed)
    elapsed = time.time() - t0
    save_exr(hdr, output_path)
    if save_preview_png:
        save_preview(hdr, output_path.rsplit(".", 1)[0] + "_preview.png")
    return {
        "max": round(float(hdr.max()), 2),
        "pct_above_1": round(float((hdr > 1).mean() * 100), 2),
        "p99": round(float(np.percentile(hdr, 99)), 2),
        "time": round(elapsed, 1),
    }


def main():
    ap = argparse.ArgumentParser(description="LumiPic klein-4B inference")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str)
    g.add_argument("--image-dir", type=str)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default="./hdr_output")
    ap.add_argument("--model", "--base", type=str, default=DEFAULT_BASE_MODEL,
                    help="Base model HF id. Use FLUX.2-klein-base-9B with klein9b_* weights.")
    ap.add_argument("--lora", type=str, default=DEFAULT_LORA_REPO)
    ap.add_argument("--weight-name", type=str, default=DEFAULT_LORA_WEIGHT)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--no-cpu-offload", action="store_true",
                    help="Disable cpu_offload for 32GB+ GPUs (~2x faster)")
    args = ap.parse_args()

    pipe = load_pipeline(args.model, args.lora, args.weight_name,
                         cpu_offload=not args.no_cpu_offload)
    logc3 = LogC3()

    if args.image:
        images = [(args.image, args.output or args.image.rsplit(".", 1)[0] + "_hdr.exr")]
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        images = []
        for f in sorted(os.listdir(args.image_dir)):
            if os.path.splitext(f)[1].lower() in exts:
                stem = os.path.splitext(f)[0]
                images.append((os.path.join(args.image_dir, f),
                               os.path.join(args.output_dir, f"{stem}_hdr.exr")))

    if not images:
        print("No images found!"); sys.exit(1)

    print(f"\nProcessing {len(images)} image(s)...\n")
    for img_path, out_path in images:
        print(f"  {os.path.basename(img_path)}", end=" -> ", flush=True)
        s = process_image(pipe, logc3, img_path, out_path,
                          args.steps, args.guidance, args.seed,
                          not args.no_preview)
        print(f"max={s['max']:.1f} >1={s['pct_above_1']:.1f}% p99={s['p99']:.1f} ({s['time']:.1f}s)")

    print(f"\nDone. HDR files saved to: {args.output_dir if args.image_dir else os.path.dirname(out_path)}")


if __name__ == "__main__":
    main()
