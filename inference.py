"""HDRDiT — Single-Image SDR to HDR Reconstruction via LogC3-Encoded Diffusion Transformer LoRA.

Converts standard dynamic range (SDR) images to high dynamic range (HDR) EXR files
using a LoRA fine-tuned on Qwen-Image-Edit-2511 with ARRI LogC3 encoding.

Usage:
    # Single image
    python inference.py --image photo.jpg --output photo_hdr.exr

    # Directory of images
    python inference.py --image-dir ./inputs --output-dir ./outputs

    # With custom settings
    python inference.py --image photo.jpg --steps 40 --guidance 3.0 --seed 42

Requirements:
    pip install torch diffusers transformers accelerate safetensors Pillow opencv-python numpy
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


# ── ARRI LogC3 (EI 800) ──────────────────────────────────────────────────

class LogC3:
    """ARRI LogC3 transfer function (EI 800).

    Compresses scene-linear values [0, ~55.1] into [0, 1] range.
    Used by professional cinema cameras (ARRI ALEXA) and VFX pipelines.
    Provides ~14 stops of dynamic range with perceptually uniform distribution.
    """
    A = 5.555556
    B = 0.052272
    C = 0.247190
    D = 0.385537
    E = 5.367655
    F = 0.092809
    CUT = 0.010591

    def decompress(self, logc: torch.Tensor) -> torch.Tensor:
        """Convert LogC3 [0, 1] back to scene-linear HDR values."""
        logc = torch.clamp(logc, 0.0, 1.0)
        cut_log = self.E * self.CUT + self.F
        linear_from_log = (torch.pow(10.0, (logc - self.D) / self.C) - self.B) / self.A
        linear_from_lin = (logc - self.F) / self.E
        return torch.where(logc >= cut_log, linear_from_log, linear_from_lin)


# ── Tone mapping for preview ─────────────────────────────────────────────

def tonemap_reinhard(hdr: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Simple Reinhard tonemap for preview PNG generation."""
    hdr = np.maximum(hdr, 0)
    mapped = hdr / (1.0 + hdr)
    return np.clip(np.power(mapped, 1.0 / gamma), 0, 1)


# ── Main inference ────────────────────────────────────────────────────────

def load_pipeline(model_id: str = "Qwen/Qwen-Image-Edit-2511",
                  lora_id: str = "oumoumad/HDRDiT",
                  device: str = "cuda",
                  dtype=torch.bfloat16):
    """Load the base model and HDRDiT LoRA weights.

    Args:
        model_id: HuggingFace ID for the base model.
                  Default: "Qwen/Qwen-Image-Edit-2511"
        lora_id:  HuggingFace repo ID (e.g., "oumoumad/HDRDiT")
                  or local path to a .safetensors file.
        device:   "cuda" or "cpu".
        dtype:    torch.bfloat16 (recommended) or torch.float16.
    """
    from diffusers import QwenImageEditPipeline

    print(f"Loading base model: {model_id}")
    pipe = QwenImageEditPipeline.from_pretrained(model_id, torch_dtype=dtype)

    print(f"Loading HDRDiT LoRA: {lora_id}")
    if os.path.isfile(lora_id):
        # Local .safetensors file
        pipe.load_lora_weights(lora_id)
    else:
        # HuggingFace repo — downloads automatically
        pipe.load_lora_weights(lora_id)

    pipe = pipe.to(device)
    return pipe


def convert_to_hdr(pipe, image: Image.Image, logc3: LogC3,
                   steps: int = 40, guidance: float = 3.0,
                   seed: int = 42) -> np.ndarray:
    """Run SDR→HDR conversion. Returns linear HDR numpy array [H, W, 3]."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    output = pipe(
        prompt="Convert this image to HDR",
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        output_type="pt",
    )

    # Decode: VAE output is LogC3-encoded [0, 1]
    pixel = output.images[0].float()
    logc3_data = torch.clamp(pixel, 0, 1).permute(1, 2, 0).cpu()

    # Decompress LogC3 → scene-linear HDR
    hdr = logc3.decompress(logc3_data)
    return hdr.numpy()


def save_exr(hdr: np.ndarray, path: str):
    """Save linear HDR image as EXR (half-float, ZIP compression)."""
    bgr = hdr[:, :, ::-1].copy().astype(np.float32)
    cv2.imwrite(path, bgr,
                [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
                 cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP])


def save_preview(hdr: np.ndarray, path: str):
    """Save tonemapped preview PNG."""
    preview = tonemap_reinhard(hdr)
    Image.fromarray((preview * 255).clip(0, 255).astype(np.uint8)).save(path)


def process_image(pipe, logc3, image_path, output_path,
                  steps=40, guidance=3.0, seed=42, save_preview_png=True):
    """Process a single image: SDR → HDR EXR."""
    image = Image.open(image_path).convert("RGB")

    t0 = time.time()
    hdr = convert_to_hdr(pipe, image, logc3, steps, guidance, seed)
    elapsed = time.time() - t0

    # Save EXR
    save_exr(hdr, output_path)

    # Optional preview
    if save_preview_png:
        preview_path = output_path.rsplit('.', 1)[0] + '_preview.png'
        save_preview(hdr, preview_path)

    # Stats
    hdr_max = float(hdr.max())
    pct_above_1 = float((hdr > 1.0).mean() * 100)
    p99 = float(np.percentile(hdr, 99))

    return {
        "max": round(hdr_max, 2),
        "pct_above_1": round(pct_above_1, 2),
        "p99": round(p99, 2),
        "time": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="HDRDiT — Convert SDR images to HDR EXR files"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Single input image path")
    group.add_argument("--image-dir", type=str, help="Directory of input images")

    parser.add_argument("--output", type=str, default=None,
                        help="Output EXR path (for single image)")
    parser.add_argument("--output-dir", type=str, default="./hdr_output",
                        help="Output directory (for batch)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen-Image-Edit-2511",
                        help="Base model ID")
    parser.add_argument("--lora", type=str, default="oumoumad/HDRDiT",
                        help="LoRA weights (HuggingFace repo ID or local .safetensors path)")
    parser.add_argument("--steps", type=int, default=40,
                        help="Inference steps (default: 40)")
    parser.add_argument("--guidance", type=float, default=3.0,
                        help="Guidance scale (default: 3.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip generating preview PNGs")
    args = parser.parse_args()

    # Load pipeline
    pipe = load_pipeline(args.model, args.lora)
    logc3 = LogC3()

    # Collect images
    if args.image:
        images = [(args.image, args.output or args.image.rsplit('.', 1)[0] + '_hdr.exr')]
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        images = []
        for f in sorted(os.listdir(args.image_dir)):
            if os.path.splitext(f)[1].lower() in exts:
                stem = os.path.splitext(f)[0]
                images.append((
                    os.path.join(args.image_dir, f),
                    os.path.join(args.output_dir, f"{stem}_hdr.exr"),
                ))

    if not images:
        print("No images found!")
        sys.exit(1)

    print(f"\nProcessing {len(images)} image(s)...\n")

    for img_path, out_path in images:
        print(f"  {os.path.basename(img_path)}", end=" → ", flush=True)
        stats = process_image(pipe, logc3, img_path, out_path,
                              args.steps, args.guidance, args.seed,
                              not args.no_preview)
        print(f"max={stats['max']:.1f} >1.0={stats['pct_above_1']:.1f}% "
              f"p99={stats['p99']:.1f} ({stats['time']:.1f}s)")

    print(f"\nDone! HDR files saved to: {args.output_dir if args.image_dir else os.path.dirname(out_path)}")


if __name__ == "__main__":
    main()
