"""
Convert a CLIP / MobileCLIP image or text encoder from PyTorch to int8 TFLite.

Pipeline (each stage logs so we can see exactly where conversion breaks):
  1. Load model via open_clip (supports OpenAI CLIP and MobileCLIP/2 variants).
  2. Wrap the chosen tower (image or text) as a standalone nn.Module.
  3. Sanity-check the PyTorch forward pass + capture a reference embedding.
  4. Export to float32 TFLite via ai-edge-torch (proves the graph converts).
  5. Export to int8 TFLite with post-training quantization.
  6. Verify: run TFLite vs PyTorch on the same input, report cosine similarity.

Start with an easy, MIT-licensed model (ViT-B-32/openai) to prove the toolchain,
then point --model at MobileCLIP-S0 (the real target for the low-end F22).

Usage:
  python convert.py --model ViT-B-32 --pretrained openai --tower image
  python convert.py --model MobileCLIP-S0 --pretrained datacompdr --tower image
"""

import argparse
import os

import numpy as np
import torch


def log(stage: str, msg: str = "") -> None:
    print(f"[{stage}] {msg}", flush=True)


def build_tower(model, tower: str) -> torch.nn.Module:
    """Wrap encode_image / encode_text as a standalone module for export."""

    class ImageTower(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, image):
            return self.m.encode_image(image)

    class TextTower(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, tokens):
            return self.m.encode_text(tokens)

    return (ImageTower(model) if tower == "image" else TextTower(model)).eval()


def sample_input(model, tower: str, preprocess) -> torch.Tensor:
    """A single representative input tensor for the chosen tower."""
    if tower == "image":
        # Model's expected image size comes from the preprocess transform.
        size = preprocess.transforms[0].size
        side = size if isinstance(size, int) else size[0]
        return torch.randn(1, 3, side, side)
    # Text tower: CLIP context length is 77. Use int32 tokens — CLIP's encode_text
    # does an argmax over token ids to find the EOT position, and the TFLite
    # converter rejects arg_max on int64 (but accepts int32).
    return torch.randint(0, 49407, (1, 77), dtype=torch.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--tower", choices=["image", "text"], default="image")
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"{args.model}_{args.pretrained}_{args.tower}".replace("/", "-")

    log("load", f"open_clip {args.model} / {args.pretrained}")
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model.eval()

    tower = build_tower(model, args.tower)
    x = sample_input(model, args.tower, preprocess)
    log("load", f"tower={args.tower} input shape={tuple(x.shape)} dtype={x.dtype}")

    log("torch-forward", "running reference PyTorch forward")
    with torch.no_grad():
        ref = tower(x).float().numpy()
    log("torch-forward", f"embedding shape={ref.shape}")

    log("convert-f32", "litert_torch.convert (float32)")
    import litert_torch

    edge_f32 = litert_torch.convert(tower, (x,))
    f32_path = os.path.join(args.outdir, f"{tag}_f32.tflite")
    edge_f32.export(f32_path)
    log("convert-f32", f"wrote {f32_path} ({os.path.getsize(f32_path)/1e6:.1f} MB)")

    log("verify-f32", "comparing TFLite(f32) vs PyTorch")
    got = edge_f32(x)
    got = got.detach().numpy() if hasattr(got, "detach") else np.asarray(got)
    cos = float(
        np.dot(ref.ravel(), got.ravel())
        / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-9)
    )
    log("verify-f32", f"cosine similarity = {cos:.5f}")

    log("done", "float32 path OK — int8 quantization is the next stage")


if __name__ == "__main__":
    main()
