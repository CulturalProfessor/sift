"""
Quantize a converted float32 CLIP/MobileCLIP encoder .tflite (image OR text) to
int8 using ai-edge-quantizer (operates on the flatbuffer directly).

Finding from the image encoder: activation quantization destroys CLIP/ViT
accuracy (static/dynamic int8 give cosine ~0.1-0.3); WEIGHT-ONLY int8 preserves
it (cosine ~0.98) while still shrinking ~3.6x. So weight_only_wi8_afp32 is the
default. Weight-only needs no calibration and keeps float I/O.

Usage:
  # image encoder
  python quantize_tflite.py --tower image \
      --f32 output/MobileCLIP2-S0_dfndr2b_image_f32.tflite
  # text encoder
  python quantize_tflite.py --tower text \
      --f32 output/MobileCLIP2-S0_dfndr2b_text_f32.tflite
"""

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image


def log(stage, msg=""):
    print(f"[{stage}] {msg}", flush=True)


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def cosine(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def run_tflite(path, feed):
    """Run a (possibly quantized) tflite model. `feed` is a float32 image array
    or an int token array. Handles integer-quantized float I/O via scale/zp;
    passes raw integer token inputs through unchanged."""
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]

    x = feed
    if np.issubdtype(x.dtype, np.floating) and inp["dtype"] != np.float32:
        scale, zp = inp["quantization"]  # quantized float input
        x = np.round(x / scale + zp).astype(inp["dtype"])
    else:
        x = x.astype(inp["dtype"])  # raw tokens, or matching float
    it.set_tensor(inp["index"], x)
    it.invoke()
    raw = it.get_tensor(out["index"])
    if out["dtype"] != np.float32:
        scale, zp = out["quantization"]
        return (raw.astype(np.float32) - zp) * scale
    return raw.astype(np.float32)


def build_image_calib(calib_dir, n, preprocess):
    paths = sorted(
        p for e in ("jpg", "jpeg", "png") for p in glob.glob(f"{calib_dir}/*.{e}")
    )[:n]
    return [
        preprocess(Image.open(p).convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
        for p in paths
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f32", required=True)
    ap.add_argument("--tower", choices=["image", "text"], default="image")
    ap.add_argument("--model", default="MobileCLIP2-S0")
    ap.add_argument("--pretrained", default="dfndr2b")
    ap.add_argument("--calib-dir", default="calib_images")
    ap.add_argument("--calib-n", type=int, default=40)
    ap.add_argument("--outdir", default="output")
    ap.add_argument(
        "--recipe",
        default="weight_only_wi8_afp32",
        choices=[
            "static_wi8_ai8",
            "static_wi8_ai16",
            "dynamic_wi8_afp32",
            "weight_only_wi8_afp32",
        ],
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    suffix = {
        "static_wi8_ai8": "_int8",
        "static_wi8_ai16": "_wi8ai16",
        "dynamic_wi8_afp32": "_dynint8",
        "weight_only_wi8_afp32": "_wonly",
    }[args.recipe]
    out_path = os.path.join(
        args.outdir, os.path.basename(args.f32).replace("_f32.tflite", f"{suffix}.tflite")
    )

    import open_clip

    log("load", f"open_clip {args.model}/{args.pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model.eval()

    # Tower-specific: reference embedding + verification/calibration inputs.
    if args.tower == "image":
        calib = build_image_calib(args.calib_dir, args.calib_n, preprocess)
        if not calib:
            raise SystemExit(f"No calibration images in {args.calib_dir}/")
        log("calib", f"loaded {len(calib)} images")
        verify_in = calib[0]
        with torch.no_grad():
            ref = model.encode_image(torch.from_numpy(calib[0])).float().numpy()
        calib_key = "args_0"
    else:
        tokenizer = open_clip.get_tokenizer(args.model)
        texts = ["a photo of a dog", "a screenshot of text", "food on a plate"]
        toks = [tokenizer([t]).to(torch.int32).numpy() for t in texts]
        calib = toks
        verify_in = toks[0]
        with torch.no_grad():
            ref = model.encode_text(torch.from_numpy(toks[0]).long()).float().numpy()
        calib_key = "args_0"

    from ai_edge_quantizer import quantizer, recipe

    log("quant", f"{args.recipe} on {args.tower} encoder")
    qt = quantizer.Quantizer(args.f32, getattr(recipe, args.recipe)())

    calib_result = None
    if qt.need_calibration:
        log("quant", "calibrating")
        calib_result = qt.calibrate({"serving_default": [{calib_key: a} for a in calib]})

    qt.quantize(calib_result, serialize_to_path=out_path)
    log("quant", f"wrote {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")

    got = run_tflite(out_path, verify_in)
    log("verify", f"cosine (int8 vs float PyTorch) = {cosine(ref, got):.5f}")
    log("done", "")


if __name__ == "__main__":
    main()
