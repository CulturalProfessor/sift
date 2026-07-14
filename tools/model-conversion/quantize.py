"""
Quantize a CLIP / MobileCLIP image tower to static int8 TFLite via PT2E.

Static full-int8 (weights + activations) is what gives the real CPU speedup on
the no-NPU target device, and needs a small representative calibration set. We
calibrate on real photos pulled from the target phone (--calib-dir), which is
the ideal distribution. Calibration images stay local and are gitignored.

Flow (torch 2.12 / litert-torch 0.9):
  export_for_training -> prepare_pt2e(quantizer) -> calibrate -> convert_pt2e
  -> litert_torch.convert(quant_config=QuantConfig(pt2e_quantizer=quantizer))

Usage:
  python quantize.py --model MobileCLIP2-S0 --pretrained dfndr2b \
      --calib-dir calib_images --calib-n 40
"""

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image


def log(stage: str, msg: str = "") -> None:
    print(f"[{stage}] {msg}", flush=True)


class ImageTower(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, image):
        return self.m.encode_image(image)


def load_calib(calib_dir: str, n: int, preprocess) -> list[torch.Tensor]:
    paths = sorted(
        p
        for ext in ("jpg", "jpeg", "png", "webp")
        for p in glob.glob(os.path.join(calib_dir, f"*.{ext}"))
        + glob.glob(os.path.join(calib_dir, f"*.{ext.upper()}"))
    )[:n]
    imgs = []
    for p in paths:
        try:
            imgs.append(preprocess(Image.open(p).convert("RGB")).unsqueeze(0))
        except Exception as e:  # skip unreadable files
            log("calib", f"skip {os.path.basename(p)}: {e}")
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MobileCLIP2-S0")
    ap.add_argument("--pretrained", default="dfndr2b")
    ap.add_argument("--calib-dir", default="calib_images")
    ap.add_argument("--calib-n", type=int, default=40)
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"{args.model}_{args.pretrained}_image".replace("/", "-")

    log("load", f"open_clip {args.model} / {args.pretrained}")
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model.eval()
    tower = ImageTower(model).eval()

    side = preprocess.transforms[0].size
    side = side if isinstance(side, int) else side[0]
    sample = torch.randn(1, 3, side, side)

    log("calib", f"loading up to {args.calib_n} images from {args.calib_dir}/")
    calib = load_calib(args.calib_dir, args.calib_n, preprocess)
    if not calib:
        raise SystemExit(f"No calibration images found in {args.calib_dir}/")
    log("calib", f"loaded {len(calib)} calibration images (input {side}x{side})")

    with torch.no_grad():
        ref = tower(sample).float().numpy()

    log("pt2e", "prepare_pt2e with static per-channel int8 config")
    import litert_torch
    from litert_torch.quantize import pt2e_quantizer
    from litert_torch.quantize.quant_config import QuantConfig
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

    # Per-tensor (is_per_channel=False): per-axis conv-weight quantization currently
    # trips the stablehlo->TFLite converter on the stem conv. Per-tensor is broadly
    # supported; revisit per-channel later if accuracy needs it.
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(
            is_per_channel=False, is_dynamic=False
        )
    )
    # torch 2.12 consolidated export_for_training into torch.export.export,
    # which now produces the training-compatible IR that prepare_pt2e needs.
    exported = torch.export.export(tower, (sample,)).module()
    prepared = prepare_pt2e(exported, quantizer)

    log("calib", "running calibration forward passes")
    with torch.no_grad():
        for img in calib:
            prepared(img)

    quantized = convert_pt2e(prepared)
    # Exported models don't support .eval() directly; use the pt2e helper so the
    # converter sees an eval-mode graph (better on-device perf/compatibility).
    from torchao.quantization.pt2e import move_exported_model_to_eval

    move_exported_model_to_eval(quantized)
    log("pt2e", "convert_pt2e done")

    log("convert-int8", "litert_torch.convert with int8 quant_config")
    edge = litert_torch.convert(
        quantized, (sample,), quant_config=QuantConfig(pt2e_quantizer=quantizer)
    )
    out = os.path.join(args.outdir, f"{tag}_int8.tflite")
    edge.export(out)
    log("convert-int8", f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")

    log("verify", "int8 TFLite vs float PyTorch on the same input")
    got = edge(sample)
    got = got.detach().numpy() if hasattr(got, "detach") else np.asarray(got)
    cos = float(
        np.dot(ref.ravel(), got.ravel())
        / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-9)
    )
    log("verify", f"cosine similarity (int8 vs float) = {cos:.5f}")
    log("done", "int8 quantization complete")


if __name__ == "__main__":
    main()
