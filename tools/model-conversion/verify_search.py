"""
End-to-end sanity check for the int8 image encoder: does semantic search return
the right photos?

Encodes every image in --img-dir with the quantized TFLite image encoder, then
for each text query (PyTorch text encoder as reference) ranks the images by
cosine similarity and copies the top-K into output/search_test/<query>/ so you
can open the folder and eyeball whether the matches make sense.

This is the real product function, so it's the most meaningful verification:
if "food" surfaces your food photos and "screenshot" surfaces screenshots, the
int8 embeddings live in the correct semantic space.

Usage:
  python verify_search.py --img-dir calib_images --topk 3 \
      --queries "food" "a screenshot of text" "people" "gym equipment"
"""

import argparse
import glob
import os
import shutil

import numpy as np
import torch
from PIL import Image


def log(msg):
    print(msg, flush=True)


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MobileCLIP2-S0")
    ap.add_argument("--pretrained", default="dfndr2b")
    ap.add_argument(
        "--tflite", default="output/MobileCLIP2-S0_dfndr2b_image_wonly.tflite"
    )
    ap.add_argument(
        "--text-tflite", default="output/MobileCLIP2-S0_dfndr2b_text_wonly.tflite"
    )
    ap.add_argument("--img-dir", default="calib_images")
    ap.add_argument("--topk", type=int, default=3)
    # Prompt template markedly improves CLIP retrieval over a bare query.
    ap.add_argument("--template", default="a photo of {}")
    ap.add_argument(
        "--queries",
        nargs="+",
        default=["food", "a screenshot of text", "people", "gym or exercise"],
    )
    ap.add_argument("--outdir", default="output/search_test")
    args = ap.parse_args()

    import open_clip
    from ai_edge_litert.interpreter import Interpreter

    log(f"loading tokenizer + preprocess for {args.model}")
    _, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)

    # Both towers now run as int8 TFLite — exactly the on-device path.
    img_it = Interpreter(model_path=args.tflite)
    img_it.allocate_tensors()
    inp, out = img_it.get_input_details()[0], img_it.get_output_details()[0]
    txt_it = Interpreter(model_path=args.text_tflite)
    txt_it.allocate_tensors()
    tinp, tout = txt_it.get_input_details()[0], txt_it.get_output_details()[0]

    def encode_image_tflite(arr):
        if inp["dtype"] != np.float32:
            s, z = inp["quantization"]
            arr = np.round(arr / s + z).astype(inp["dtype"])
        img_it.set_tensor(inp["index"], arr)
        img_it.invoke()
        raw = img_it.get_tensor(out["index"])
        if out["dtype"] != np.float32:
            s, z = out["quantization"]
            return (raw.astype(np.float32) - z) * s
        return raw.astype(np.float32)

    def encode_text_tflite(text):
        toks = tokenizer([args.template.format(text)]).to(torch.int32).numpy()
        txt_it.set_tensor(tinp["index"], toks.astype(tinp["dtype"]))
        txt_it.invoke()
        raw = txt_it.get_tensor(tout["index"])
        if tout["dtype"] != np.float32:
            s, z = tout["quantization"]
            return (raw.astype(np.float32) - z) * s
        return raw.astype(np.float32)

    paths = sorted(
        p for e in ("jpg", "jpeg", "png") for p in glob.glob(f"{args.img_dir}/*.{e}")
    )
    log(f"encoding {len(paths)} images with the int8 TFLite image encoder...")
    embs = []
    for p in paths:
        arr = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
        embs.append(encode_image_tflite(arr).ravel())
    img_emb = l2(np.array(embs))

    if os.path.isdir(args.outdir):
        shutil.rmtree(args.outdir)

    for q in args.queries:
        tvec = l2(encode_text_tflite(q).ravel())
        sims = img_emb @ tvec
        order = np.argsort(-sims)[: args.topk]
        qdir = os.path.join(args.outdir, q.replace(" ", "_")[:40])
        os.makedirs(qdir, exist_ok=True)
        log(f'\nquery: "{q}"')
        for rank, i in enumerate(order):
            name = os.path.basename(paths[i])
            log(f"  #{rank + 1}  score={sims[i]:.3f}  {name}")
            shutil.copy(paths[i], os.path.join(qdir, f"{rank + 1}_{sims[i]:.2f}_{name}"))

    log(f"\nTop matches copied to {args.outdir}/<query>/ — open them to eyeball.")


if __name__ == "__main__":
    main()
