#!/usr/bin/env bash
# Regenerate the on-device model assets that the app bundles.
#
# These are intentionally NOT committed to git: the MobileCLIP weights are under
# Apple's research-only license (redistribution restricted) and the .tflite files
# are large. This script reproduces them from open_clip so a fresh clone is one
# command away from buildable.
#
# Prereqs:
#   - the uv venv set up in this dir:  uv venv --python 3.12 .venv &&
#     VIRTUAL_ENV=.venv uv pip install -r requirements.txt
#   - a handful of images in calib_images/ (any ~20 photos; used only to compute a
#     verification embedding for the image encoder — weight-only needs no real
#     calibration set). e.g. `adb pull` a few from a device, or drop any jpgs in.
#
# Output: android/app/src/main/assets/{image_encoder,text_encoder}.tflite + bpe_merges.txt
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
export TF_ENABLE_ONEDNN_OPTS=0
MODEL=MobileCLIP2-S0
PRE=dfndr2b
ASSETS=../../android/app/src/main/assets

echo "==> image encoder: float32 -> weight-only int8"
$PY convert.py --model $MODEL --pretrained $PRE --tower image
$PY quantize_tflite.py --tower image --f32 output/${MODEL}_${PRE}_image_f32.tflite

echo "==> text encoder: float32 -> weight-only int8"
$PY convert.py --model $MODEL --pretrained $PRE --tower text
$PY quantize_tflite.py --tower text --f32 output/${MODEL}_${PRE}_text_f32.tflite

echo "==> extracting CLIP BPE merges"
$PY - <<'PYEOF'
import gzip, os, open_clip
p = os.path.join(os.path.dirname(open_clip.__file__), "bpe_simple_vocab_16e6.txt.gz")
merges = gzip.open(p).read().decode("utf-8").split("\n")[1 : 49152 - 256 - 2 + 1]
open("../../android/app/src/main/assets/bpe_merges.txt", "w").write("\n".join(merges))
print("wrote", len(merges), "merges")
PYEOF

echo "==> copying int8 models into app assets"
mkdir -p "$ASSETS"
cp output/${MODEL}_${PRE}_image_wonly.tflite "$ASSETS/image_encoder.tflite"
cp output/${MODEL}_${PRE}_text_wonly.tflite  "$ASSETS/text_encoder.tflite"

echo "done. assets:"
ls -lh "$ASSETS"/image_encoder.tflite "$ASSETS"/text_encoder.tflite "$ASSETS"/bpe_merges.txt
