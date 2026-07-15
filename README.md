# Sift

**On-device semantic search for your photos and videos.** Type what you're
looking for, "dog at the beach", "handwritten notes", "sunset", and Sift
finds it, including the **exact moment inside a video**. Fully offline: no
cloud, no account, no API cost. Runs on budget/old Android hardware, not just
flagships.

---

## What it does

- **Semantic photo search**: searches by *meaning*, not filenames or tags.
  Uses a CLIP-style dual encoder so "food" finds your meals even with no
  metadata.
- **Video moment search (the differentiator)**: indexes representative
  keyframes of each video and, on a hit, opens the video **seeked to that
  timestamp**. Almost no offline app does this.
- **100% on-device**: your photos and videos never leave your phone. There is
  no server anywhere in the app.
- **Incremental indexing**: the first index is a one-time background pass;
  after that only new/changed items are embedded.
- **Runs on low-end hardware**: developed and tested on a Samsung Galaxy F22
  (MediaTek Helio G80, no NPU, 4 GB RAM).

## How it works

Two pipelines at different latency budgets:

**Indexing (background, one-time per item)**
1. Each photo → `MobileCLIP2-S0` image encoder (int8 TFLite via LiteRT) →
   512-d embedding, L2-normalized and quantized to int8 (~512 bytes) in SQLite.
2. Videos → **adaptive keyframing with a cheap pre-filter**: walk candidate
   frames, skip ones visually near-identical to the last embedded frame
   (16×16 grayscale diff, avoids the expensive encoder), and store a keyframe
   only when the *embedding* shows a real scene change. Each keyframe carries
   its timestamp.

**Search (foreground, per query, target <300 ms)**
1. Query text → CLIP BPE tokenizer (ported to TypeScript) → `MobileCLIP2-S0`
   text encoder (int8 TFLite) → 512-d embedding.
2. **Brute-force cosine** over all stored int8 embeddings, photos and video
   keyframes share one vector space, so a single query ranks both. At
   personal-library scale a linear scan is a few tens of ms; an ANN index would
   add memory for no benefit.
3. Results returned with a match score; video hits carry the best-matching
   moment's timestamp.

### Engineering highlights

- **Self-converted, self-quantized models.** MobileCLIP2-S0 (PyTorch) →
  float32 TFLite via `litert-torch` (perfect fidelity), then **int8** via
  `ai-edge-quantizer`. Finding: full activation quantization destroys CLIP/ViT
  accuracy (cosine ~0.1 to 0.3); **weight-only int8** preserves it (cosine ~0.99)
  at 3.6× smaller. Image encoder: 13 MB. Text encoder: 65 MB.
- **Preprocessing fidelity.** Matched the Python reference on-device to cosine
  0.978, the key was replicating torchvision's antialiased downsampling with a
  two-pass `inSampleSize` bitmap decode.
- **Native performance.** Decode + preprocess + inference all run in Kotlin
  (LiteRT, multi-threaded XNNPACK). Indexing runs at background thread priority
  with a device-tuned thermal throttle; SQLite is in WAL mode so search stays
  responsive during a long index.
- **CLIP BPE tokenizer in TypeScript**: a faithful port of open_clip's
  `SimpleTokenizer`, verified byte-for-byte against Python.

## Tech stack

React Native (Android-first) · Kotlin native module · LiteRT / TensorFlow Lite
· MobileCLIP2-S0 (int8) · SQLite · ExoPlayer (media3) · TypeScript.

## Build & run

```bash
# 1. Generate the bundled model assets (not committed, see "Models" below)
cd tools/model-conversion
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt
# put ~20 images in calib_images/ (any photos), then:
./build_assets.sh

# 2. Install JS deps and run on a connected Android device
cd ../..
npm install
npm start            # Metro, in one terminal
npm run android      # build + install, in another
```

Requires the Android SDK + NDK. Minimum device spec: ARM64, Android 10+.

### Release APK

```bash
cd android && ./gradlew assembleRelease
# -> android/app/build/outputs/apk/release/app-release.apk (~150 MB, all ABIs)
```

## Models

The `.tflite` encoders and BPE merges are **not committed**: the MobileCLIP
weights are under Apple's research-only license (redistribution restricted),
and the files are large. `tools/model-conversion/build_assets.sh` reproduces
them byte-for-byte from open_clip. To publish/distribute, swap to OpenAI CLIP
ViT-B/32 (MIT-licensed); the conversion pipeline is model-agnostic.

## Privacy

Everything runs on the device. There is no network code, no analytics, no
account. The search index lives in the app's private storage and is built by
scanning the device's own gallery. Sharing the APK indexes the recipient's
photos fresh; it carries no one else's data.
