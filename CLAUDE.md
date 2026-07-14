# Sift

On-device semantic search for your photos and videos — type or say what you're
looking for ("dog at the beach") and find it, including the exact timestamp
inside a video. Fully offline, no cloud, no API cost. Runs on budget/old
Android hardware, not just flagships.

## Why this project exists

Portfolio piece to diversify a backend-heavy resume (Nest.js, Next.js,
Postgres, Redis, BullMQ, AWS — queue systems, email platforms, infra work)
with AI + mobile skills that aren't currently represented at all. Picked
after a long exploration of alternatives (see "Ideas considered and
rejected" below) — chosen specifically because it's the one idea that's both
something the author would use daily (camera roll search is a near-universal
pain point) and demonstrates a genuinely new skill vector (multimodal
embeddings, on-device vector search) rather than re-proving networking/infra
skills already well-covered by existing projects (Minidoc, Go Torrent).

## Platform & target device

- React Native, **Android-first** — no iOS testing available (no Mac, no
  iOS device). Keep code cross-platform-friendly in structure, but don't
  invest in iOS-specific testing/polish yet.
- Primary test device: **Samsung Galaxy F22** (launched July 2021) —
  MediaTek Helio G80 (12nm, 2×Cortex-A75 @2.0GHz + 6×Cortex-A55 @1.8GHz),
  Mali-G52 MC2 GPU, 4GB or 6GB RAM, **no dedicated NPU**. This is a
  genuinely low-end chipset, not a flagship — treat it as the real stress
  test for the "works on old/budget phones, stays fast" requirement, not a
  theoretical target.
- Minimum spec to design for: ARM64, 4GB+ RAM, Android 10+. Below that,
  show a clear "below minimum spec" state rather than pretend it works.

## Core architecture

Two pipelines at different latency budgets — this split is the central
design decision, don't collapse it:

**1. Indexing (background, one-time per photo/video — can be slow)**
- MobileCLIP-style dual encoder (separate image encoder + text encoder).
  Image encoder runs once per photo when added to the library.
- Run via **TFLite/LiteRT** (preferred over ONNX Runtime — better Android
  fit, GPU delegate support for Mali). No NPU on the target device, so
  acceleration comes from GPU delegate (modest gains, Mali-G52 MC2 is
  entry-level) and CPU int8 SIMD — **int8 quantization is load-bearing
  here, not optional**, since there's no accelerator to hide costs behind.
- Store embeddings as int8 blobs in SQLite (asset ID + blob; ~512 bytes/photo
  → tens of MB even for large libraries).
- Incremental, not full re-index — only embed new photos.
- Background execution is unreliable (Doze/battery optimization throttles
  it even though Android is more permissive than iOS). Use `WorkManager`
  for opportunistic background indexing, but design the real UX around
  **foreground indexing with a visible progress bar** on first run /
  app-open, not silent always-on background indexing.
- Expect indexing to be noticeably slower on this device than on a
  flagship — that's fine and expected. The "fast" requirement applies to
  search, not to one-time bulk indexing.

**2. Search (foreground, per-query — must be fast, <300ms target)**
- Query text → text encoder (small, ~50-150ms) → embedding.
- **Brute-force cosine similarity** over stored int8 embeddings. No
  ANN/HNSW — at personal-library scale (thousands, even tens of thousands
  of items) a linear scan is a few tens of ms; an ANN index adds memory
  overhead for zero benefit at this scale. This was a deliberate call, not
  a shortcut — state the reasoning if asked about it.
- Return top-K above a relevance threshold, not top-K regardless of match
  quality.

**3. Video extension (the actual differentiator over plain photo search)**
- Adaptive keyframing: embed a candidate frame, compare to the last
  *stored* keyframe's embedding, only store if sufficiently different
  (scene changed). Reuses the same embedding machinery as photos. More
  interesting and more compute-efficient than fixed-interval sampling —
  especially important given no NPU on the target device.
- Store timestamp with each video keyframe embedding.
- A video search hit opens the video and **seeks to that timestamp** — this
  is the feature with little to no offline prior art (most competitors stop
  at static photo search).

**Model tiering**: smallest MobileCLIP variant on <4GB RAM devices, larger
variant on 6GB+, detected at runtime — same pattern used across other
on-device AI ideas explored for this project.

## Tooling

- **VSCode** as primary editor for the RN/TypeScript app code.
- Android SDK + platform-tools + build-tools + NDK — installed via a
  **one-time Android Studio installer run** (easiest way to avoid
  version-mismatch pain), but Android Studio itself is **not** the daily
  driver.
- Keep Android Studio around only for occasionally editing/debugging native
  Kotlin files (VSCode's Kotlin support is workable but weaker).
- Testing: **physical Galaxy F22 via USB debugging (`adb`)** — no
  emulator/AVD needed since a real (and appropriately low-end) device is
  available.
- Cost: **$0** — no APIs, no cloud, no Apple Developer account (Android-only
  for now avoids the $99/year iOS cost entirely).

## Naming

"Sift" — chosen over "Recollect"/"ReCollect" (already taken on the Play
Store by an app with nearly the same pitch: "AI that understands meaning,
not just keywords") and over "Trove"/"Seek" variants. Verify Play Store
listing name, npm package name, and domain availability before final
commit if not already done.

## Ideas considered and rejected during scoping (don't re-litigate these)

- **Remote-control/monitoring of Claude Code or Codex from phone** —
  rejected outright. Anthropic shipped this natively (Remote Control, Feb
  2026) and OpenAI shipped Codex in the ChatGPT mobile app (May 2026),
  plus third-party apps already exist (Nimbalyst, Happy, AgentsRoom,
  Paseo). No gap to fill.
- **Remote desktop, phone-to-laptop** — rejected outright. Extremely
  saturated (TeamViewer, AnyDesk, RustDesk, Chrome Remote Desktop,
  Splashtop), no AI angle at all, and requires costly relay/TURN server
  infra — contradicts the zero-cost constraint.
- **Semantic photo search alone (no video)** — crowded (ISearch, Rare
  Gallery, PicFinder AI, PhotoCHAT AI, Photoreka, Windows 11 Copilot+ PCs
  ship this at the OS level). Still fine as a portfolio *technique* demo,
  but the video-moment-search extension is what makes Sift differentiated
  rather than a clone.
- **Photo declutter (duplicate/blur detection)** — crowded (Sortify, AI
  Duplicate Photo Cleaner, CleanMyPhone, Unclutter).
- **Offline two-way voice translator** — deprioritized; Apple Translate and
  Google Translate already ship offline conversation mode natively.
- **Accessibility scene description for visually impaired users** — solid
  idea technically, but user wanted a broader (non-niche) audience, so
  parked in favor of Sift.
- **Fully local RAG "chat with your docs/notes"** — viable, and a
  `@react-native-rag/executorch` library already exists for the plumbing
  (so using it out of the box would be integration work, not engineering —
  differentiation would require implementing the vector search layer
  yourself, or picking a narrower niche like offline codebase chat).
  Parked, not built.
- **Low-latency voice assistant (on-device/hybrid STT→LLM→TTS with
  barge-in)** — technically strong idea, scoped in detail, but treated as
  a separate potential project, not merged into Sift.

## Status

Naming and architecture decided. No code written yet. Next step: scope
actual repo structure and build order (photo-search baseline first, prove
the pipeline end-to-end, then extend to video moment search).
