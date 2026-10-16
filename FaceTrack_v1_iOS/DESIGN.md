# FaceTrack — on-device face detection + SAM-style tracking for iPhone 17

**Design document.** What the system does, how each part works, and why each decision beat the alternatives.

---

## 1. What we are building

A live camera app that runs entirely on the phone's Neural Engine, GPU and CPU (no server). It:

1. **Detects** every face in the camera frame with a small, openly licensed detector.
2. **Tracks** each face across frames with a SAM-family video segmentation model. Each face keeps its ID and gets a pixel-accurate mask, including through head turns and short occlusions, which is where detectors fail.

The requirements, and where each one is handled:

| Requirement | How it is met | Section |
|---|---|---|
| Open models only | YuNet (MIT) + EdgeTAM (Apache-2.0), both code and weights | §3 |
| Detector must be very light yet effective | YuNet: 53 k parameters, 0.41 GFLOPs at 384×640, WIDER Face AP 0.884 / 0.866 / 0.750 | §3.1 |
| Tracker must be SAM-based | EdgeTAM, Meta's on-device SAM 2 variant (CVPR 2025) | §3.2 |
| Very efficient | 4-graph streaming export, shared per-frame work, gated compute, fp16 end to end | §4, §6 |
| Very effective | Matches Meta's official tracker (mean IoU 0.9999); detector keeps the tracker from drifting | §4.6, §5 |

---

## 2. The system at a glance

```
                camera 720x1280 BGRA @30fps  (latest frame wins; late frames are dropped)
                                 │
            ┌────────────────────┴─────────────────────┐
            │ vImage letterbox 384x640                  │ vImage squash 512x512 (§4.7)
            ▼                                           ▼
  ┌───────────────────┐   runs in parallel   ┌────────────────────────────────┐
  │ YuNet  (GPU)      │◀────────────────────▶│ EdgeTAM image encoder  (ANE)   │  once per frame,
  │ every 3rd frame   │                      │ RepViT-M1 + FPN + shared       │  shared by all faces.
  │ (every frame when │                      │ layer-0 self-attention         │  Skipped entirely when
  │  nothing tracked) │                      └───────────────┬────────────────┘  no faces are present
  └─────────┬─────────┘                                      │
            │ boxes                                          │ feat, tgt0, hr0, hr1
            ▼                                                ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ TrackManager: associate → re-anchor / propagate → spawn → retire             │
  │                                                                              │
  │   per face:  memory attention ──▶ mask decoder ──▶ memory encoder            │
  │              (reads the face's      (mask, IoU,      (writes a new memory    │
  │               memory bank)           object score)    into the bank)         │
  └──────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                SwiftUI overlay: ID, box, mask, occluded state
```

The detector and the tracker cover each other's weaknesses:

* **The detector** finds new faces, confirms that tracks are still faces, and re-anchors a track when it starts to drift.
* **EdgeTAM** keeps each face's identity and mask between detections. It also covers poses and occlusions the detector misses (YuNet is weak on extreme pose and occlusion, per your comparison doc).

---

## 3. Choosing the models

### 3.1 Detector: YuNet (OpenCV Zoo, 2023mar, MIT)

| Candidate | License (code / weights) | Size | Why it lost (or won) |
|---|---|---|---|
| **YuNet** ✅ | MIT / MIT | 53 k params, ~230 KB | Tiny and multi-scale (faces ~10–300 px). Returns 5 landmarks. WIDER Face val AP 0.884 / 0.866 / 0.750 (easy / medium / hard). Converts cleanly to Core ML. |
| BlazeFace short-range (MediaPipe) | Apache-2.0 | ~0.2 MB | 128×128 input, built for selfie distance (≲2 m). Small or far faces are missed, which rules out group shots and the rear camera. |
| BlazeFace full-range | Apache-2.0 | ~1.1 MB | Covers longer range, but anchor decoding is coarser and it has fewer maintained ports. YuNet matches its range at less compute. |
| SCRFD-0.5/2.5 GF (InsightFace) | MIT / **non-commercial** weights | 0.6–0.8 M | Best accuracy per FLOP, but the weights are not openly licensed. Fails the "openly available" requirement. |
| RetinaFace-MobileNet0.25 | MIT / WIDER-derived | 0.4 M | Slower than SCRFD and less accurate than YuNet per FLOP. |
| YOLOv8n-face / YOLO-face variants | **AGPL** + WIDER terms | 3 M+ | AGPL is viral, and the model is 50× larger than it needs to be. |
| Apple Vision `VNDetectFaceRectanglesRequest` | closed | – | Not an open model. |

**Decision:** YuNet. It is the most accurate fully permissive detector at this size, and at 0.41 GFLOPs it costs about 0.3 % of one EdgeTAM frame. That means the detector is never what limits the frame rate.

**How it is converted** (`conversion/convert_yunet.py`):

* The ONNX graph is loaded with `onnx2torch`. Its two dynamic ops (Resize, Reshape) are replaced with trace-friendly equivalents. This works because YuNet's scales and channel counts are constants.
* **Box decoding is moved inside the Core ML graph.** The graph applies `score = √(cls·obj)`, grid offsets, `exp(w,h)·stride` and the landmark offsets itself. It outputs `[1, 5040, 15]` rows of `score, x, y, w, h, 5×(lx, ly)` in input pixels, so the phone only has to threshold and run NMS on a few candidates.
* The decode matches `cv2.FaceDetectorYN` to 1.5e-5 px on three test images, including an 18-face crowd.
* Input is 384×640 portrait (the multiple of 32 nearest to 360×640). The 720×1280 camera frame is scaled by 0.5 and placed top-left, with 24 px of black padding on the right.

### 3.2 Tracker: EdgeTAM (Meta, CVPR 2025, Apache-2.0)

| Candidate | On-device speed | Video quality (J&F) | Why it lost (or won) |
|---|---|---|---|
| **EdgeTAM** ✅ | **16 FPS on iPhone 15 Pro Max** (22× faster than SAM 2) | DAVIS17 87.7 · MOSE 70.0 · SA-V val 72.3 · SA-V test 71.7 | Built specifically for phones: RepViT-M1 backbone, 2 memory-attention blocks, and a 2D spatial perceiver that shrinks each memory from 4096 tokens to 512. The checkpoint and code are Apache-2.0. |
| EfficientTAM-Ti/S | ~10 FPS on iPhone 15 Pro Max | SA-V test ≈ SAM 2 B+ | Also a strong option, but slower than EdgeTAM on the same phone (the plain ViT encoder is heavier). A reasonable swap if you need a bit more accuracy. |
| SAM 2 / 2.1 (Hiera) | EdgeTAM reports being 22× faster than SAM 2 on iPhone | slightly higher | The Hiera encoder and 4 full-resolution memory-attention layers are too heavy for real time on a phone. |
| SAM 3 / SAM 3.1 | datacenter GPU (H100/H200) | adds concept prompts | Built for data-centre GPUs. Detection by text concepts is more than face tracking needs. |
| MobileSAM / EfficientSAM | fast | **no video memory** | Image-only. "Tracking" would mean re-prompting from scratch every frame, with no identity, no occlusion handling and no temporal consistency. |

**Decision:** EdgeTAM. Of the openly licensed SAM-family *video* models, it has the fastest published iPhone results (16 FPS vs ~10 FPS for EfficientTAM), with accuracy close to SAM 2.

---

## 4. Getting EdgeTAM onto Core ML as a real tracker

### 4.1 The problem with the official export

EdgeTAM's repository ships `coreml/export_to_coreml.py`, but it only exports the **image encoder, prompt encoder and mask decoder**, which is single-image SAM. Its "video tracking" demo re-segments each frame from the clicked points and never uses the memory bank, so it does not track in the SAM 2 sense. There are also two smaller issues:

* It skips the ImageNet mean/std normalisation.
* Its prompt-encoder wrapper ignores the box input.

The memory path that makes SAM 2 a tracker is the memory encoder, the spatial perceiver and memory attention. That path had to be exported from scratch (`conversion/edgetam_coreml_modules.py`).

### 4.2 Four static-shape graphs

| Graph | Runs | Inputs → outputs | GFLOPs |
|---|---|---|---|
| `EdgeTAMImageEncoder` | once per frame | image 1024² → `feat`, `feat_init`, `tgt0`, `hr0`, `hr1` | 58.1 (11.3 at 512, §4.7) |
| `EdgeTAMMemoryAttention` | once per face | `tgt0`, memory bank, slot mask → `pix_feat` | 74.1 (default) / 56.6 (fast) |
| `EdgeTAMMaskDecoder` (2 functions) | once per face | `pix_feat`, `hr0/1`, points → 4 masks, 4 IoUs, 4 pointers, object score | 3.6 |
| `EdgeTAMMemoryEncoder` | once per face | `feat`, chosen mask → 512×64 memory tokens | 12.1 |
| YuNet (for scale) | every 3rd frame | 384×640 → candidates | 0.41 |

Why split it this way instead of one big graph:

* **Work that is shared across faces runs once.** Everything that depends only on the frame lives in the encoder, and everything that depends on a face's history lives in the three per-face graphs.
* **Every shape is static.** The Neural Engine only accelerates fixed shapes; a dynamic shape falls back to GPU or CPU. The memory bank's variable length is handled with masking (§4.3), not with dynamic shapes.
* **Graph boundaries sit where the tensors are smallest** and where the app needs control flow: picking the best of 4 masks, the stability test, and ring-buffer updates.

### 4.3 Tricks that keep the export exact *and* fast

**(a) Memory bank with fixed slots and an attention mask (exact).**
SAM 2 concatenates a variable number of memories: conditioning frames, up to 6 recent frames and up to 16 object pointers. The export instead always passes a fixed layout:

```
[ cond 0 | cond 1 | recent 0 (newest) … recent 5 | 16 pointer slots ]
   512      512      512 each                       4 tokens each      = 4160 keys × 64-d
```

Each slot gets a validity flag. Empty slots get −10⁴ added to their attention logits, so their softmax weight is 0 (the flag is expanded to per-token bias inside the graph). Masking a key out of a softmax is mathematically identical to never concatenating it, so a static shape costs no accuracy.

**(b) Positional encodings are constants and are baked in.**

* The perceiver's position codes depend only on the grid shape: zeros for the 256 global latents and a 16×16 sine grid for the 256 2-D latents.
* The temporal encodings depend only on slot index: conditioning slots use `t_pos = 0`, and recent slot *k* uses the encoding of "k+1 frames ago".
* Object pointers carry no temporal encoding in EdgeTAM's config.

So the whole `memory_pos` tensor is a compile-time constant. The app never computes or uploads positional data.

**(c) Complex-number RoPE rewritten with real numbers.**
EdgeTAM's rotary attention uses `torch.polar` and `view_as_complex`, which Core ML cannot represent. The export rewrites it with the identity `(a,b) → (a·cosθ − b·sinθ, a·sinθ + b·cosθ)` and precomputed interleaved cos/sin tables.

* Queries use a 64×64 grid.
* Keys use a 16×16 grid on each slot's 2-D latents only. Global latents and pointer tokens get cos=1, sin=0, which reproduces `num_k_exclude_rope` and the v2 per-frame repeat exactly.

**(d) The first self-attention block is hoisted into the encoder (about 19 GFLOPs saved per extra face).**
Memory-attention layer 0 starts with a self-attention over the current frame's tokens, and that block never sees the memory. It is therefore identical for every face, so it runs once inside the encoder graph (`tgt0`). Only cross-attention and layer 1 run per face.

**(e) Prompt encoder folded into the decoder, with two functions in one package.**
Box prompts are encoded the way `SAM2VideoPredictor` does it: two corner points labelled 2 and 3, plus the automatic padding point. Tracking frames use a single padding point (label −1). Because the token count differs (1 vs 2 input points), the decoder ships as a Core ML **multifunction** package (`track` and `prompt`) that stores the weights once. `point_embeddings[label]` is rewritten with `where()` so it converts cleanly instead of using boolean index assignment.

**(f) Occlusion gating inside the graph.**
When the object score is ≤ 0, the graph itself sets the masks to −1024 and the pointer to `no_obj_ptr`, matching SAM 2's `fixed_no_obj_ptr` behaviour. The app never has to handle those tensors.

**(g) fp16 end to end.**
Every tensor passed between graphs is declared fp16 (`hr0` alone is 2 M values). The app passes `MLMultiArray` outputs directly as inputs to the next graph, with no fp32 round trip and no copies. Only the point coordinates stay fp32, because fp16 has only 0.5-px resolution near 1024.

**(h) Chunked attention.**
Softmax is computed per query row, so splitting the 4096 queries into blocks of 512 gives exactly the same result (verified bit-identical). Each block's score matrix is about 4 MB instead of about 34 MB, small enough to stay in the Neural Engine's on-chip memory. On iPhone 17 this cut memory attention from 44 ms to 34.5 ms (§6.3). The block size is set with `export_edgetam.py --attn-chunk`.

### 4.4 SAM 2 post-processing, replicated

`build_sam2_video_predictor` enables several behaviours that the plain model does not. All of them are reproduced:

| Behaviour | Where |
|---|---|
| Single-mask output with dynamic fallback to the best multimask output when stability < 0.98 (box prompts) | `EdgeTAM.addBox` (Swift) / `reference_tracker.add_box` |
| Multimask output while tracking; the pointer comes from the chosen mask's token | `EdgeTAM.track` |
| The pointer for prompted frames always comes from the single-mask token | `addBox` uses `obj_ptr[0]` |
| Masks from prompts are **binarised** before memory encoding; tracked masks use `sigmoid·20−10` | `binarize` input of the memory-encoder graph |
| `fill_hole_area=8` | Not used (a CUDA-only op that only touches the displayed mask, never the memory) |

### 4.5 Memory policy for faces

| Slot type | Paper setting | FaceTrack default | Why |
|---|---|---|---|
| Conditioning (prompted) frames | 1 (the first click) | **2: the first anchor + the latest detector re-anchor** | A face's look changes with pose and lighting. Keeping the first anchor preserves identity, and keeping the newest re-anchor keeps appearance current. |
| Recent frames | 6 | 6 | Unchanged. |
| Object pointers | 16 | 16 | Unchanged; they are cheap (4 tokens each). |

The trade-off is set when exporting:

* `--num-cond 1 --num-recent 6` reproduces the paper exactly.
* `--num-cond 1 --num-recent 3` is the "fast" preset: 2112 keys instead of 4160, and 24 % fewer memory-attention FLOPs.

### 4.6 Validation (done on this export)

| Check | Result |
|---|---|
| YuNet in-graph decode vs `cv2.FaceDetectorYN` (3 images, 20 faces) | Identical detections, max box error 1.5e-5 px |
| Streaming EdgeTAM (fixed slots, real RoPE, hoisted SA) vs Meta's **official `SAM2VideoPredictor`**, 40-frame clip that includes a full occlusion and re-appearance | **Mean mask IoU 0.9999, min 0.9992**, max logit difference 0.06. Both agree the face disappears on frames 18–22 and both recover it. |
| Same clip, simulated fp16 (all four graphs in half precision) vs fp32 | Mean IoU 0.994, min 0.933, no overflow (largest activation is the −1024 sentinel) |
| Full pipeline (YuNet + track manager + EdgeTAM) on a two-face clip with occlusion | Both faces tracked; face #1 keeps its ID through a 5-frame blackout (see `docs/reference_pipeline_occlusion.jpg`) |

`conversion/verify_coreml_mac.py` repeats the torch-vs-Core ML comparison with the real `.mlpackage` files on your Mac, and reports per-graph latency for each compute unit. Run it once after any re-export (see README).

### 4.7 Input size: 512 instead of 1024 (current default)

EdgeTAM's checkpoint runs at 768 or 512 input without retraining; only the input size and the RoPE grid sizes change (shown first by the unofficial [faster-edgetam](https://github.com/korhansevinc/faster-edgetam) fork). Faces are large, smooth regions, so 1024 is more resolution than tracking them needs. Every graph shrinks with the feature grid (S/16) and mask grid (S/4); the memory bank stays 512 tokens per frame.

| Input size | Encoder | Memory attention / face | Memory encoder / face | Decoder / face | 1 face total |
|---|---|---|---|---|---|
| 1024 (paper) | 58.1 | 74.1 | 12.1 | 3.6 | 147.9 GFLOPs |
| 768 | 28.4 | 37.7 | 6.9 | 2.1 | 75.1 GFLOPs |
| **512 (default)** | 11.3 | 15.7 | 3.2 | 0.9 | **31.1 GFLOPs (4.8× less)** |

Checks run on the export:

| Check | Result |
|---|---|
| Streaming export vs Meta's official `SAM2VideoPredictor` at the same size (40-frame portrait clip with occlusion) | 512: mean IoU 1.000, min 0.998 · 768: mean 1.000, min 1.000 |
| Tracking accuracy against the **true** face positions (composited group clip, 5 faces 34–92 px wide, 4-frame occlusion) | 1024, 768 and 512 all keep every face in 100 % of visible frames and all drop it during the occlusion; mean centre error 0.04–0.20 face widths at every size (512 no worse than 1024) |
| Mask agreement with the 1024 output | Selfie face: 0.95 mean IoU at 512, 0.98 at 768. Differences are at the mask edge (how much hair and neck is included), not in where the face is |

512 is the default (`export_edgetam.py --image-size 512`, read by the app from `edgetam_config.json`). Use `--image-size 1024` to restore the paper setting; the previous 1024 models are kept in `conversion/models_1024/`.

---

## 5. Runtime: what happens every frame

### 5.1 Per-frame steps (`Pipeline.process`)

1. **Drop or accept.** If the previous frame is still in flight, the new one is dropped. Latency stays at one frame instead of growing.
2. **Scale.** vImage (NEON) produces the 384×640 letterbox (if the detector is due) and the 1024×1024 squash (if any track exists). Both come from IOSurface-backed `CVPixelBufferPool`s, so nothing is allocated per frame and Core ML reads the buffers without copying.
3. **Detect and encode concurrently.** YuNet runs on the **GPU** at the same time as the EdgeTAM encoder on the **Neural Engine**, so detection costs almost no extra wall-clock time.
4. **No faces tracked?** The encoder is skipped entirely and only YuNet runs (0.4 GFLOPs). The encoder runs only once a detection appears.
5. **Update the tracks** (`TrackManager.update`), described next.

### 5.2 TrackManager policy

For each frame:

1. **Associate.** Match detections to tracks greedily by IoU against each track's last box (IoU ≥ 0.30).
2. **Update existing tracks.** For each track:
   * If it is matched *and* either its box disagrees with the detection (IoU < 0.50) or its own mask-quality estimate is low (< 0.60), **re-anchor** it. The detection box becomes a new conditioning prompt on top of the existing memory, like a SAM 2 correction click, with at least 0.5 s between re-anchors.
   * Otherwise, **propagate** it one frame with EdgeTAM.
3. **Spawn** a track (conditioning frame, `feat_init`) for each unmatched detection with score ≥ 0.80, up to 4 tracks.
4. **Retire** tracks that are:
   * *occluded*, with object score ≤ 0 for more than 1 s;
   * *unconfirmed*, with no matching detection for more than 5 s while EdgeTAM's own quality estimate is below 0.80, or 30 s while it is above. This is the guard against a mask drifting onto hair or a neck. The longer limit for confident tracks exists because YuNet misses tilted (~90°) and profile faces that EdgeTAM keeps tracking correctly; this was found on the first real-camera run.
   * *duplicates*, where two visible tracks overlap by IoU > 0.60; the younger one is dropped.

The reasoning behind each threshold:

| Knob | Value | Reasoning |
|---|---|---|
| `detInterval` | 3 | Faces rarely appear and disappear within 100 ms. Every 3rd frame at 30 fps finds new faces within ~0.1 s at one third of the detector cost. |
| `detScore` | 0.80 | Only confident detections start tracks, because a false track costs about 90 GFLOPs per frame until it is retired. Tracks that already exist are kept alive by EdgeTAM, not by the detector. |
| `reanchorIoU` / `minTrackIoUPred` | 0.50 / 0.60 | Re-prompt only on real disagreement. Every re-anchor overwrites the second conditioning slot. |
| `reanchorMinGap` | 0.5 s | Prevents re-anchoring on consecutive detections (jitter) and keeps recent memories meaningful. |
| `maxMissing` | 1 s | SAM 2's object-score head handles occlusion well (validated in §4.6). A 1 s grace period covers people walking past the camera. |
| `maxUnconfirmed` / `maxUnconfirmedConfident` | 5 s / 30 s | Tilted and profile faces can go undetected for a long time while EdgeTAM still tracks them correctly, so confident tracks get the long limit. |

All time limits are in seconds rather than frames, so the policy behaves the same at 12 FPS (Mac) or 30 FPS.
| `maxTracks` | 4 | Cost scales linearly with the number of faces (§6.1). |

The same policy is implemented in Python (`conversion/pipeline_reference.py`), so thresholds can be tuned on recorded clips on a Mac before touching Swift.

### 5.3 Coordinate systems

| Space | Size | Used by |
|---|---|---|
| Camera frame | 720×1280, portrait, mirrored for the front camera | Stored track boxes, overlay (aspect-fill like the preview layer) |
| Detector input | 384×640, letterboxed top-left, scale 0.5 | YuNet only; boxes are divided by the scale |
| Model space | S×S (S = 512 by default, §4.7), squashed (non-uniform), exactly like SAM 2's video loader | Box prompts: `x·S/720`, `y·S/1280` |
| Mask grid | (S/4)×(S/4) over the squashed frame (128×128 at 512) | Mask boxes: `x·720/(S/4)`, `y·1280/(S/4)`; overlay stretches the mask image over the frame rect |

---

## 6. Efficiency, quantified

### 6.1 Where the compute goes

Measured with `torch.utils.flop_counter` on the exported modules:

| | GFLOPs |
|---|---|
| Per frame, shared: EdgeTAM encoder + hoisted self-attention | 58.1 |
| Per frame, shared: YuNet (amortised over 3 frames) | 0.14 |
| Per face: memory attention (default / fast) | 74.1 / 56.6 |
| Per face: mask decoder | 3.6 |
| Per face: memory encoder | 12.1 |
| **1 face** | **≈ 148** (fast preset: ≈ 131) |
| **4 faces** | **≈ 418** (fast preset: ≈ 348) |
| No faces | 0.41 (detector only) |

What this means:

* **The per-face memory attention dominates.** That matches the EdgeTAM paper, which identifies cross-attention to memory as the on-device bottleneck.
* The two levers that matter most are the **memory-slot preset** and **`maxTracks`**.
* Hoisting layer-0 self-attention (§4.3d) saves about 19 GFLOPs for each face beyond the first.

### 6.2 Every efficiency decision in one place

| Decision | Saves |
|---|---|
| Run the EdgeTAM encoder only when a face exists | ~99.7 % of compute on empty scenes |
| Detector every 3rd frame, and on the GPU in parallel with the ANE encoder | Detector wall-clock ≈ 0 |
| Share the encoder and layer-0 self-attention across faces | 58 + ~19 GFLOPs for each face beyond the first |
| Spatial perceiver memories (512 instead of 4096 tokens per frame) | 8× shorter keys (inherited from EdgeTAM) |
| Static shapes plus masking instead of dynamic shapes | Keeps every graph on the Neural Engine |
| fp16 tensors between graphs, fed output-to-input without copies | Tens of MB of fp32↔fp16 conversion and copying per face per frame |
| 720p capture (the models see at most 1024²) | Sensor readout, ISP and memory bandwidth |
| Latest-frame-wins dropping | No latency build-up when a frame runs long |
| Pooled IOSurface pixel buffers + vImage | No per-frame allocations; zero-copy into Core ML |

### 6.3 Measured on iPhone 17

Measured with the app's own per-stage timers (printed to Xcode's console as `[FaceTrack] stages`), one face in view, default memory preset:

| Stage | Runs on (Core ML compute plan) | Unchunked attention | Chunked attention (512 queries) |
|---|---|---|---|
| YuNet (in parallel with the encoder) | GPU 98 % | 8 ms | 7.5 ms |
| Image encoder + shared self-attention | ANE 100 % | 30 ms | 27.6 ms |
| Memory attention, per face | ANE 100 % | 44 ms | **34.5 ms** |
| Memory encoder, per face | ANE 100 % | 7.5 ms | 6.2 ms |
| Mask decoder, per face | ANE 100 % | 4.4 ms | 4.2 ms |
| Memory assembly + CPU post-processing, per face | CPU | 1.2 ms | 1.2 ms |
| **Total per frame** | | **88 ms (≈ 9.3 FPS)** | **75 ms (≈ 10 FPS)** |

What the numbers say:

* **Every EdgeTAM graph runs entirely on the Neural Engine**, so the cost is the work itself, not CPU fallback.
* **Memory attention is the bottleneck** (about 75 % of the per-face time). Its unchunked 4096 × 4160 score matrix (~34 MB in fp16) did not fit in the Neural Engine's on-chip memory. Chunking the queries into blocks of 512 (§4.3h) cut it by 22 % with bit-identical results.
* **The displayed FPS is below 1000 / total** because the app waits for the next camera frame after finishing one (camera at 30 FPS).

At 1024 this is below EdgeTAM's published 16 FPS on iPhone 15 Pro Max. Part of the gap is the larger memory bank (2 conditioning + 6 recent slots vs 1 + 6). At 512 with 60 FPS capture (current defaults) the app processes a frame in about 23 ms and shows about 30 FPS with one face.

**Input size 1024 vs 512 (§4.7)**, measured back to back on the same phone (iOS 27.0.1), one face, chunked attention:

| Stage | 1024 | 512 | Change |
|---|---|---|---|
| YuNet (in parallel with the encoder) | 7.8 ms | 14–15 ms | slower (see below) |
| Image encoder + shared self-attention | 27.8 ms | 9.3–9.7 ms | −66 % |
| Memory attention, per face | 34.9–35.3 ms | 18.1–19.2 ms | −47 % |
| Memory encoder, per face | 6.2 ms | 2.6 ms | −58 % |
| Mask decoder, per face | 4.2–4.4 ms | 2.2–2.4 ms | −46 % |
| Memory assembly (CPU), per face | 1.1 ms | 5.2–5.9 ms | slower (see below) |
| **Total per frame** | **75.9 ms (10.0 FPS)** | **42.3–43.7 ms (15.2 FPS)** | **−44 %, +52 % FPS** |

The Neural Engine stages shrink by 46–66 %, less than the 4.8× drop in GFLOPs, because fixed per-call overheads don't shrink.

Two CPU-side stages got slower at 512: memory assembly (1.1 → 5.5 ms) and YuNet (7.8 → 14.5 ms, now on the critical path every third frame). Both read Core ML outputs one small piece at a time, each piece taking a lock on the output buffer: 512 locked reads per memory slot, and 5,040 × up to 15 per YuNet frame. Reading each output in one access (`MLArray+Utils.swift`, `FaceDetector.swift`) fixed both:

| Stage (512, one face) | Before the fix | After |
|---|---|---|
| Memory assembly, per face | 5.2–5.9 ms | **0.1–0.2 ms** |
| YuNet | 14–15 ms | **9.5–10.1 ms** |
| Per face (all stages) | 28.4–30.1 ms | 23.6–23.7 ms |
| **Total per frame** | 42.3–43.7 ms | **37.4–37.7 ms** |
| Displayed FPS | 15.2 | 15.0–15.1 |

**Why the FPS didn't move:** the camera delivers a frame every 33.3 ms (30 FPS) and the app always takes the newest one. At 37.5 ms per frame the app finishes just after a frame arrives and waits for the next, so it processes every second frame: 15 FPS. Getting below 33 ms per frame (30 FPS), or capturing at 60 FPS (a frame every 16.7 ms, so about 20 FPS at 37.5 ms), turns the saved time into visible FPS.

**Capturing at 60 FPS** (`CameraController.selectFrameRate`, falls back to the best rate the camera offers) fixed that, and did more than expected:

| 512, one face | Camera 30 FPS | Camera 60 FPS |
|---|---|---|
| YuNet | 9.5–10.1 ms | 6.2–7.0 ms |
| Image encoder | 10.1–10.2 ms | 6.1–8.3 ms |
| Memory attention, per face | 16.8–17.7 ms | 11.9–13.8 ms |
| Memory encoder / decoder, per face | 2.8 / 3.1 ms | 1.7 / 1.4 ms |
| **Total per frame** | **37.5 ms** | **22.7–23.0 ms** (one dip to 30.6 ms) |
| **Displayed FPS** | **15** | **30** (one dip to 23.9) |

The models didn't change, yet every Neural Engine stage got about 35 % faster. The likely reason: at 30 FPS the chip sat idle for ~29 ms between frames and clocked down; at 60 FPS the next frame is always waiting, so it stays busy at full speed. This is inferred from the timings, not measured directly. At ~23 ms per frame the app now processes every second 60 FPS frame, so 30 FPS. Reaching 60 FPS would need under 16.7 ms per frame.

The on-screen HUD shows FPS and milliseconds for detection, encoding, per-face work and total. `ComputePlanReport.swift` prints where every model's operations are placed (`[FaceTrack] plan …`).

If you need more speed, in order of impact:

1. **Fast memory preset:**
   ```bash
   python export_edgetam.py --num-cond 1 --num-recent 3
   ```
   Cuts memory-attention FLOPs by 24 %, with a small accuracy loss on long occlusions.
2. **Lower the face cap:** set `maxTracks = 2` in `TrackPolicy`.
3. **Compress weights:**
   ```bash
   python export_edgetam.py --weights int8        # or palette6
   ```
   Makes the models about 2× smaller. On A17 Pro and later, int8 weights can also speed up Neural Engine execution. Re-run `verify_coreml_mac.py` afterwards.
4. **Batch faces** through one memory-attention call (future work, §8). This is the same idea as SAM 3.1's "multiplexing", which processes all objects in one pass.

---

## 7. Effectiveness: why this tracks faces well

* **Exact SAM 2 semantics.** The on-device tracker is the same algorithm as Meta's reference implementation (IoU 0.9999, §4.6). The paper's accuracy therefore carries over, apart from fp16 effects, which `verify_coreml_mac.py` measures.
* **The detector corrects drift.** SAM trackers occasionally leak onto neighbouring regions. Periodic detector checks re-anchor the track with a fresh box and push it into the second conditioning slot, so the tracker is re-grounded without forgetting the original anchor.
* **Occlusion is modelled, not guessed.** EdgeTAM's object-score head says when the face is gone, and the memory bank brings it back with the same ID once it reappears.
* **Pixel masks, not just boxes.** Boxes come from the mask, so they hug the face instead of the detector's fixed anchor shape. The masks can be used later for blur, relighting or recognition crops.

---

## 8. Limitations and next steps

| Limitation | Next step |
|---|---|
| Numeric fp16 behaviour on the Neural Engine is only simulated here | Run `verify_coreml_mac.py`. If one graph drifts, re-export that graph with `--precision fp32` (it will run on GPU) |
| A face that leaves the frame for more than 1 s comes back with a new ID | Add a recognition embedding (e.g. **SFace**, Apache-2.0, from your comparison doc) per track and re-attach IDs on re-entry |
| Per-face cost is linear | Batch the per-face graphs over faces (batch dimension with enumerated shapes) |
| Fast motion blur | Add a motion prior (SAMURAI-style Kalman scoring of the 3 candidate masks), training-free |
| YuNet misses faces rotated ~90° (e.g. someone lying down), so such faces are only tracked after an upright detection | Optionally run the detector on a 90°-rotated copy of the frame when it finds nothing |

---

## 9. File map

```
FaceTrack/
├── DESIGN.md                      ← this document
├── README.md                      ← build & run
├── docs/reference_pipeline_occlusion.jpg
├── conversion/
│   ├── convert_yunet.py           YuNet ONNX → Core ML with in-graph decode
│   ├── edgetam_coreml_modules.py  the 4 streaming EdgeTAM graphs (§4)
│   ├── export_edgetam.py          → .mlpackage (+ memory presets, int8/palette options)
│   ├── reference_tracker.py       Python twin of EdgeTAM.swift (test oracle)
│   ├── pipeline_reference.py      Python twin of Pipeline + TrackManager (tuning on clips)
│   ├── validate_edgetam.py        streaming export vs Meta's official SAM2VideoPredictor
│   ├── verify_coreml_mac.py       Core ML vs torch on your Mac + per-graph latency
│   ├── ct_compat.py               coremltools 9 / NumPy 2 fix
│   ├── setup.sh, requirements.txt
│   └── face_detection_yunet_2023mar.onnx
└── ios/
    ├── project.yml                XcodeGen spec (iOS 18+)
    └── FaceTrack/
        ├── FaceTrackApp.swift, ContentView.swift   UI, HUD, overlay
        ├── CameraController.swift                  720p BGRA portrait capture
        ├── PixelBufferScaler.swift                 vImage letterbox / squash, pooled buffers
        ├── FaceDetector.swift                      YuNet + NMS
        ├── EdgeTAM.swift                           streaming tracker, memory bank
        ├── TrackManager.swift                      detection-guided policy
        ├── Pipeline.swift                          scheduling, concurrency, stats
        ├── MLArray+Utils.swift
        └── Models/                                 5 converted models + edgetam_config.json
```

## 10. Licenses

| Component | License | Notes |
|---|---|---|
| YuNet model (OpenCV Zoo) | MIT | |
| EdgeTAM code and checkpoint (Meta) | Apache-2.0 | |
| This project's code | Yours | |

Both models are fine for commercial use. Keep their license notices in the app's acknowledgements.

## Sources

* [EdgeTAM paper (arXiv 2501.07256)](https://arxiv.org/abs/2501.07256) · [CVPR 2025 version](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_EdgeTAM_On-Device_Track_Anything_Model_CVPR_2025_paper.pdf) · [EdgeTAM repository](https://github.com/facebookresearch/EdgeTAM)
* [Efficient Track Anything (arXiv 2411.18933)](https://arxiv.org/abs/2411.18933) · [EfficientTAM repository](https://github.com/yformer/EfficientTAM)
* [SAM 2 repository](https://github.com/facebookresearch/sam2) · [SAM 3 / 3.1 (Meta AI blog)](https://ai.meta.com/blog/segment-anything-model-3/)
* [YuNet, OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
* [MediaPipe face detector](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector) · [BlazeFace short-range model card](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf)
* [iPhone 17 technical specifications (Apple)](https://www.apple.com/iphone-17/specs/)
* [faster-edgetam (unofficial fork with 512/768 configs)](https://github.com/korhansevinc/faster-edgetam)
