# FaceTrack v2: design and decisions

v1 (`../FaceTrack_v1_iOS`) ran on the iPhone with Core ML. v2 has two constraints:

1. **Open source with open weights** for every model, including face tracking.
2. **Not tied to iOS.** The app will be re-implemented on a different interface,
   so nothing may depend on Apple-only models, formats or APIs. It runs on the Mac
   for now.

This document lists every decision again, what was considered, and why the
choice was made.

---

## 1. Architecture: engine and interface are separate

```
 camera / video  ──>  INTERFACE (replaceable)            ENGINE (portable)
                      app/mac_viewer.py                  facetrack/
                      - grabs frames                     - Engine.process(frame, t)
                      - shows the window       ──────>     -> FrameResult
                      - draws results          <──────       faces: id, box, mask
                                                             hands: id, Left/Right,
                                                                    21 points, gesture
```

* The **engine** takes a frame (a NumPy image) and a time in seconds and returns
  plain data. It uses only NumPy, a handful of OpenCV image functions (resize,
  colour conversion, rotate/crop) and ONNX Runtime.
* The **interface** owns the camera, the window and the drawing. For the Mac
  it is about 250 lines of OpenCV.
* `FrameResult.to_dict()` turns a result into JSON. A web page or another
  process can therefore use the engine without sharing any code.

### What happens per frame

```
 face path                                        hand path (parallel thread)
 YuNet (every 3rd frame;  ║  EdgeTAM              each tracked hand: landmark model on a
 every frame if no faces) ║  image encoder        crop predicted from its last landmarks
                 ╲        ║        ╱              palm detector for NEW hands only
                  track manager                   (every 5th frame; every frame if none)
                  per face: memory attention      gesture rules -> smoothing
                  + mask decoder + memory encoder
```

* The face path and the hand path run at the same time. ONNX Runtime releases
  Python's lock while a model runs, so the threads really are parallel.
* Inside the face path, YuNet runs at the same time as the EdgeTAM encoder,
  as in v1.
* The interface uses **latest-frame-wins**, also as in v1. The camera thread
  keeps only the newest frame. The engine processes the newest frame when it
  is free, and frames that arrive while it is busy are skipped.

---

## 2. Every choice, reconsidered

### 2.1 Model format: **ONNX** (changed from Core ML)

* ONNX is an open standard (Linux Foundation).
* Every major runtime can load it, and every training framework can export it.
* Core ML `.mlpackage` files only run on Apple devices.

### 2.2 Runtime: **ONNX Runtime** (changed from Core ML)

| Option | Runs on | Why not chosen |
|---|---|---|
| **ONNX Runtime** (chosen) | macOS, Windows, Linux, iOS, Android, web (onnxruntime-web) | - |
| Core ML | Apple only | breaks constraint 2 |
| LiteRT / TFLite | mobile, desktop CPU | the PyTorch models would need an extra conversion, and desktop GPU support is limited |
| ExecuTorch | mobile | younger, with fewer desktop accelerators, and no browser target |
| OpenCV DNN | everywhere OpenCV runs | slower for transformer graphs like EdgeTAM, with fewer accelerators |

ONNX Runtime uses the same `.onnx` files everywhere and only switches its
*execution provider* (the accelerator backend):

| Platform | Execution provider |
|---|---|
| macOS / iOS | Core ML (Neural Engine / GPU) |
| Windows | DirectML |
| NVIDIA GPUs | CUDA |
| Android | NNAPI / QNN |
| Anywhere | CPU |

Using the Core ML backend on the Mac is still allowed: the models stay open
ONNX files, and Core ML is only one interchangeable backend. `runtime.py`, with
`accel="auto"`, does the following:
1. Times each available backend on each model once.
2. Keeps the fastest per model, because a backend can win on one model and lose
   on another.
3. Caches the choice.

### 2.3 Face detector: **YuNet** (kept)

| Option | Licence | Verdict |
|---|---|---|
| **YuNet** (OpenCV Zoo 2023mar) | MIT | **kept**: 53k parameters, sub-millisecond to a few ms, 5 face landmarks, already validated in v1 |
| BlazeFace (MediaPipe) | Apache-2.0 | open, but its short-range model is designed for faces close to the camera |
| SCRFD / RetinaFace (InsightFace) | code MIT, **weights non-commercial research only** | breaks constraint 1 for anything beyond research |
| RetinaFace-MobileNet (community ports) | MIT | open, but larger and slower than YuNet for no gain at webcam distances |

**Change:** the original YuNet ONNX is fixed at 640×640. `export/yunet_modules.py`
re-exports it at **640×384**, to fit a landscape 16:9 webcam with little
padding. The box decoding is now inside the graph, as in v1. The weights are
unchanged: `validate_onnx.py` shows a maximum difference of 4e-4 px.

### 2.4 Face tracker: **EdgeTAM** (kept)

The tracker must be SAM-based (v1 requirement). Candidates with open weights:

| Option | Licence | Size | Verdict |
|---|---|---|---|
| **EdgeTAM** (Meta, CVPR 2025) | Apache-2.0 | 13.9M params | **kept**: the smallest SAM 2-family video tracker, built for on-device use |
| SAM 2.1 tiny (Meta) | Apache-2.0 | 38.9M params | about 3× bigger, for a small accuracy gain on general objects |
| EfficientTAM (Meta) | Apache-2.0 | ViT-based | lighter than SAM 2, but still heavier than EdgeTAM |

The four-graph streaming split from v1 is runtime-neutral and is kept unchanged
(`export/edgetam_modules.py`):

| Graph | Runs |
|---|---|
| image encoder | once per frame |
| memory attention | per face |
| mask decoder | per face |
| memory encoder | per face |

These v1 decisions are also kept:

| Decision | Value |
|---|---|
| Input size | 512 |
| Memory per face | 2 anchors + 6 recent frames + 16 pointers |
| Empty memory slots | fixed size, masked out |
| Layer-0 attention | computed once in the encoder for all faces |
| Policy | time-based track retirement, all thresholds unchanged |

**Change:** the export target is ONNX (opset 17) instead of Core ML. The decoder
is one graph with a variable number of prompt points (2 = box, 1 = tracking),
replacing v1's two-function Core ML package.

### 2.5 Hand detection and 21 points: **MediaPipe palm detector + hand landmarks** (new)

| Option | Licence | Size | Verdict |
|---|---|---|---|
| **MediaPipe palm + landmarks** (OpenCV Zoo ONNX, 2023feb) | Apache-2.0 | 0.96M + 1.01M weights, 3.9 + 4.1 MB | **chosen** |
| RTMDet-nano + RTMPose-m hand (OpenMMLab) | Apache-2.0 | about 1M + 13.6M | more accurate on hard poses, but about 7× heavier; no Left/Right. Fallback if accuracy is not enough |
| HaGRID YOLOv10n gesture detector | dataset CC BY-SA variant, YOLOv10 code AGPL-3.0 | about 2.3M | boxes only, so no pinch or pointing; restrictive licences |
| ai-forever dynamic_gestures | modified Apache-2.0 | - | boxes only; useful as a reference for swipe and zoom logic |
| Ultralytics YOLO hand-pose, OpenPose hand | AGPL / non-commercial | - | licence |
| Apple Vision hand pose | closed, iOS/macOS only | - | breaks both constraints |

Why MediaPipe:
* It is the lightest option: about 2M weights for both models together.
* It gives 21 points per hand, which pinch, pointing and finger counting need.
* It says which hand is left and which is right.
* It is Apache-2.0 and already published as ONNX.

**Hand tracking** works the way MediaPipe does it, and the same way the face
path detects only every 3rd frame:
* A tracked hand's next crop is built from its own landmarks of the previous
  frame (`palm_from_landmarks`).
* The palm detector therefore runs only to find **new** hands: every 5th frame,
  or every frame while no hand is tracked.
* A hand is dropped when the landmark model's presence score falls below 0.6.
* A new hand needs a score of 0.8.

The pre- and post-processing is ported from OpenCV Zoo. On all 24 test photos it
gives the same palms and landmarks as OpenCV Zoo's own code, to within
0.02 px.

**Left/Right:** this ONNX model reports the hand correctly for an ordinary,
un-mirrored camera image. The viewer mirrors the picture like a selfie view, and
a mirror swaps left and right, so the engine flips the model's answer back
(`mirrored_input=True`, the default). For a non-mirrored source pass
`mirrored_input=False`. First version had this backwards; it was spotted live on
the Mac (your right hand showed as "Left") and is now checked by
`tests/test_handedness.py`: the model's answer must swap when the picture is
mirrored, and the engine must report the same real hand either way.

### 2.6 Gestures: **rules on the 21 points** (new)

* Rules first, because they need no training data, have no cost and are easy
  to explain.
* A learned classifier comes later, only for gestures the rules can't
  separate.
* Each finger is classified as extended or folded using the model's **3-D world
  landmarks**. These don't change with hand size, distance or rotation.
  * A finger is extended when its tip is at least 15% farther from the wrist
    than its middle joint, and the finger bends less than 80°.
  * The thumb counts as out when its tip is more than 0.85 palm lengths from
    the ring-finger base and it is fairly straight.
* Pinch and pointing direction use the **image** points, because they are about
  what the camera sees.
* A label is shown once it wins 3 of the last 5 frames, so it doesn't flicker.

| Gesture | Rule (T = thumb, I M R P = fingers) |
|---|---|
| Open palm | I M R P extended |
| Fist | nothing extended |
| Thumbs up / down | only T, tip above / below its base |
| Pointing up/down/left/right | only I (thumb any), direction of the index finger |
| Victory | I M, thumb folded |
| Three | I M R, thumb folded |
| Rock | I P, thumb folded |
| I love you | T I P, index clearly straight |
| Call me | T P (a half-bent index allowed) |
| OK | thumb tip touches index tip, M R P extended |
| Pinch | thumb tip touches index tip, index only partly bent |

**Limitation:** the thresholds were tuned on the same 24 photos they are tested
on (17/17 with a rule), so live use is the real test. The photos are from
HaGRID (CC BY-SA variant) and are used only as test data.

### 2.7 Interface for now: **OpenCV window** (new)

* OpenCV is already a dependency of the engine.
* Its window works on macOS, Windows and Linux.
* It is deliberately minimal, because it will be replaced.

---

## 3. v1 → v2 at a glance

| Part | v1 (iOS) | v2 |
|---|---|---|
| Face models | YuNet + EdgeTAM | same models and weights |
| Format / runtime | Core ML | ONNX / ONNX Runtime |
| EdgeTAM graphs, memory layout, 512 input | yes | carried over unchanged |
| Track policy and thresholds | `TrackManager.swift` | `faces.py`, same values |
| Camera, threading, UI | Swift, AVFoundation | Python, OpenCV (thin, replaceable) |
| Hands + gestures | - | MediaPipe palm + landmarks (ONNX), rules |
| Backend choice | fixed (GPU / Neural Engine) | measured per model (`accel="auto"`) |

---

## 4. Validation

| Check | Result |
|---|---|
| Each exported ONNX graph vs its PyTorch module (`export/validate_onnx.py`) | max difference ≤ 1e-5 for EdgeTAM outputs; 4e-4 px for YuNet boxes |
| v2 face path vs v1 reference pipeline (`tests/test_faces_vs_v1.py`), 3 clips, 136 frames | **identical**: same track IDs on every frame, mask IoU 1.0000 |
| Hand pre/post-processing vs OpenCV Zoo's code, 24 photos | same palms; landmarks within 0.02 px |
| Hand detection on 24 photos | 24/24 found |
| Gesture rules (`tests/test_gestures.py`) | 17/17, also via the tracking path |
| End to end (`tests/test_engine.py`): face clip with a hand pasted in that moves, rotates and changes pose | one hand ID throughout; "Open palm" then "Victory" after the change; face tracked |

---

## 5. Speed

All times are per frame, with one face and one hand.

| Machine | Total | Notes |
|---|---|---|
| Cloud test machine, 2 x86 cores, CPU | about 400 ms | not representative: everything shares 2 slow cores |
| Linux VM on the MacBook Pro, 4 Apple Silicon cores, CPU only | 131 ms | encoder 62.5, memory attention 43.7, decoder 7.5, memory encoder 17.0, YuNet 22.1, palm 27.3, landmarks 20.0 (hand stages run in parallel and slow down because they share 4 cores) |
| MacBook Pro, native macOS, all cores, CPU or Core ML backend | *to be measured on first run* | the viewer prints per-stage times |

Per-model CPU times on the 2-core test machine:

| Model | Time |
|---|---|
| encoder | 96 ms |
| memory attention | 123 ms |
| decoder | 14 ms |
| memory encoder | 38 ms |
| YuNet | 5 ms |
| palm detector | 11 ms |
| landmarks | 3 ms |

Most of the cost is EdgeTAM's encoder and memory attention, as on the iPhone.

If the Mac is slower than wanted, the levers are, in order:
1. The Core ML backend, which `auto` picks when it is faster.
2. The smaller memory preset (2 anchors + 3 recent). On the iPhone this
   preset lost no accuracy; on CPU it roughly halves memory attention.
3. Fewer CPU threads per model, so the face and hand paths don't compete.

---

## 6. Moving to another interface later

Everything model-related already moves as is: the `.onnx` files,
`engine_config.json` and the result format.

| Target | How |
|---|---|
| Desktop app in another toolkit, or a local web page | keep the Python engine; the new UI calls `Engine.process`, or receives `FrameResult.to_dict()` JSON over a local socket |
| Browser only | onnxruntime-web (WebGPU/WASM) with the same ONNX files; port `facetrack/` to JavaScript |
| Android | ONNX Runtime Android; port `facetrack/` to Kotlin |
| iOS again | ONNX Runtime iOS with the Core ML backend; or keep v1's Swift code, whose logic is the same |

The engine is about 1,000 lines of straightforward code (`faces.py`, `hands.py`,
`gestures.py`, `engine.py`). It uses only array maths and five image functions:
* resize
* colour conversion
* pad
* rotate
* crop

These exist in OpenCV for every platform (including opencv.js) or take a few
lines to write. Each file says which v1 or OpenCV Zoo code it mirrors.

---

## 7. Models shipped

| File | What | Weights | Size | Licence |
|---|---|---|---|---|
| `yunet_640x384.onnx` | YuNet, decoding in-graph | 0.053M | 0.4 MB | MIT |
| `face_detection_yunet_2023mar.onnx` | original YuNet (source for the re-export, reference) | 0.053M | 0.2 MB | MIT |
| `edgetam_image_encoder.onnx` | EdgeTAM encoder + layer-0 attention | 13.9M params for EdgeTAM in total; the files also hold baked-in position tables | 24.0 MB | Apache-2.0 |
| `edgetam_memory_attention.onnx` | memory attention (fixed 24 slots) | (included above) | 22.5 MB | Apache-2.0 |
| `edgetam_mask_decoder.onnx` | prompt encoder + mask decoder + pointer head | (included above) | 19.9 MB | Apache-2.0 |
| `edgetam_memory_encoder.onnx` | memory encoder + spatial perceiver | (included above) | 6.9 MB | Apache-2.0 |
| `palm_detection_mediapipe_2023feb.onnx` | palm detector, 192×192 | 0.96M | 3.9 MB | Apache-2.0 |
| `handpose_estimation_mediapipe_2023feb.onnx` | 21 hand landmarks + Left/Right + presence, 224×224 | 1.01M | 4.1 MB | Apache-2.0 |

All models are fp32 ONNX, about 82 MB in total. Licence texts are in
`models/LICENSES/`.

---

## 8. Next steps (one at a time)

1. Moving gestures from landmark paths over about 0.5 s: swipe left/right/up/down,
   wave, pinch-and-drag, two-hand zoom.
2. A small learned classifier (PyTorch → ONNX) on the 21 points, for gestures the
   rules can't separate.
3. Associate each hand with the nearest face ID ("face #2: thumbs up").
