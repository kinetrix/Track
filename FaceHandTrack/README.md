# FaceHandTrack

Face detection + tracking and hand gesture tracking from a camera (Intel RealSense
D455f colour stream, or any webcam), built only on open-source models with open
weights. Three versions of the same project, one per subfolder:

| Folder | Runs on | Language / runtime | Accelerator | Window |
|---|---|---|---|---|
| [`mac_native/`](mac_native/README.md) | macOS 15+ (Apple silicon) | Swift + Core ML | Neural Engine + GPU | native Mac app |
| [`ubuntu_native/`](ubuntu_native/README.md) | Ubuntu 22.04 / 24.04 (x86_64 PC or laptop) | Python + PyTorch | NVIDIA GPU (CUDA), else CPU | desktop window |
| [`onnx/`](onnx/README.md) | macOS, Linux, Windows (also headless / robot) | Python + ONNX Runtime | Core ML / CUDA / DirectML / CPU, picked per model | window, or browser stream when headless |

All three use the same models, the same tracking rules and the same 15 gestures:

| Job | Model | Licence |
|---|---|---|
| Find faces | YuNet (OpenCV Zoo) | MIT |
| Track faces (pixel masks, stable IDs) | EdgeTAM (Meta), streaming SAM 2 tracker | Apache-2.0 |
| Find hands | MediaPipe palm detector (Google, via OpenCV Zoo) | Apache-2.0 |
| 21 points per hand | MediaPipe hand landmarks (Google, via OpenCV Zoo) | Apache-2.0 |
| Gestures | hand-written rules on the 21 points | - |

Gestures: Open palm, Fist, Thumbs up, Thumbs down, Pointing up/down/left/right,
Victory, Three, Rock, I love you, Call me, OK, Pinch. Each hand also shows Left/Right
(the person's real hand, mirrored view or not) and the number of extended fingers.

## Which one to use

- **On the Mac:** `mac_native` (fastest; double-click `Build and Run.command`).
  `onnx` also runs here (`Start FaceTrack.command`).
- **On the Ubuntu laptop:** `ubuntu_native` (`./setup.sh`, then `./run.sh`).
- **Anywhere else, or as the portable reference** (other OS, a robot computer, no
  screen): `onnx`.

## Camera

All three pick the RealSense colour sensor automatically when it is plugged in
(USB 3), and fall back to the computer's own camera. Only the colour stream is
used; depth is not needed. On Ubuntu the app reads it with Intel's `pyrealsense2`
SDK, or the camera's plain V4L2 device without it. On macOS 27 the system lists only
the D455f's depth sensor, so `mac_native` reads the colour stream through Intel's
librealsense (`Run with RealSense.command`, details in `mac_native/README.md`); the
`onnx` viewer on the Mac falls back to the MacBook camera.

## Shared design

```
camera frame ─┬─ face path:  YuNet (every 3rd frame) ─┐
              │              EdgeTAM image encoder ────┴─ per face: memory attention → mask decoder → memory encoder
              └─ hand path:  palm detector (new hands, every 5th frame)
                             hand landmarks (from last frame's landmarks) → gesture rules → 3-of-5 smoothing
```

The two paths run in parallel, and frames that arrive while one is being
processed are skipped (latest frame wins), so delay never builds up. The full
reasoning behind every choice is in [`onnx/DESIGN.md`](onnx/DESIGN.md); the
original iPhone face tracker is in `../FaceTrack_v1_iOS`.
