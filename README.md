# FaceHandTrack

Real-time **face tracking** and **hand gesture tracking** using only open-source models with published weights.

| Folder | What it is |
|---|---|
| [`FaceHandTrack/`](FaceHandTrack) | Current project, in three versions: [`mac_native/`](FaceHandTrack/mac_native) (Swift + Core ML app for Apple silicon Macs), [`ubuntu_native/`](FaceHandTrack/ubuntu_native) (Python + PyTorch, CUDA or CPU, for Ubuntu) and [`onnx/`](FaceHandTrack/onnx) (Python + ONNX Runtime, portable reference for any OS or headless). Reads the Intel RealSense D455f colour stream or any webcam. |
| [`FaceTrack_v1_iOS/`](FaceTrack_v1_iOS) | v1: iPhone app (Swift + Core ML). YuNet finds faces, EdgeTAM (Meta's on-device SAM 2) follows them with pixel masks and stable IDs. About 30 FPS on an iPhone 17. |

## Models

| Model | Job | Licence |
|---|---|---|
| YuNet (OpenCV Zoo) | find faces | MIT |
| EdgeTAM (Meta) | follow faces: masks + IDs | Apache-2.0 |
| MediaPipe palm detector (Google, via OpenCV Zoo) | find hands | Apache-2.0 |
| MediaPipe hand landmarks (Google, via OpenCV Zoo) | 21 points per hand + Left/Right | Apache-2.0 |

Gestures (15, rule-based on the 21 points): Open palm, Fist, Thumbs up/down, Pointing up/down/left/right, Victory, Three, Rock, I love you, Call me, OK, Pinch.

## Quick start

- **Mac:** `FaceHandTrack/mac_native/` → double-click `Build and Run.command` (or `Run with RealSense.command` for the RealSense colour camera).
- **Ubuntu:** `cd FaceHandTrack/ubuntu_native && ./setup.sh && ./run.sh`
- **Anywhere else:** `cd FaceHandTrack/onnx && ./start.sh` (first run sets up its environment)

Which version to use, camera notes and the shared design are in [`FaceHandTrack/README.md`](FaceHandTrack/README.md); each version's folder has its own `README.md`, and the design reasoning is in [`FaceHandTrack/onnx/DESIGN.md`](FaceHandTrack/onnx/DESIGN.md).
