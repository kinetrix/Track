# FaceHandTrack

Real-time **face tracking** and **hand gesture tracking** using only open-source models with published weights.

| Folder | What it is |
|---|---|
| [`FaceTrack_v1_iOS/`](FaceTrack_v1_iOS) | v1: iPhone app (Swift + Core ML). YuNet finds faces, EdgeTAM (Meta's on-device SAM 2) follows them with pixel masks and stable IDs. About 30 FPS on an iPhone 17. |
| [`FaceTrack_v2_Open/`](FaceTrack_v2_Open) | v2: portable engine (Python + ONNX Runtime) with a Mac viewer. Same face models, plus MediaPipe palm detector + hand landmarks and 15 rule-based gestures. Built so the interface can be swapped later (web, Android, desktop). |

## Models

| Model | Job | Licence |
|---|---|---|
| YuNet (OpenCV Zoo) | find faces | MIT |
| EdgeTAM (Meta) | follow faces: masks + IDs | Apache-2.0 |
| MediaPipe palm detector (Google, via OpenCV Zoo) | find hands | Apache-2.0 |
| MediaPipe hand landmarks (Google, via OpenCV Zoo) | 21 points per hand + Left/Right | Apache-2.0 |

## Quick start (v2 on a Mac)

```bash
cd FaceTrack_v2_Open
./start.sh          # first run sets up .venv and asks for camera access
```

Design decisions, tests and results are in each folder's `README.md` and `DESIGN.md`.
