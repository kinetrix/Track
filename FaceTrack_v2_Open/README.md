# FaceTrack v2 (open models, any interface)

Face detection + tracking and hand gesture tracking from a webcam, built only on
open-source models with open weights, all in the open ONNX format:

| Job | Model | Licence |
|---|---|---|
| Find faces | YuNet (OpenCV Zoo) | MIT |
| Track faces (pixel masks, stable IDs) | EdgeTAM (Meta), streaming SAM 2 tracker | Apache-2.0 |
| Find hands | MediaPipe palm detector (Google, via OpenCV Zoo) | Apache-2.0 |
| 21 points per hand | MediaPipe hand landmarks (Google, via OpenCV Zoo) | Apache-2.0 |
| Gestures | hand-written rules on the 21 points (no model) | - |

The engine (`facetrack/`) is plain Python + NumPy + ONNX Runtime and knows
nothing about cameras or windows. The Mac viewer (`app/mac_viewer.py`) is a thin
layer on top, so a different interface can replace it later. Why every choice
was made is in **[DESIGN.md](DESIGN.md)**.

## Run on the Mac

Double-click **`Start FaceTrack.command`** in Finder, or in Terminal:

```bash
cd "FaceTrack_v2_Open"
./start.sh
```

The first run:
1. sets up a private Python environment in `.venv` (about a minute; if macOS only
   has its built-in Python 3.9, a private Python 3.12 is downloaded into `.tools/`);
2. asks for **camera access for Terminal**: click Allow. If you missed it, enable it in
   System Settings > Privacy & Security > Camera, then run again;
3. measures which backend (CPU or Core ML) is fastest for each model and remembers
   the choice in `~/.cache/facetrack/`, so later launches start straight away.

In the window:

| Key | Action |
|---|---|
| `q` | quit |
| `m` | face masks on/off |
| `r` | reset all tracks |
| `f` / `h` | faces / hands on/off |
| `s` | save a screenshot |

The top-left shows processed FPS, the camera FPS, and milliseconds per stage.
Terminal prints the same every 2 seconds, plus track events ("face new #1", "hand drop #2").

Options: `./start.sh --accel cpu` (force CPU), `--no-hands`, `--no-faces`,
`--no-mirror`, `--video file.mp4`, `--camera 1`.

## Gestures

Open palm, Fist, Thumbs up, Thumbs down, Pointing up/down/left/right, Victory,
Three, Rock, I love you, Call me, OK, Pinch. Each hand also shows Left/Right and
the number of extended fingers. A label appears once it wins 3 of the last 5 frames.

## Folder layout

```
facetrack/        the engine (platform-neutral)
  engine.py         Engine.process(frame, t) -> FrameResult
  faces.py          YuNet detector, EdgeTAM streaming tracker, face track manager
  hands.py          palm detector, hand landmarks, hand tracker
  gestures.py       gesture rules + smoothing
  runtime.py        ONNX Runtime sessions, picks the fastest backend per model
  types.py          result structures (Face, Hand, FrameResult; to_dict() for JSON)
app/mac_viewer.py   camera + window (the only platform-specific code)
models/             all ONNX models + engine_config.json + LICENSES/
export/             how the ONNX files were made (PyTorch -> ONNX) and validated
tests/              checks: gestures on labelled photos, v2 vs v1 face tracking, end-to-end
```

## Tests

```bash
.venv/bin/python tests/test_gestures.py                         # 17/17 labelled photos
.venv/bin/python tests/test_engine.py --clip ../FaceTrack_v1_iOS/conversion/eval/clip_selfie
# needs the export environment (PyTorch), compares with the v1 reference:
export/.venv/bin/python tests/test_faces_vs_v1.py --clip ../FaceTrack_v1_iOS/conversion/eval/clip_group
```

## Re-exporting the models (optional)

```bash
cd export
./setup.sh                                   # PyTorch env + EdgeTAM checkout + OpenCV Zoo models
.venv/bin/python export_onnx.py              # -> ../models (EdgeTAM at 512, YuNet at 640x384)
.venv/bin/python validate_onnx.py            # ONNX vs PyTorch, max difference per output
```
