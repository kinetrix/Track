# FaceHandTrack: ONNX version (open models, any OS)

Face detection + tracking and hand gesture tracking from a camera (Intel RealSense
colour stream or any webcam), built only on open-source models with open weights,
all in the open ONNX format. This is the portable version of the project: the same
code runs on macOS, Linux and Windows with ONNX Runtime (sibling folders
`../mac_native` and `../ubuntu_native` are the platform-native versions).

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
cd "/Users/vignesh/Desktop/MSCS/Sem 3/AP2/Face Recognition/FaceHandTrack/onnx"
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

**Camera.** With an Intel RealSense (D455f etc.) plugged in, its colour sensor is used
automatically if macOS lists it (as "... RGB"); otherwise the MacBook camera (see
"RealSense on macOS" below). Terminal prints every camera it found
with its number, e.g. `found: 0: iPhone Camera, 1: Intel(R) RealSense(TM) Depth Camera 455 RGB, ...`.
Pick one yourself with `--camera realsense`, `--camera builtin`, `--camera 1` (a number from
that list) or part of a name (`--camera iPhone`). `c` in the window switches to the next camera.
Use a USB 3 port/cable for the RealSense (USB 2 limits the colour stream).

Options: `./start.sh --accel cpu` (force CPU), `--no-hands`, `--no-faces`,
`--no-mirror`, `--video file.mp4`, `--camera realsense`, `--width 1280 --height 720`.

**RealSense on macOS.** On this Mac (macOS 27, D455f) macOS lists only the camera's
depth sensor ("Intel(R) RealSense(TM) Depth Camera 455f Depth"), which sends no
picture, so the viewer skips it. Quit `realsense-viewer` first (it holds the camera
while it runs) and use a USB 3 port. Without an "RGB" entry, the colour stream needs
Intel's SDK (librealsense), which on macOS must run as administrator (`sudo`). On Linux
(`../ubuntu_native`, or `--source realsense` below) the colour stream works directly.

## Run on Linux (OrbStack on the Mac, or any Linux PC / robot)

A Linux machine has no window for the app and cannot use the MacBook camera
directly. So on Linux the engine runs **headless**: frames come in over the
network (or from a video file) and the annotated picture is served to a browser.

**Once:** in the Mac's Terminal create an Ubuntu machine and open its shell:

```bash
orb create ubuntu facetrack
orb -m facetrack
```

Then, inside that Linux shell (your Mac folders appear at the same path):

```bash
cd "/Users/vignesh/Desktop/MSCS/Sem 3/AP2/Face Recognition/FaceHandTrack/onnx"
./setup_linux.sh
```

**Check it works (no camera needed):**

```bash
.venv-linux/bin/python tests/test_gestures.py
.venv-linux/bin/python tests/test_handedness.py
.venv-linux/bin/python app/headless_viewer.py --source ../../FaceTrack_v1_iOS/conversion/eval/clip_selfie --save clip_out.mp4 --port 0
open clip_out.mp4
```

This one shows no window while it runs: it writes the annotated video `clip_out.mp4`
into this folder, and `open` plays it on the Mac.

**Live, with the Mac camera:**

1. Mac Terminal (window 1), share the camera:
   ```bash
   cd "/Users/vignesh/Desktop/MSCS/Sem 3/AP2/Face Recognition/FaceHandTrack/onnx"
   .venv/bin/python app/camera_server.py
   ```
2. Linux shell (window 2), run the engine on it:
   ```bash
   .venv-linux/bin/python app/headless_viewer.py --source http://host.orb.internal:8080/stream.mjpg --mirror
   ```
3. The Mac's browser opens on **http://localhost:8090** by itself and shows the live picture
   (open that address yourself if it doesn't). There is no app window on Linux: the browser is the screen.

`--mirror` gives a selfie view (Left/Right stay correct either way). Leave it off
for a robot's outward-facing camera.

**Linux PC / robot with the RealSense plugged in** (a real Linux machine; an OrbStack
machine cannot see USB cameras): `--source realsense` reads the colour stream directly
(`realsense:1` = second camera, or `realsense:<serial number>`). It uses Intel's
`pyrealsense2` if installed (`.venv-linux/bin/pip install pyrealsense2`, plus the udev rules
described in `../ubuntu_native/README.md`), otherwise the camera's plain V4L2 webcam device. `--json results.jsonl` writes one line per
frame (faces, hands, gestures) for other programs. Linux uses the CPU (no Core ML),
so expect lower FPS than the Mac app; on a robot with an NVIDIA GPU use `--accel cuda`.
Stop either side with Ctrl+C.

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
app/headless_viewer.py  Linux / robot: stream or file in, browser stream + JSON out
app/camera_server.py    shares the Mac camera over the network (for Linux machines)
app/realsense.py    RealSense colour stream on Linux (pyrealsense2, or plain V4L2)
app/mjpeg.py        tiny web-stream server used by both
models/             all ONNX models + engine_config.json + LICENSES/
export/             how the ONNX files were made (PyTorch -> ONNX) and validated
tests/              checks: gestures on labelled photos, v2 vs v1 face tracking, end-to-end
```

## Tests

```bash
.venv/bin/python tests/test_gestures.py                         # 17/17 labelled photos
.venv/bin/python tests/test_engine.py --clip ../../FaceTrack_v1_iOS/conversion/eval/clip_selfie
# needs the export environment (PyTorch), compares with the v1 reference:
export/.venv/bin/python tests/test_faces_vs_v1.py --clip ../../FaceTrack_v1_iOS/conversion/eval/clip_group
```

## Re-exporting the models (optional)

```bash
cd export
./setup.sh                                   # PyTorch env + EdgeTAM checkout + OpenCV Zoo models
.venv/bin/python export_onnx.py              # -> ../models (EdgeTAM at 512, YuNet at 640x384)
.venv/bin/python validate_onnx.py            # ONNX vs PyTorch, max difference per output
```
