# FaceHandTrack: Ubuntu-native version (PyTorch, CUDA or CPU)

Face detection + tracking and hand gesture tracking from an Intel RealSense
camera (colour stream) or the laptop webcam, on a native Ubuntu install (for
example the Ubuntu side of a dual-boot Windows laptop). It shows its own window.

Everything runs in **PyTorch**, the native deep-learning stack on Linux: on the
NVIDIA GPU through CUDA when the laptop has one, otherwise on the CPU. Same
models, same tracking rules and same gestures as the other two versions:

| Job | Model | How it runs here |
|---|---|---|
| Find faces | YuNet (OpenCV Zoo, MIT) | published weights loaded into PyTorch |
| Track faces (pixel masks, stable IDs) | EdgeTAM (Meta, Apache-2.0) | Meta's own PyTorch code + checkpoint |
| Find hands | MediaPipe palm detector (Google, Apache-2.0) | published weights loaded into PyTorch |
| 21 points per hand | MediaPipe hand landmarks (Google, Apache-2.0) | published weights loaded into PyTorch |
| Gestures | hand-written rules on the 21 points | Python |

How it differs from `../onnx`: no ONNX Runtime; EdgeTAM runs from its original
checkpoint (not from exported graphs), the camera is read with Intel's RealSense
SDK, and the window is a normal desktop window. On the same frames both give
the same face tracks and hand landmarks (checked on the v1 test clips).

## Setup (once)

Copy the whole `FaceHandTrack` folder to the Ubuntu laptop (USB stick, or
`git clone` of the GitHub repo), then in a terminal:

```bash
cd FaceHandTrack/ubuntu_native
./setup.sh
```

It asks for your password once (for `apt` and the camera rules) and then:
1. installs `python3-venv`, `git`, `curl` if missing;
2. creates a private Python environment in `.venv`;
3. installs PyTorch: the CUDA build if an NVIDIA driver is working (`nvidia-smi`),
   otherwise the smaller CPU-only build (`./setup.sh --cpu` forces CPU);
4. installs OpenCV, ONNX loader, Intel's `pyrealsense2`, and EdgeTAM's few dependencies;
5. downloads EdgeTAM's code and checkpoint into `third_party/EdgeTAM`;
6. installs the RealSense udev rules (lets the app open the camera without `sudo`;
   `./setup.sh --no-udev` skips this). Unplug and replug the camera afterwards.

The last lines say whether PyTorch can use the GPU and which RealSense cameras it sees.

**NVIDIA GPU but "CUDA available: False"?** The NVIDIA driver is missing or too old:
`sudo ubuntu-drivers install`, reboot, run `./setup.sh` again. On a dual-boot laptop
with Secure Boot on, Ubuntu asks to enrol a key (MOK) during the driver install;
without that the driver does not load.

## Run

```bash
./run.sh
```

With a RealSense plugged in (USB 3 port) it uses its colour stream; otherwise the
laptop webcam (`/dev/video0`). The first launch takes a few seconds to load the models.

| Key | Action |
|---|---|
| `q` | quit |
| `m` | face masks on/off |
| `r` | reset all tracks |
| `f` / `h` | faces / hands on/off |
| `s` | save a screenshot |

The top-left shows processed FPS, camera FPS, milliseconds per stage, and the
device (`cuda (NVIDIA ...)` or `cpu`). The terminal prints the same every 2 s plus
track events ("face new #1", "hand drop #2").

Options:

```bash
./run.sh --list-cameras           # what the laptop sees (V4L2 devices + RealSense serial numbers)
./run.sh --camera realsense:1     # second RealSense camera (or realsense:<serial number>)
./run.sh --camera 0               # /dev/video0 instead of the RealSense
./run.sh --no-mirror              # camera facing away from you (e.g. on a robot)
./run.sh --accel cpu              # force CPU
./run.sh --video clip.mp4         # a video file or a folder of .jpg frames
./run.sh --width 848 --height 480 # smaller camera mode
```

`--no-mirror` only changes the picture; Left/Right always name the person's real hand.

**Camera notes.** The app uses Intel's SDK (`pyrealsense2`) for the RealSense and
falls back to the camera's plain V4L2 webcam device if the SDK can't open it (for
example without the udev rules). If neither works: check `./run.sh --list-cameras`,
use a USB 3 port and cable, and replug the camera. Only the colour stream is used.

## Checks

```bash
.venv/bin/python tests/test_gestures.py       # 17/17 labelled hand photos
.venv/bin/python tests/test_handedness.py     # Left/Right consistent with a mirrored view
.venv/bin/python tests/test_engine.py --clip ../../FaceTrack_v1_iOS/conversion/eval/clip_selfie
```

`test_engine.py` (any folder of `.jpg` frames with a face works as `--clip`) pastes a
moving hand into the clip, checks the hand keeps one ID and the right gestures, writes
`tests/engine_check.mp4`, and prints the engine time per frame on this machine.

## Folder layout

```
facehand/            the engine (no camera / window code)
  engine.py            Engine.process(frame, t) -> FrameResult
  faces.py             YuNet detector, EdgeTAM streaming tracker (PyTorch), face track manager
  hands.py             palm detector, hand landmarks, hand tracker
  gestures.py          gesture rules + smoothing
  edgetam_modules.py   EdgeTAM split into 4 streaming stages (wraps Meta's code)
  weights.py           loads the published YuNet / MediaPipe weights as PyTorch modules
  backend.py           CUDA / CPU choice and timing
  types.py             result structures (Face, Hand, FrameResult; to_dict() for JSON)
app/viewer.py        camera + window
app/realsense.py     RealSense colour stream (pyrealsense2, or plain V4L2)
models/              YuNet + MediaPipe weight files, engine_config.json, LICENSES/
third_party/EdgeTAM/ EdgeTAM code + checkpoint (downloaded by setup.sh)
tests/               gesture, handedness and end-to-end checks
```
