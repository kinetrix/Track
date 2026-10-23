# FaceHandTrack: Mac-native version (Swift + Core ML)

A native macOS app: face detection + tracking and hand gesture tracking from an
Intel RealSense camera's colour stream (or the MacBook camera), with every model
running through Apple's Core ML on the Neural Engine / GPU. No Python at run time.

| Job | Model | Runs on |
|---|---|---|
| Find faces | YuNet (OpenCV Zoo, MIT), 640x384 | GPU |
| Track faces (pixel masks, stable IDs) | EdgeTAM (Meta, Apache-2.0), 512x512, streaming memory | Neural Engine |
| Find hands | MediaPipe palm detector (Google, Apache-2.0) | GPU |
| 21 points per hand | MediaPipe hand landmarks (Google, Apache-2.0) | GPU |
| Gestures | hand-written rules on the 21 points | CPU (Swift) |

Same models, tracking rules and gestures as `../onnx` and `../ubuntu_native`; the
face tracker is the one from the iOS app (`../../FaceTrack_v1_iOS`), and the hand
tracker and gesture rules are line-by-line Swift ports of the Python engine.

## Build and run

Needs Xcode 16 or newer (already on this Mac) and macOS 15 or newer.

Double-click **`Build and Run.command`** in Finder. It builds the app with Xcode's
command-line tools (the first build also compiles the Core ML models, about a
minute), leaves **`FaceHandTrack.app`** in this folder, and starts it. macOS asks
once for camera access: click Allow. Next time just double-click `FaceHandTrack.app`.

Or open `FaceHandTrack.xcodeproj` in Xcode and press Run (⌘R).

In the window:

| Control | Action |
|---|---|
| Camera | pick the camera; a RealSense colour sensor is chosen first when plugged in |
| Mirror | selfie view (Left/Right always name the person's real hand) |
| Masks / Faces / Hands | show face masks; turn the face or hand path off |
| Reset | forget all tracks |

The top-left shows FPS, faces and hands, and milliseconds per stage. Each hand is
labelled `Left #1: Victory (2)`: real hand, track ID, gesture, extended fingers.

**RealSense colour camera.** macOS shows the D455f to apps only as its depth sensor
("Intel(R) RealSense(TM) Depth Camera 455f Depth", infrared-only formats), so the app reads
the colour (2-D) stream through Intel's SDK (librealsense, from Homebrew: `brew install
librealsense`, already installed here) instead. macOS lets the SDK take the camera over USB
only as administrator, so:

1. Quit `realsense-viewer` (it holds the camera while it runs).
2. Plug the RealSense into a USB 3 port.
3. Double-click **`Run with RealSense.command`**. It builds the app, then asks for your Mac
   password (Terminal shows nothing while you type) and starts it on the colour stream
   (1280x720 @ 30 fps; smaller modes on USB 2).

Quit the app (⌘Q) to stop; the camera is then handed back to macOS. Without the RealSense,
`Build and Run.command` / `FaceHandTrack.app` use the MacBook camera as before.

## Speed (MacBook Pro M5, macOS 27)

Live with the MacBook camera: 30 FPS (the camera's own rate), about 18 ms per frame
with one face and the hand path running. Per model on the Neural Engine: face
detector 0.6 ms, EdgeTAM encoder 2.6 ms, per face 7-8 ms (memory attention 5.5,
decoder 0.9, memory encoder 1.4), palm detector 0.5 ms, hand landmarks 0.3 ms.

## Checks

`tests/Run Self-Test.command` builds the app and runs its command-line checks
without a camera: the hand models and gesture rules on 24 labelled hand photos, the
whole pipeline on a clip with a moving hand and on a group clip, and a speed test
of every model on CPU / GPU / Neural Engine (`logs/selftest.txt`, `logs/benchmark.json`).
It uses the ONNX version's Python environment (`../onnx/.venv`, made by its
`setup.sh`) to create the test images from the v1 test clips and to compare every
result with the Python engine (`logs/compare.txt`). Last run: same gesture and same
Left/Right on 24/24 photos; on the moving-hand clip the same face IDs and hand gestures
on 40/40 frames. Face masks differ only slightly (the Core ML models run in 16-bit).

## Folder layout

```
FaceHandTrack/                  the app (Swift)
  FaceHandTrackApp.swift          app entry; command-line self-test
  ContentView.swift               window: camera preview, overlay, controls
  CameraController.swift          AVFoundation camera (RealSense / built-in / iPhone), mirroring
  Pipeline.swift                  per frame: face path || hand path, latest-frame-wins
  FaceDetector.swift              YuNet
  EdgeTAM.swift, TrackManager.swift   EdgeTAM streaming tracker + face track policy (from v1)
  HandTracker.swift               palm detector, hand landmarks, hand tracking
  Gestures.swift                  gesture rules + smoothing
  ImageSampler.swift              model-input crops straight from the camera frame
  PixelBufferScaler.swift, MLArray+Utils.swift   helpers (from v1)
  SelfTest.swift                  command-line checks
  Models/                         Core ML models + edgetam_config.json
FaceHandTrack.xcodeproj         Xcode project
conversion/convert_models.py    how YuNet / palm / landmark models were converted to Core ML
tests/                          self-test scripts and the Python comparison
```

The EdgeTAM Core ML models come from the iOS version
(`../../FaceTrack_v1_iOS/conversion/export_edgetam.py`). To re-make the other three:

```bash
cd conversion
python3 -m venv .venv && .venv/bin/pip install torch onnx onnx2torch coremltools
.venv/bin/python convert_models.py --models ../../onnx/models --out ../FaceHandTrack/Models
```
