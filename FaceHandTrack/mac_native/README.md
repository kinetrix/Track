# FaceHandTrack: Mac-native version (Swift + Core ML)

A native macOS app: face detection + tracking and hand gesture tracking from an
Intel RealSense camera's colour stream (or the MacBook camera), with every model
running through Apple's Core ML on the Neural Engine / GPU. No Python at run time.

| Job | Model | Runs on |
|---|---|---|
| Find faces | YuNet (OpenCV Zoo, MIT), 640x384, every frame | GPU |
| Face fingerprints (a returning face gets its old ID back; saved names) | SFace (OpenCV Zoo, Apache-2.0), 112x112 aligned crop | Neural Engine |
| Track faces with pixel masks (optional, `Face tracker: EdgeTAM`) | EdgeTAM (Meta, Apache-2.0), 512x512, streaming memory | Neural Engine |
| Find hands | MediaPipe palm detector (Google, Apache-2.0) | GPU |
| 21 points per hand | MediaPipe hand landmarks (Google, Apache-2.0) | GPU |
| Gestures | hand-written rules on the 21 points | CPU (Swift) |
| Whose hand, who is talking to the camera | rules on the face and hand boxes (no model) | CPU (Swift) |
| Help requests (Signal for Help, both hands up) | rules on the hand points (no model) | CPU (Swift) |
| Waving to call the robot, where to turn | rules on hand motion and face position (no model) | CPU (Swift) |

The default face tracker is the one from the robot version (`../jetson`): YuNet on every
frame, faces followed from frame to frame, SFace fingerprints to give a returning face its old
ID back (a returning face keeps being checked for 3 s, a fresh fingerprint every 0.25 s, since
its first view is often blurred or turned; the console says why a face kept a new ID, and the
status line shows each face's closest saved name and how well its fingerprints agree). The EdgeTAM tracker from the iOS app (`../../FaceTrack_v1_iOS`) is still there for
pixel masks. The hand tracker and gesture rules are line-by-line Swift ports of the Python engine.

## Who is talking to the camera

Every hand is linked to the person (face) it belongs to, and a person who faces the camera and
holds a gesture is marked as addressing it. No extra model, a few comparisons per frame:

| Rule | What it does |
|---|---|
| reach | a hand belongs only to a face within 5 face widths |
| size | a person's hand (wrist to middle knuckle) is about 0.6 x their face width, so a hand much smaller or larger than a face belongs to someone nearer or farther away (a depth cue from one camera) |
| side | a right hand is usually on the person's right of their face (follows the Mirror setting) |
| one each | each person has at most one left and one right hand |
| sticky | a hand keeps its person unless another fits clearly better |
| facing | nose roughly between the eyes (YuNet's points) = turned to the camera |
| addressing | facing + the same gesture held 0.5 s by that person's hand (restarts if the hand changes person); stays on 1 s after the gesture ends |

The window draws each hand in its person's colour with a dashed line to their face and the
label `#2 · Right: Thumbs up (1)` (`?` = no one). The person addressing the camera gets a yellow
frame and a `talking: Thumbs up` banner. The console prints `hand #3 (Right) -> face #1 (reach 2.1,
size 0.64)` when a hand changes person (the size numbers are for tuning the rule) and
`face #1 addressing: Thumbs up` when someone starts addressing it. Rules and numbers:
`PersonLinker.swift` (`LinkPolicy`).

## Help requests

Two deliberate signals raise a red **HELP REQUESTED** banner naming the person (`Vignesh (#1)`,
`face #2`, or `someone (face not seen)`), with the time, and play the alert sound. The banner
stays until **Acknowledge** is clicked; the same person can't raise another for 10 s.

| Signal | How |
|---|---|
| Signal for Help | palm to the camera with the thumb tucked in (four fingers up), hold about half a second, then fold the fingers down over the thumb (a fist) within a second, hand held still (a waving, blurred hand can look like this) |
| Both hands up | both hands open and raised above the chin, held 2 s |

No model: finger states from the hand points (`HelpDetector.swift`, `HelpPolicy`). The console
prints `HELP REQUESTED: Signal for Help from Vignesh (#1)`. On the robot this becomes a ROS 2
message (not yet). A flat "stop" palm with the thumb pressed to the side also counts as the tucked
palm, so stop-then-fist within a second triggers it too.

## Waving to call the robot

An open hand waved side to side at about shoulder height or above (3 swings of about 5 cm or
more within 2 s, counted per person and hand side so it survives the hand tracker losing a fast
hand) calls the robot: that person's label shows `waving`, they get the robot's
attention, and a cyan badge at the bottom says where the robot should turn to face them, e.g.
`Robot → #1 Vignesh (waved) · turn left 12°, up 3°` (robot's left / right, from the camera's
field of view: 70° assumed here, 69° on the robot's D435i). Someone who starts addressing the
camera also gets the attention when no one has it. The attention is released when that person is
out of sight for 2 s or shows nothing for 30 s, or someone else waves. Console: `face #1 waved:
calling the robot`, `robot attention -> face #1 (waved)`, and the turn on the status line. On the
robot the turn becomes a head / body command (not yet). `AttentionTracker.swift`, `AttentionPolicy`.

**Names.** Type a name and click **Save face**: the fingerprint of the person addressing the
camera (else the largest face) is saved, and that person shows as `#1 Vignesh` from then on,
also after a restart. Faces need to be at least 64 px tall. Save a few views (straight on,
slightly turned) for steadier recognition. **Forget names** clears them. Saved in
`~/Library/Application Support/FaceHandTrack/known_faces.json`.

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
| Face tracker | Fast (YuNet + SFace, default, up to 3 faces) or EdgeTAM (pixel masks, up to 4) |
| Mirror | selfie view (Left/Right always name the person's real hand) |
| Masks / Faces / Hands | show face masks (EdgeTAM only); turn the face or hand path off |
| Reset | forget all tracks |
| Name, Save face, Forget names | saved names (below) |

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
  Pipeline.swift                  per frame: face path || hand path, then person links; saved names
  FaceDetector.swift              YuNet
  FastFaceTracker.swift           default face tracker (as ../jetson): motion + overlap, re-ID
  FaceEmbedder.swift              SFace fingerprints (5-point alignment)
  PersonLinker.swift              whose hand, who is facing / addressing the camera
  HelpDetector.swift              help requests: Signal for Help, both hands up
  AttentionTracker.swift          waving to call the robot; who it attends to and where to turn
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
(`../../FaceTrack_v1_iOS/conversion/export_edgetam.py`). To re-make the other four (SFace fp16
matches the ONNX model at cosine 0.99999):

```bash
cd conversion
python3 -m venv .venv && .venv/bin/pip install torch onnx onnx2torch coremltools
.venv/bin/python convert_models.py --models ../../onnx/models --out ../FaceHandTrack/Models
```
