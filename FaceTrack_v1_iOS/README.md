# FaceTrack

On-device face detection and tracking for iPhone, using only open models:

* **YuNet** (MIT) detects faces.
* **EdgeTAM** (Apache-2.0), Meta's on-device SAM 2, tracks them with pixel masks and persistent IDs.

Both run on Core ML. The design decisions are explained in **[DESIGN.md](DESIGN.md)**.

The converted models are already in `ios/FaceTrack/Models`, so you can build the app without Python.

## Build and run on your iPhone

You need Xcode 16 or later and an iPhone on iOS 18 or later.

1. Open `ios/FaceTrack.xcodeproj` in Xcode.
2. In Xcode, open *Signing & Capabilities* and select your team.
3. Choose your iPhone as the run destination and press Run.

You can also pick **My Mac (Designed for iPhone)** as the destination. On an Apple Silicon Mac this runs the same app with the Mac's camera and Neural Engine.

The first launch compiles the models for the Neural Engine, which takes a few seconds. After that, launches are instant.

In the app:

* The HUD shows FPS and the milliseconds spent on each stage.
* **Flip** switches between the front and back camera.
* **Masks** / **Boxes** toggles the mask overlay.
* **Reset** drops all tracks.

The project picks up every file in `ios/FaceTrack/` automatically, including the models. `ios/project.yml` is an XcodeGen spec you can use instead if you prefer to generate the project.

## Re-running the conversions (optional)

Run these on your Mac.

1. Install the dependencies and clone EdgeTAM (code plus checkpoint):
   ```bash
   cd conversion
   ./setup.sh
   source .venv/bin/activate
   ```
2. Convert the detector. The width and height set the detector's input size:
   ```bash
   python convert_yunet.py --width 384 --height 640
   ```
3. Export EdgeTAM:
   ```bash
   python export_edgetam.py
   ```
   Useful options:
   * `--image-size 512` (default), `768` or `1024` (the paper setting) for EdgeTAM's input size
   * `--num-cond 1 --num-recent 3` for the faster memory preset
   * `--weights int8` for smaller models

## Checking the export

Compare the streaming export with Meta's official `SAM2VideoPredictor`. This runs anywhere PyTorch runs:

```bash
python validate_edgetam.py --frames <folder of jpgs> --box x1 y1 x2 y2 --image-size 512
```

Compare the Core ML packages with PyTorch and see per-graph latency. This needs macOS:

```bash
python verify_coreml_mac.py --frames <folder of jpgs> --box x1 y1 x2 y2 --compute-units CPU_AND_NE
```

Run the whole pipeline on a clip and write an annotated video, which is useful for tuning `TrackPolicy`:

```bash
python pipeline_reference.py --frames <folder of jpgs> --out annotated.mp4
```
