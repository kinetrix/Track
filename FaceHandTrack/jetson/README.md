# FaceHandTrack: Jetson version (Unitree G1, C++ + TensorRT)

Runs on the G1's development computer (Jetson Orin NX 16 GB, JetPack 5.1.1: CUDA 11.4,
TensorRT 8.5.2, OpenCV 4.2, g++ 9.4) with nothing extra installed and no internet.
Every model runs as a TensorRT fp16 engine built on the robot.

Status: head camera read directly (`/dev/video4`, 1280x720 @ 30, frame rate kept fixed in dim
light) → face path (YuNet on every frame → face tracks with stable IDs, SFace fingerprints give a
returning person their old ID and recognise saved names; up to 3 faces) in parallel with the hand
path (palm detector → 21 hand landmarks → Left/Right + 15 gestures, up to 2 hands per face) →
caregiver rules (same as the Mac app, `../mac_native`): whose hand is whose, who is addressing
the robot, help requests, waving to call the robot and where it should turn → a browser page with
the video, help alerts (Acknowledge button) and name saving. The older EdgeTAM face tracker is
still there (`--tracker edgetam`, ~20-25 ms per face, so too slow for 3 faces at 30 fps).
Next: check the depth rule with two people, ROS 2 Foxy messages, turning toward the person.

## Caregiver rules (no model)

| Rule | Robot version |
|---|---|
| whose hand | nearest face within 5 face widths, hand ≈ 0.6 × face width in size, right hand on the person's right (camera not mirrored), one left + one right per person, sticky; **pose rule** ready (`PersonLinker::update(..., poses)`: hand wrist on a person's pose wrist = theirs, both visible wrists far away = not theirs) but the robot doesn't run the pose model yet, so a farther person's hand can still go to the nearer person |
| depth | `--camera realsense` (the default when built with librealsense, which ROS Noetic's package provides): colour + depth from the head D435i. A hand can only belong to a person whose face is at about its distance (at most 0.45 m nearer the camera, 0.35 m farther; own hands were at most 0.3 m nearer, wrongly linked hands 0.54-0.75 m), 4 cost per metre of difference. Arm check: along the line from the wrist to that person's shoulder, the share of depth readings on a body at the expected depth (2 x the missing share as cost; the line to someone else's shoulder crosses background or a nearer person); the status line shows each face's and hand's distance, and the log each new link's. Depth is read only at the points the rules need (face centre, palm), mapped with the camera's calibration, so no whole-frame alignment |
| steadier | a hand moves to another person only after 5 frames in a row say so (at once if its person is gone); a hand lost and found again within 15 frames where it was keeps its number, person and gesture; with depth, a face with no usable fingerprint that reappears where a face was lost (within 3 s, 1.5 face widths, 0.25 m) gets that face's number back |
| addressing | facing the camera (YuNet points) + the same gesture held 0.5 s by that person's hand |
| help | Signal for Help (thumb-tucked palm, then a fist, wrist still) **twice within 6 s** (once was too easy to trigger by relaxing a hand: 12 alerts in 7 min on the Mac), or both hands open above the chin 2 s; the page shows "give it once more" after the first; ignored from a person turned away from the camera |
| wave | open hand, 3 side-to-side swings within 2 s, **measured relative to the person's face** so the robot turning doesn't look like a wave |
| attention / turn | the last person who waved (or started addressing it); horizontal / vertical angle from the 69° camera, robot's left / right |
| names | page: type a name, Save face (the person talking, else the largest face); kept in `~/facehandtrack/known_faces.txt` |
| hands | a tracked hand is kept down to presence 0.45 (0.6 before: moving hands were lost and re-found constantly) |

Checked by `tests/desktop/test_rules.cpp` (21 constructed scenes, all pass).

Speed: the frame is uploaded to the GPU once and every model input is made there by one CUDA
kernel (`preprocess.cu`); the CPU waits only for results it needs (masks, scores); each
model's GPU work is recorded once as a CUDA graph and replayed (`--no-graphs` turns that off).

## Files

```
src/            C++ app: trt_model (TensorRT engine wrapper, CUDA graphs, GPU timers),
                preprocess (GPU frame + input kernel), camera (V4L2 / videohub reader
                thread), face_detector (YuNet), face_source (what main needs from a face
                path), fast_face_tracker (default face path), face_embedder (SFace),
                edgetam (streaming tracker, GPU-resident memory), face_tracker (EdgeTAM
                track policy), face_pipeline (EdgeTAM face path), person_linker, help_detector,
                attention_tracker (caregiver rules), hands (palm detector,
                landmarks, hand tracker), gestures, hand_pipeline, mjpeg_server, main
tests/desktop/  the same face + hand paths on ONNX Runtime (CPU), compared frame by frame
                with the Python engine
build.sh        compiles ./facehandtrack on the robot
convert/        makes the EdgeTAM ONNX graphs buildable with TensorRT 8.5
models/         converted EdgeTAM graphs (generated by convert/, not in git)
```

## On the robot

Layout used: `~/facehandtrack/models` (ONNX), `~/facehandtrack/engines` (TensorRT
engines, `<model>.fp16.engine`, built with `trtexec`), `~/facehandtrack/jetson` (this folder).

The head camera (RealSense D435i colour, `/dev/video4`) is held by Unitree's `videohub_pc4`
until it is stopped; a reboot starts it again.

```bash
sudo pkill -f videohub_pc4  # if it comes back at once (Unitree's master service restarts it), run
                            # it on the same line as the app: sudo pkill -f videohub_pc4; ./facehandtrack
sudo fuser -v /dev/video*  # nothing should list /dev/video4
cd ~/facehandtrack/jetson
./build.sh
./facehandtrack            # then open http://192.168.123.164:8090/ on the laptop
```

The page shows the video, help alerts with an **Acknowledge** button, who the robot is attending
to and how far to turn, and a box to save the face of the person talking to the robot under a name.

Face path (default): YuNet on every frame; each detection is linked to the track whose
predicted box (constant velocity) it overlaps most. A track missing for 1 s is dropped, but
its SFace fingerprint is kept for 30 s: a face that appears and matches a fingerprint (cosine
>= 0.55 and 0.10 above the next-best) gets that ID back. Fingerprints are taken only from faces
at least 64 px tall (smaller ones look too alike between people) and refreshed once a second
from an unobstructed view. A face that just appeared keeps being checked against lost faces for
3 s (a fresh fingerprint every 0.25 s), since its first view is often blurred or turned; a newer
fingerprint of a person replaces their older one in the gallery.

Options: `--camera /dev/video4|videohub|<gstreamer pipeline> --width 1280 --height 720 --fps 30
--max-faces 3 --max-hands 6 --no-faces --no-hands --tracker fast|edgetam --no-reid
--attn fp16|fp32 --compare-attn --no-graphs --mirrored --port 8090 --stream-fps 15
--jpeg-quality 80 --engines ~/facehandtrack/engines --names ~/facehandtrack/known_faces.txt`
(`--width/--height/--fps` only for `/dev/videoN`; `--max-hands` defaults to 2 x `--max-faces`).
`--no-reid` skips SFace (IDs are not given back). `--attn`/`--compare-attn` are for
`--tracker edgetam`: the latter also runs the fp32 memory attention and prints how closely the
fp16 masks match it. Ctrl+C stops it.

Engines expected in `--engines` (`.engine` files, built by `trtexec` from `models/` and
`../onnx/models/`): `yunet_640x384.fp16`, `face_recognition_sface_2021dec.fp16`,
`palm_detection_mediapipe_2023feb.fp16`, `handpose_estimation_mediapipe_2023feb.fp16`; for
`--tracker edgetam` also `edgetam_image_encoder.fp16`, `edgetam_memory_attention.fp16`
(+ `.fp32` for `--attn fp32` / `--compare-attn`), `edgetam_mask_decoder_n1.fp16`,
`edgetam_mask_decoder_n2.fp16`, `edgetam_memory_encoder.fp16`.
`--mirrored` only for a selfie-style mirrored camera (it swaps Left / Right).
