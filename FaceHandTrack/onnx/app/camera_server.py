#!/usr/bin/env python3
"""Mac side: share the Mac's camera over the network as an MJPEG stream.

A Linux machine (OrbStack VM, Docker container, a robot's computer) cannot use
the MacBook's built-in camera directly, so the Mac sends it the frames.

    .venv/bin/python app/camera_server.py            # http://localhost:8080/stream.mjpg
    .venv/bin/python app/camera_server.py --camera builtin --port 8080

Frames are sent as the camera gives them (not mirrored); the receiver decides.
Press Ctrl+C to stop.
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mac_viewer import resolve_camera  # noqa: E402
from mjpeg import MJPEGServer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="auto",
                    help="auto (default: RealSense if plugged in, else the Mac's camera), realsense, builtin, "
                         "a number, or part of a camera name")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality 1-100")
    args = ap.parse_args()

    idx = resolve_camera(str(args.camera))
    api = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(idx, api)
    for _ in range(30):  # first run: wait for the macOS camera permission answer
        if cap.isOpened():
            break
        print("Waiting for camera access (allow Terminal in the macOS prompt)...")
        time.sleep(2)
        cap = cv2.VideoCapture(idx, api)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Allow Terminal in System Settings > Privacy & Security > Camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    server = MJPEGServer(args.port)
    print(f"Streaming camera {idx} at http://localhost:{args.port}/stream.mjpg")
    print(f"From an OrbStack Linux machine use: http://host.orb.internal:{args.port}/stream.mjpg   (Ctrl+C to stop)")
    n, t0 = 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if ok:
                server.publish(jpg.tobytes())
            n += 1
            if time.time() - t0 >= 5:
                print(f"[camera] {n / (time.time() - t0):.1f} FPS, {frame.shape[1]}x{frame.shape[0]}")
                n, t0 = 0, time.time()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        server.close()


if __name__ == "__main__":
    main()
