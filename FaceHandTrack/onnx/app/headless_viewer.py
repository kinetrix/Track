#!/usr/bin/env python3
"""Run the engine without a screen: for Linux machines, containers and robots.

Input  (--source): an Intel RealSense camera on this Linux machine ("realsense"),
                   a network stream (e.g. the Mac camera via camera_server.py),
                   a video file, a folder of .jpg frames, or a local camera number.
Output: the annotated picture as a web stream (open http://<host>:8090 in a browser),
        optionally saved to a video file, plus timings and events in the terminal,
        and optionally one JSON line per frame (--json) for other programs.

    # Linux PC / robot with a RealSense camera plugged in (colour stream):
    python app/headless_viewer.py --source realsense
    # OrbStack Linux machine, live Mac camera (run camera_server.py on the Mac first):
    python app/headless_viewer.py --source http://host.orb.internal:8080/stream.mjpg
    # a recorded clip, saved to a video, no web stream:
    python app/headless_viewer.py --source clip.mp4 --save out.mp4 --port 0

Live streams use latest-frame-wins (frames that arrive while busy are skipped, as
on the iPhone and Mac). Files and frame folders are processed frame by frame.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app"))
from facetrack import Engine  # noqa: E402
from mac_viewer import draw  # noqa: E402  (the same drawing as the Mac window)
from mjpeg import MJPEGServer  # noqa: E402


class LiveSource(threading.Thread):
    """Reads a live stream / camera continuously and keeps only the newest frame."""

    def __init__(self, src):
        super().__init__(daemon=True)
        if isinstance(src, str) and src.startswith("realsense"):
            from realsense import open_realsense
            self.cap = open_realsense(src)
        else:
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open {src!r}. Is camera_server.py running on the Mac?")
        self.lock = threading.Lock()
        self.frame, self.seq, self.running = None, 0, True

    def run(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame, self.seq = f, self.seq + 1

    def frames(self):
        last = -1
        while self.running:
            with self.lock:
                f, s = self.frame, self.seq
            if f is None or s == last:
                time.sleep(0.002)
                continue
            last = s
            yield f


def file_frames(src):
    if os.path.isdir(src):
        for p in sorted(glob.glob(os.path.join(src, "*.jpg")) + glob.glob(os.path.join(src, "*.png"))):
            yield cv2.imread(p)
        return
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {src!r}")
    while True:
        ok, f = cap.read()
        if not ok:
            return
        yield f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="realsense (or realsense:1 / realsense:<serial>), stream URL, video file, frame folder, "
                         "or camera number")
    ap.add_argument("--port", type=int, default=8090, help="web stream port (0 = off)")
    ap.add_argument("--save", help="also write the annotated video here (.mp4)")
    ap.add_argument("--json", help="write one JSON line per frame (faces, hands, gestures, timings) here")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run until stopped)")
    ap.add_argument("--accel", default="auto", help="auto | cpu | cuda | coreml | directml")
    ap.add_argument("--mirror", action="store_true",
                    help="mirror the picture like a selfie view (use for a camera facing you; not for a robot camera)")
    ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser automatically")
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    args = ap.parse_args()

    print("Loading models (the first launch also measures the fastest backend per model)...")
    engine = Engine(args.models, accel=args.accel, mirrored_input=args.mirror)
    print("backends:", engine.backends)

    src = args.source
    live = src.isdigit() or "://" in src or src.startswith("realsense")
    if live:
        reader = LiveSource(int(src) if src.isdigit() else src)
        reader.start()
        frames = reader.frames()
    else:
        frames = file_frames(src)

    server = MJPEGServer(args.port) if args.port else None
    if server:
        url = f"http://localhost:{args.port}"
        print(f"Watch it in a browser: {url}   (Ctrl+C to stop)")
        if not args.no_open:
            # OrbStack Linux machines have a `mac` command that runs a macOS command,
            # so the Mac's browser opens on the stream; on macOS itself use `open`.
            opener = ["mac", "open", url] if shutil.which("mac") else ["open", url] if sys.platform == "darwin" else None
            if opener:
                subprocess.Popen(opener, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    writer, jlog = None, open(args.json, "w") if args.json else None
    t0 = time.perf_counter()
    n, fps, t_last, last_print = 0, 0.0, None, 0.0
    try:
        for frame in frames:
            if args.mirror:
                frame = cv2.flip(frame, 1)
            res = engine.process(frame, time.perf_counter() - t0 if live else n / 30.0)
            now = time.perf_counter()
            if t_last is not None:
                fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6)
            t_last = now
            for e in res.events:
                print(f"[{res.time:7.2f}s] {e}")
            img = draw(frame, res, not args.no_masks, fps, fps, engine.backends)
            if server:
                server.publish(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 15, (img.shape[1], img.shape[0]))
                writer.write(img)
            if jlog:
                jlog.write(json.dumps(res.to_dict()) + "\n")
            n += 1
            if now - last_print > 2.0:
                t = res.timings
                hands = ", ".join(f"{h.handedness} {h.gesture or '-'}" for h in res.hands) or "none"
                print(f"[FaceTrack] {fps:4.1f} fps | faces {len(res.faces)} | hands: {hands} | " +
                      " ".join(f"{k} {v:.1f}" for k, v in t.items()) + " ms")
                last_print = now
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if live:
            reader.running = False
        if writer:
            writer.release()
            print("saved", args.save)
        if jlog:
            jlog.close()
        if server:
            server.close()
        engine.close()
        if live:
            reader.cap.release()
        print(f"processed {n} frames")


if __name__ == "__main__":
    main()
