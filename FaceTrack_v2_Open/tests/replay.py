#!/usr/bin/env python3
"""Replay recorded frames (app/mac_viewer.py --record DIR) through the engine and
print per-frame hands, gestures and events. Used to reproduce live problems offline.

    python tests/replay.py recordings/session [--accel cpu] [--start 0] [--end 200]
"""
import argparse
import glob
import os
import re
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from facetrack import Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--accel", default="auto")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10 ** 9)
    ap.add_argument("--no-faces", action="store_true")
    args = ap.parse_args()
    eng = Engine(os.path.join(ROOT, "models"), accel=args.accel, faces=not args.no_faces, verbose=False)
    print("backends:", eng.backends, flush=True)
    files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))[args.start:args.end]
    for f in files:
        m = re.search(r"_(\d+\.\d+)\.jpg$", f)
        t = float(m.group(1)) if m else 0.0
        r = eng.process(cv2.imread(f), t)
        hands = [(h.id, h.handedness, round(h.score, 2), h.raw_gesture or "-", h.gesture or "-") for h in r.hands]
        print(os.path.basename(f)[:5], "faces", len(r.faces), "hands", hands, r.events,
              f"{r.timings.get('total', 0):.0f} ms", flush=True)


if __name__ == "__main__":
    main()
