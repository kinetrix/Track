#!/usr/bin/env python3
"""Writes the images the Mac app's self-test runs on (lossless PNG):
    hand_canvases/   the 24 labelled hand photos on a 640x480 grey canvas (as tests/test_gestures.py)
    sequence/        40 frames: a face clip with a hand photo pasted in, moving and changing pose
                     (open palm -> victory), as tests/test_engine.py
    faces/           the first 30 frames of a group clip (several faces)

    python make_test_frames.py --onnx ../../onnx --clips ../../../FaceTrack_v1_iOS/conversion/eval
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(HERE, "..", "..", "onnx"))
    ap.add_argument("--clips", default=os.path.join(HERE, "..", "..", "..", "FaceTrack_v1_iOS", "conversion", "eval"))
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(args.onnx, "tests"))
    from test_engine import paste  # noqa: E402

    hand_dir = os.path.join(args.onnx, "tests", "hand_images")
    out = os.path.join(HERE, "hand_canvases")
    os.makedirs(out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(hand_dir, "*.jpg"))):
        tile = cv2.imread(f)
        canvas = np.full((480, 640, 3), 200, np.uint8)
        canvas[90:90 + tile.shape[0], 170:170 + tile.shape[1]] = tile
        cv2.imwrite(os.path.join(out, os.path.basename(f)[:-4] + ".png"), canvas)

    files = sorted(glob.glob(os.path.join(args.clips, "clip_selfie", "*.jpg")))
    first = cv2.imread(files[0])
    H, W = first.shape[:2]
    tiles = [cv2.imread(os.path.join(hand_dir, n)) for n in ("palm.jpg", "peace.jpg")]
    scale = 0.45 * min(H, W) / tiles[0].shape[0]
    tiles = [cv2.resize(t, None, fx=scale, fy=scale) for t in tiles]
    out = os.path.join(HERE, "sequence")
    os.makedirs(out, exist_ok=True)
    n = len(files)
    for i, f in enumerate(files):
        a = i / max(n - 1, 1)
        tile = tiles[0] if i < n // 2 else tiles[1]
        frame = paste(cv2.imread(f), tile, int(W * (0.25 + 0.5 * a)), int(H * 0.7), angle=-15 + 30 * a)
        cv2.imwrite(os.path.join(out, f"{i:03d}.png"), frame)

    out = os.path.join(HERE, "faces")
    os.makedirs(out, exist_ok=True)
    for i, f in enumerate(sorted(glob.glob(os.path.join(args.clips, "clip_group", "*.jpg")))[:30]):
        cv2.imwrite(os.path.join(out, f"{i:03d}.png"), cv2.imread(f))
    print("done")


if __name__ == "__main__":
    main()
