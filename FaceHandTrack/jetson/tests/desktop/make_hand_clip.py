"""Desktop check input: v1 face frames with a hand photo pasted in, moving across the frame and
changing pose half-way (open palm -> victory), as onnx/tests/test_engine.py does.
    python3 make_hand_clip.py <face clip dir> <out dir>
"""
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(HERE, "..", "..", "..", "onnx", "tests", "hand_images")


def paste(frame, tile, cx, cy, angle):
    h, w = tile.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[:, 2] += [cx - w / 2, cy - h / 2]
    warped = cv2.warpAffine(tile, M, (frame.shape[1], frame.shape[0]), borderValue=(0, 0, 0))
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (frame.shape[1], frame.shape[0]))
    out = frame.copy()
    out[mask > 0] = warped[mask > 0]
    return out


files = sorted(glob.glob(os.path.join(sys.argv[1], "*.jpg")))
os.makedirs(sys.argv[2], exist_ok=True)
H, W = cv2.imread(files[0]).shape[:2]
tiles = [cv2.imread(os.path.join(IMAGES, n)) for n in ("palm.jpg", "peace.jpg")]
scale = 0.45 * min(H, W) / tiles[0].shape[0]
tiles = [cv2.resize(t, None, fx=scale, fy=scale) for t in tiles]
n = len(files)
for i, f in enumerate(files):
    a = i / max(n - 1, 1)
    tile = tiles[0] if i < n // 2 else tiles[1]
    frame = paste(cv2.imread(f), tile, int(W * (0.25 + 0.5 * a)), int(H * 0.7), angle=-15 + 30 * a)
    cv2.imwrite(os.path.join(sys.argv[2], f"{i:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
