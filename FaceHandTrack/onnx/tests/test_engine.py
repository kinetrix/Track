#!/usr/bin/env python3
"""End-to-end check of the engine on a synthetic clip: face frames from the v1
test clips with a hand photo pasted in, moving across the frame and changing
pose half-way (open palm -> victory). Checks that the hand keeps one ID, that
the gesture is right before and after the change, and that faces are tracked;
writes an annotated video for a visual check.

    python tests/test_engine.py --clip ../../FaceTrack_v1_iOS/conversion/eval/clip_selfie
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app"))
from facetrack import Engine  # noqa: E402
from mac_viewer import draw  # noqa: E402


def paste(frame, tile, cx, cy, angle):
    """Paste a hand photo (grey background keyed out roughly) rotated at (cx, cy)."""
    h, w = tile.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[:, 2] += [cx - w / 2, cy - h / 2]
    warped = cv2.warpAffine(tile, M, (frame.shape[1], frame.shape[0]), borderValue=(0, 0, 0))
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (frame.shape[1], frame.shape[0]))
    out = frame.copy()
    out[mask > 0] = warped[mask > 0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "engine_check.mp4"))
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.clip, "*.jpg")))
    first = cv2.imread(files[0])
    H, W = first.shape[:2]
    tiles = [cv2.imread(os.path.join(ROOT, "tests", "hand_images", n)) for n in ("palm.jpg", "peace.jpg")]
    scale = 0.45 * min(H, W) / tiles[0].shape[0]
    tiles = [cv2.resize(t, None, fx=scale, fy=scale) for t in tiles]

    eng = Engine(os.path.join(ROOT, "models"), accel="cpu", mirrored_input=False, verbose=False)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    n = len(files)
    hand_ids, labels, face_counts, totals = set(), [], [], []
    for i, f in enumerate(files):
        a = i / max(n - 1, 1)
        tile = tiles[0] if i < n // 2 else tiles[1]
        frame = paste(cv2.imread(f), tile, int(W * (0.25 + 0.5 * a)), int(H * 0.7), angle=-15 + 30 * a)
        res = eng.process(frame, i / 30.0)
        hand_ids |= {h.id for h in res.hands}
        labels.append(res.hands[0].gesture if res.hands else None)
        face_counts.append(sum(fc.visible for fc in res.faces))
        totals.append(res.timings["total"])
        vw.write(draw(frame, res, True, 0.0, 0.0, eng.backends))
        if i in (n // 4, 3 * n // 4):
            cv2.imwrite(args.out.replace(".mp4", f"_{i}.jpg"), draw(frame, res, True, 0.0, 0.0, eng.backends))
    vw.release()
    first_half = [l for l in labels[2:n // 2]]
    second_half = [l for l in labels[n // 2 + 4:]]
    print(f"frames {n}  hand ids seen {sorted(hand_ids)}  faces visible per frame {min(face_counts)}..{max(face_counts)}")
    print("labels:", labels)
    ok_ids = len(hand_ids) == 1
    ok_a = all(l == "Open palm" for l in first_half)
    ok_b = all(l == "Victory" for l in second_half)
    print(f"one hand id: {ok_ids} | open palm in first half: {ok_a} | victory after the change: {ok_b}")
    print(f"engine time per frame on this machine (CPU): median {np.median(totals):.0f} ms")
    print("wrote", args.out)
    return 0 if ok_ids and ok_a and ok_b else 1


if __name__ == "__main__":
    sys.exit(main())
