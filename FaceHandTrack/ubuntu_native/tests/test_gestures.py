#!/usr/bin/env python3
"""Gesture rules on 24 labelled HaGRID sample photos (tests/hand_images).

Each photo is placed on a 640x480 grey canvas (webcam-like scale), run through
palm detector -> hand landmarks -> rules, and compared with the expected label.
Also checks the tracking path: re-running the landmark model from its own
landmarks (no palm detector) must give the same gesture.

    .venv/bin/python tests/test_gestures.py
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from facehand.backend import pick_device  # noqa: E402
from facehand.gestures import classify  # noqa: E402
from facehand.hands import HandLandmarker, PalmDetector, palm_from_landmarks  # noqa: E402

# HaGRID class -> what the rules should say. None = no rule for this pose (any answer accepted).
EXPECTED = {
    "call": "Call me", "dislike": "Thumbs down", "fist": "Fist", "four": "Open palm", "like": "Thumbs up",
    "mute": "Pointing up", "ok": "OK", "one": "Pointing up", "palm": "Open palm", "peace": "Victory",
    "peace_inv": "Victory", "rock": "Rock", "stop": "Open palm", "stop_inv": "Open palm", "three": "Three",
    "two_up": "Victory", "two_up_inv": "Victory",
    "grabbing": None, "grip": None, "point": None, "pinkie": None, "three2": None, "three3": None,
    "middle_finger": None,
}


def main():
    M = os.path.join(ROOT, "models")
    dev, _ = pick_device("auto")
    pd = PalmDetector(os.path.join(M, "palm_detection_mediapipe_2023feb.onnx"), dev)
    hl = HandLandmarker(os.path.join(M, "handpose_estimation_mediapipe_2023feb.onnx"), dev)
    ok = checked = 0
    for name, want in EXPECTED.items():
        tile = cv2.imread(os.path.join(ROOT, "tests", "hand_images", f"{name}.jpg"))
        canvas = np.full((480, 640, 3), 200, np.uint8)
        canvas[90:90 + tile.shape[0], 170:170 + tile.shape[1]] = tile
        palms = pd(canvas)
        if not palms:
            print(f"{name:14s} NO HAND FOUND")
            checked += want is not None
            continue
        lm, world, _, _ = hl(canvas, palms[0][0])
        got, n = classify(lm, world)
        lm2, world2, _, _ = hl(canvas, palm_from_landmarks(lm))  # tracking path
        got2, _ = classify(lm2, world2)
        if want is None:
            print(f"{name:14s} {got or '-':14s} fingers {n}   (no rule)")
            continue
        checked += 1
        good = got == want and got2 == want
        ok += good
        print(f"{name:14s} {got or '-':14s} fingers {n}   expected {want:12s} {'ok' if good else 'MISS'}"
              + ("" if got2 == got else f"  (tracking path: {got2 or '-'})"))
    print(f"\n{ok}/{checked} correct")
    return 0 if ok == checked else 1


if __name__ == "__main__":
    sys.exit(main())
