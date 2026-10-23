#!/usr/bin/env python3
"""Compares the Mac app's self-test output with the Python (ONNX) engine on the same images.

    ../../onnx/.venv/bin/python compare_selftest.py logs/hands.json logs/sequence.json logs/faces.json

hands.json     per image: palm box, 21 landmarks, handedness score, gesture (single image + tracking path)
sequence.json  per frame: face boxes/ids and hand ids/gestures/landmarks over a moving-hand clip
faces.json     per frame: face boxes/ids over a group clip
Any of the three may be missing.
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.join(HERE, "..", "..", "onnx")
sys.path.insert(0, ONNX)
from facetrack import Engine  # noqa: E402
from facetrack.gestures import classify  # noqa: E402
from facetrack.hands import HandLandmarker, PalmDetector, palm_from_landmarks  # noqa: E402
from facetrack.runtime import Runtime  # noqa: E402

M = os.path.join(ONNX, "models")


def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def hands(path):
    rows = json.load(open(path))
    rt = Runtime("cpu", verbose=False)
    pd = PalmDetector(rt, os.path.join(M, "palm_detection_mediapipe_2023feb.onnx"))
    hl = HandLandmarker(rt, os.path.join(M, "handpose_estimation_mediapipe_2023feb.onnx"))
    same_g = same_h = n = 0
    lm_err = []
    for r in rows:
        img = cv2.imread(os.path.join(HERE, "hand_canvases", r["name"] + ".png"))
        palms = pd(img)
        if not palms or "landmarks" not in r:
            print(f"{r['name']:14s} python palms {len(palms)}  swift palms {r['palms']}")
            continue
        lm, world, right, pres = hl(img, palms[0][0])
        g, _ = classify(lm, world)
        g2, _ = classify(*hl(img, palm_from_landmarks(lm))[:2])
        s_lm = np.array(r["landmarks"]).reshape(21, 3)
        palm_size = np.linalg.norm(lm[9, :2] - lm[0, :2])
        err = np.abs(s_lm[:, :2] - lm[:, :2]).max()
        lm_err.append(err / palm_size)
        n += 1
        same_g += g == r.get("gesture") and g2 == r.get("gesture_tracking")
        same_h += (right > 0.5) == (r["right"] > 0.5)
        print(f"{r['name']:14s} gesture py {g or '-':12s} swift {r.get('gesture') or '-':12s}"
              f" | tracking py {g2 or '-':12s} swift {r.get('gesture_tracking') or '-':12s}"
              f" | right py {right:.2f} swift {r['right']:.2f} | presence py {pres:.2f} swift {r['presence']:.2f}"
              f" | landmarks max diff {err:5.1f} px ({100 * err / palm_size:4.1f}% of palm)")
    print(f"\nhands: same gestures {same_g}/{n}, same handedness {same_h}/{n}, "
          f"landmark max diff median {100 * np.median(lm_err):.1f}% / worst {100 * np.max(lm_err):.1f}% of palm length")


def sequence(path, folder, name):
    rows = json.load(open(path))
    eng = Engine(M, accel="cpu", mirrored_input=False, verbose=False)
    files = sorted(os.listdir(os.path.join(HERE, folder)))
    ious, id_same, g_same, g_n, faces_n = [], 0, 0, 0, [0, 0]
    for i, (f, r) in enumerate(zip(files, rows)):
        res = eng.process(cv2.imread(os.path.join(HERE, folder, f)), i / 30.0)
        py = {fc.id: fc for fc in res.faces if fc.visible}
        sw = {fc["id"]: fc for fc in r["faces"] if fc["visible"]}
        faces_n[0] += len(py)
        faces_n[1] += len(sw)
        id_same += sorted(py) == sorted(sw)
        for k in set(py) & set(sw):
            ious.append(box_iou(py[k].box, sw[k]["box"]))
        for h_py, h_sw in zip(res.hands, r["hands"]):
            g_n += 1
            g_same += h_py.gesture == h_sw["gesture"] and h_py.handedness == h_sw["handedness"]
    print(f"{name}: frames {len(rows)} | same face ids {id_same}/{len(rows)} | visible faces py {faces_n[0]} swift {faces_n[1]}"
          f" | face box IoU median {np.median(ious) if ious else float('nan'):.3f} min {min(ious) if ious else float('nan'):.3f}"
          + (f" | hands: same gesture+handedness {g_same}/{g_n}" if g_n else "")
          + f" | swift total ms median {np.median([r['total_ms'] for r in rows[3:]]):.1f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        if not os.path.exists(p):
            continue
        b = os.path.basename(p)
        if b.startswith("hands"):
            hands(p)
        elif b.startswith("sequence"):
            sequence(p, "sequence", "sequence (moving hand + face)")
        elif b.startswith("faces"):
            sequence(p, "faces", "faces (group clip)")
