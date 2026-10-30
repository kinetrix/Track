"""Desktop check: the Python ONNX engine's face path on the same frames, same output format
as run_clip (C++), so the two can be compared line by line.
    python3 run_clip_python.py <frames dir> [fps=15]
"""
import glob
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.join(HERE, "..", "..", "..", "onnx")
sys.path.insert(0, ONNX)
from facetrack.engine import Engine  # noqa: E402

frames = sorted(glob.glob(os.path.join(sys.argv[1], "*.jpg")))
fps = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
eng = Engine(os.path.join(ONNX, "models"), accel="cpu", hands=False, verbose=False)
for i, f in enumerate(frames):
    res = eng.process(cv2.imread(f), i / fps)
    for e in res.events:
        print(f"# {i} {e}")
    for t in eng.faces.tracks:
        b = t.box
        print(f"{i} {t.id} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f} {int(t.visible)} {t.iou_pred:.3f}")
