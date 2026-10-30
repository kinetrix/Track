"""Desktop check: the Python ONNX engine (faces + hands) on the same frames, same output format
as run_clip (C++), so the two can be compared line by line.
    python3 run_clip_python.py <frames dir> [fps=15] [--mirrored]
"""
import glob
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.join(HERE, "..", "..", "..", "onnx")
sys.path.insert(0, ONNX)
from facetrack.engine import Engine  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
frames = sorted(glob.glob(os.path.join(args[0], "*.jpg")))
fps = float(args[1]) if len(args) > 1 else 15.0
eng = Engine(os.path.join(ONNX, "models"), accel="cpu", mirrored_input="--mirrored" in sys.argv, verbose=False)
u = lambda s: (s or "-").replace(" ", "_")  # noqa: E731
for i, f in enumerate(frames):
    res = eng.process(cv2.imread(f), i / fps)
    for e in res.events:
        print(f"# {i} {e}")
    for t in eng.faces.tracks:
        b = t.box
        print(f"{i} {t.id} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f} {int(t.visible)} {t.iou_pred:.3f}")
    for h in res.hands:
        lm = h.landmarks
        print(f"H {i} {h.id} {h.handedness} {u(h.gesture)} {u(h.raw_gesture)} {h.fingers} "
              f"{lm[0, 0]:.1f} {lm[0, 1]:.1f} {lm[8, 0]:.1f} {lm[8, 1]:.1f}")
