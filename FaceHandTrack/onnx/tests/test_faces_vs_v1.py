#!/usr/bin/env python3
"""The ONNX face path must behave like the v1 PyTorch reference.

Runs the v2 engine (faces only) and the v1 reference pipeline (YuNet + EdgeTAM
PyTorch modules, same policy) on the same clip and compares, frame by frame,
the track IDs and the mask overlap (IoU) of each face.

Needs the export environment (PyTorch + EdgeTAM checkout, see export/README):
    python tests/test_faces_vs_v1.py --clip ../../FaceTrack_v1_iOS/conversion/eval/clip_group
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "export"))
from facetrack import Engine  # noqa: E402


def reference(files, fps, cfg):
    """v1 pipeline, with the PyTorch modules (detector at the same size as v2)."""
    import torch
    from edgetam_modules import (EdgeTAMImageEncoder, EdgeTAMMaskDecoder, EdgeTAMMemoryAttention,
                                 EdgeTAMMemoryEncoder, load_edgetam)
    from v1_reference import Detector, Pipeline, ReferenceTracker
    torch.set_grad_enabled(False)
    S, C, R, P = cfg["image_size"], cfg["num_cond"], cfg["num_recent"], cfg["num_ptr"]
    sam = load_edgetam(os.path.join(ROOT, "export", "EdgeTAM", "checkpoints", "edgetam.pt"), image_size=S)
    trk = ReferenceTracker(EdgeTAMImageEncoder(sam), EdgeTAMMemoryAttention(sam, C, R, P), EdgeTAMMaskDecoder(sam),
                           EdgeTAMMemoryEncoder(sam), C, R, P, S)
    det = Detector(os.path.join(ROOT, "models", "face_detection_yunet_2023mar.onnx"), cfg["det_width"],
                   cfg["det_height"])
    pipe = Pipeline(trk, det, fps=fps)
    out = []
    for f in files:
        out.append({t.tid: (t.mask.copy() if t.visible else None) for t in pipe.step(cv2.imread(f))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.clip, "*.jpg")))
    eng = Engine(os.path.join(ROOT, "models"), accel="cpu", hands=False, verbose=False)
    v2 = []
    for i, f in enumerate(files):
        r = eng.process(cv2.imread(f), i / args.fps)
        v2.append({fc.id: (fc.mask if fc.visible else None) for fc in r.faces})
    v1 = reference(files, args.fps, eng.cfg)

    ious, id_mismatch = [], 0
    for i, (a, b) in enumerate(zip(v1, v2)):
        if set(a) != set(b):
            id_mismatch += 1
            print(f"frame {i}: v1 ids {sorted(a)} v2 ids {sorted(b)}")
        for k in set(a) & set(b):
            if a[k] is None or b[k] is None:
                if (a[k] is None) != (b[k] is None):
                    print(f"frame {i} face {k}: visible in only one version")
                continue
            ma, mb = a[k] > 0, b[k] > 0
            u = (ma | mb).sum()
            ious.append((ma & mb).sum() / u if u else 1.0)
    print(f"{os.path.basename(args.clip)}: {len(files)} frames, frames with different IDs: {id_mismatch}, "
          f"mask IoU v1 vs v2: mean {np.mean(ious):.4f}, min {np.min(ious):.4f} over {len(ious)} face-frames")


if __name__ == "__main__":
    main()
