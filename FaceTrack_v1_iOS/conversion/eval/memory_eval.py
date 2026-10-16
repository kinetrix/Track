"""Compare memory presets (cond slots C, recent slots R) at a given input size
against known face positions.

    python memory_eval.py longocc_geom.json clip_longocc
    python memory_eval.py group_geom.json clip_group 0 1 2 3 4
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

CONV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, CONV)
from edgetam_coreml_modules import (EdgeTAMImageEncoder, EdgeTAMMaskDecoder, EdgeTAMMemoryAttention,  # noqa: E402
                                    EdgeTAMMemoryEncoder, load_edgetam)
from reference_tracker import ReferenceTracker  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

CKPT = os.path.join(CONV, "EdgeTAM", "checkpoints", "edgetam.pt")
S, P = 512, 16
PRESETS = [(2, 6), (2, 3), (1, 3)]
torch.set_grad_enabled(False)

geo = json.load(open(sys.argv[1]))
files = sorted(glob.glob(os.path.join(sys.argv[2], "*.jpg")))
faces = [int(a) for a in sys.argv[3:]] or [0]
W, H = Image.open(files[0]).size
o = geo["occl"]


def true_box(i, t):
    g, M = geo["gt"][i], np.array(geo["M"][t])
    pts = np.array([[g[0], g[1]], [g[0] + g[2], g[1]], [g[0], g[1] + g[3]], [g[0] + g[2], g[1] + g[3]]])
    q = pts @ M[:, :2].T + M[:, 2]
    return q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()


def status(i, t):
    x0, y0, x1, y1 = true_box(i, t)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if not (0 <= cx < W and 0 <= cy < H):
        return "out"
    if o["t0"] <= t <= o["t1"] and cx < o["x_max"]:
        return "occluded"
    return "visible"


sam = load_edgetam(CKPT, image_size=S)
enc, dec, men = EdgeTAMImageEncoder(sam).eval(), EdgeTAMMaskDecoder(sam).eval(), EdgeTAMMemoryEncoder(sam).eval()
frames = [torch.from_numpy(np.array(Image.open(f).convert("RGB").resize((S, S)), np.float32)).permute(2, 0, 1)[None] for f in files]
encoded = [enc(x) for x in frames]  # the encoder does not depend on the memory preset

for C, R in PRESETS:
    att = EdgeTAMMemoryAttention(sam, C, R, P).eval()
    Lk = (C + R) * 512 + P * 4
    with torch.enable_grad(), FlopCounterMode(display=False) as fc:
        att(encoded[0][2], torch.randn(1, Lk, 64), torch.ones(1, C + R + P))
    trk = ReferenceTracker(enc, att, dec, men, C, R, P, image_size=S)
    for i in faces:
        x0, y0, x1, y1 = true_box(i, 0)
        b = [x0 * S / W, y0 * S / H, x1 * S / W, y1 * S / H]
        hits = {"before": [0, 0], "after": [0, 0]}
        occ_ok = occ = 0
        recover = None
        errs = []
        mem = None
        for t, e in enumerate(encoded):
            if t == 0:
                mem, m, _, _ = trk.add_box(e, b)
            else:
                m, _, _ = trk.track(e, mem)
            up = torch.nn.functional.interpolate(m.float(), size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy() > 0
            st = status(i, t)
            bx0, by0, bx1, by1 = true_box(i, t)
            if st == "visible":
                k = "before" if t < o["t0"] else "after"
                hits[k][1] += 1
                ys, xs = np.nonzero(up)
                hit = len(xs) > 0 and bx0 <= xs.mean() <= bx1 and by0 <= ys.mean() <= by1
                hits[k][0] += int(hit)
                if hit:
                    errs.append(np.hypot((xs.min() + xs.max()) / 2 - (bx0 + bx1) / 2, (ys.min() + ys.max()) / 2 - (by0 + by1) / 2) / (bx1 - bx0))
                if k == "after" and hit and recover is None:
                    recover = t - o["t1"] - 1
            elif st == "occluded":
                occ += 1
                occ_ok += int(up.sum() < 0.1 * (bx1 - bx0) * (by1 - by0))
        print(f"C={C} R={R} ({Lk} keys, mem-attn {fc.get_total_flops()/1e9:.1f} GFLOPs) face{i} {int(x1-x0)}px: "
              f"hit before {hits['before'][0]}/{hits['before'][1]}, after {hits['after'][0]}/{hits['after'][1]} "
              f"(recovered {recover} frames after reappearing), gone while hidden {occ_ok}/{occ}, "
              f"centre err {np.mean(errs):.2f}", flush=True)
