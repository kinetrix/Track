"""Independent quality check on clip_group: score each EdgeTAM input size against
the known (analytic) face positions instead of against the 1024 output.

Per face and size:
  hit rate   visible frames where the mask exists and its centroid is inside the true face box
  centre err mean |mask-box centre - true face centre| / true face width (visible frames)
  occl. ok   occluded frames where the mask correctly disappears
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

from compare_sizes import build, run

torch.set_grad_enabled(False)
geo = json.load(open("group_geom.json"))
files = sorted(glob.glob("clip_group/*.jpg"))
W, H = Image.open(files[0]).size
faces = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]


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
    o = geo["occl"]
    if o["t0"] <= t <= o["t1"] and cx < o["x_max"]:
        return "occluded"
    return "visible"


rows = []
for S in (1024, 768, 512):
    trk, _ = build(S)
    for i in faces:
        x0, y0, x1, y1 = true_box(i, 0)
        masks, _ = run(trk, files, [x0, y0, x1, y1], W, H)
        hits, errs, vis, occ_ok, occ = 0, [], 0, 0, 0
        for t, m in enumerate(masks):
            st = status(i, t)
            bx0, by0, bx1, by1 = true_box(i, t)
            if st == "visible":
                vis += 1
                ys, xs = np.nonzero(m)
                if len(xs):
                    mx, my = xs.mean(), ys.mean()
                    if bx0 <= mx <= bx1 and by0 <= my <= by1:
                        hits += 1
                    ccx, ccy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
                    errs.append(np.hypot(ccx - (bx0 + bx1) / 2, ccy - (by0 + by1) / 2) / (bx1 - bx0))
            elif st == "occluded":
                occ += 1
                occ_ok += int(m.sum() < 0.1 * (bx1 - bx0) * (by1 - by0))
        rows.append((S, i, int(x1 - x0), hits, vis, np.mean(errs) if errs else float("nan"), occ_ok, occ))
        print(f"S={S:4d} face{i} ({int(x1-x0)} px wide): hit {hits}/{vis}  centre err {rows[-1][5]:.2f} face-widths  "
              f"occluded->gone {occ_ok}/{occ}", flush=True)
