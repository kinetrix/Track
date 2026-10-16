#!/usr/bin/env python3
"""Compare EdgeTAM input sizes (1024 / 768 / 512) on face clips.

For each clip and starting face box: run the streaming tracker (same modules and
memory policy as the app) at every size, upsample masks to frame resolution and
report agreement with the 1024 result, plus GFLOPs per stage.

    python compare_sizes.py --clip clip_selfie --box 314 555 455 766
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode

CONV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, CONV)
from edgetam_coreml_modules import (EdgeTAMImageEncoder, EdgeTAMMaskDecoder, EdgeTAMMemoryAttention,  # noqa: E402
                                    EdgeTAMMemoryEncoder, load_edgetam)
from reference_tracker import ReferenceTracker  # noqa: E402

CKPT = os.path.join(CONV, "EdgeTAM", "checkpoints", "edgetam.pt")
C, R, P = 2, 6, 16  # deployed memory preset


def gflops(m, *a):
    with torch.enable_grad(), FlopCounterMode(display=False) as fc:  # counter's module hooks need autograd
        m(*a)
    return fc.get_total_flops() / 1e9


def build(S):
    sam = load_edgetam(CKPT, image_size=S)
    mods = (EdgeTAMImageEncoder(sam).eval(), EdgeTAMMemoryAttention(sam, C, R, P).eval(),
            EdgeTAMMaskDecoder(sam).eval(), EdgeTAMMemoryEncoder(sam).eval())
    return ReferenceTracker(*mods, C, R, P, image_size=S), mods


def run(trk, files, box, W, H):
    S = trk.image_size
    b = [box[0] * S / W, box[1] * S / H, box[2] * S / W, box[3] * S / H]
    masks, mem, t = [], None, 0.0
    for i, f in enumerate(files):
        x = torch.from_numpy(np.array(Image.open(f).convert("RGB").resize((S, S)), np.float32)).permute(2, 0, 1)[None]
        t0 = time.perf_counter()
        e = trk.encode(x)
        if i == 0:
            mem, m, _, _ = trk.add_box(e, b)
        else:
            m, _, _ = trk.track(e, mem)
        t += time.perf_counter() - t0
        up = torch.nn.functional.interpolate(m.float(), size=(H, W), mode="bilinear", align_corners=False)
        masks.append(up[0, 0].numpy() > 0)
    return masks, t / len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--box", type=float, nargs=4, action="append", required=True, help="x y w h (YuNet format)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 768, 512])
    ap.add_argument("--flops", action="store_true")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    files = sorted(glob.glob(os.path.join(args.clip, "*.jpg")))
    W, H = Image.open(files[0]).size
    boxes = [[x, y, x + w, y + h] for x, y, w, h in args.box]

    res = {}
    for S in args.sizes:
        trk, mods = build(S)
        if args.flops:
            enc, att, dec, men = mods
            e = enc(torch.rand(1, 3, S, S) * 255)
            Lk = (C + R) * 512 + P * 4
            pix = att(e[2], torch.randn(1, Lk, 64), torch.ones(1, C + R + P))
            g = (gflops(enc, torch.rand(1, 3, S, S) * 255), gflops(att, e[2], torch.randn(1, Lk, 64), torch.ones(1, C + R + P)),
                 gflops(dec, pix, e[3], e[4], torch.zeros(1, 1, 2), -torch.ones(1, 1)),
                 gflops(men, e[0], torch.randn(1, 1, S // 4, S // 4), torch.zeros(1)))
            print(f"S={S}: GFLOPs encoder {g[0]:.1f} | mem-attn/face {g[1]:.1f} | decoder/face {g[2]:.1f} | mem-enc/face {g[3]:.1f}"
                  f" | 1 face total {sum(g):.1f}")
        res[S] = [run(trk, files, b, W, H) for b in boxes]

    ref = res[args.sizes[0]]
    for S in args.sizes[1:]:
        for k, b in enumerate(boxes):
            ious = []
            for a, c in zip(ref[k][0], res[S][k][0]):
                u = (a | c).sum()
                ious.append((a & c).sum() / u if u else 1.0)
            area = np.mean([m.sum() for m in ref[k][0]])
            print(f"{os.path.basename(args.clip)} face{k} (start {int(b[2]-b[0])}x{int(b[3]-b[1])} px, mean mask {area:.0f} px): "
                  f"S={S} vs {args.sizes[0]}  mean IoU {np.mean(ious):.3f}  min {np.min(ious):.3f}  "
                  f"CPU {1000*res[S][k][1]:.0f} ms vs {1000*ref[k][1]:.0f} ms")


if __name__ == "__main__":
    main()
