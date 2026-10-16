#!/usr/bin/env python3
"""Run on a Mac (Core ML prediction needs macOS). Tracks a face through a clip
twice — once with the PyTorch reference, once with the exported .mlpackage
files driven by the *same* ReferenceTracker logic — and reports mask IoU per
frame plus mean latency of each Core ML graph.

    python verify_coreml_mac.py --frames clip/ --box x1 y1 x2 y2 \
        --models ../ios/FaceTrack/Models --compute-units ALL
"""
import argparse
import glob
import json
import os
import time
from collections import defaultdict

import coremltools as ct
import numpy as np
import torch
from PIL import Image

from edgetam_coreml_modules import (EdgeTAMImageEncoder, EdgeTAMMaskDecoder, EdgeTAMMemoryAttention,
                                    EdgeTAMMemoryEncoder, load_edgetam)
from reference_tracker import ReferenceTracker

TIMES = defaultdict(list)


def _h(x):  # graph-to-graph tensors are fp16 in the exported models
    return x.numpy().astype(np.float16)


def _t(x):
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


class _ML:
    def __init__(self, path, cu, fn=None):
        self.name = os.path.basename(path) + (f":{fn}" if fn else "")
        self.m = ct.models.MLModel(path, compute_units=cu, function_name=fn) if fn else ct.models.MLModel(path, compute_units=cu)

    def run(self, feeds):
        t0 = time.perf_counter()
        out = self.m.predict(feeds)
        TIMES[self.name].append(time.perf_counter() - t0)
        return out


class CMEncoder(_ML):
    def __call__(self, image):
        img = Image.fromarray(image[0].permute(1, 2, 0).numpy().astype(np.uint8))
        o = self.run({"image": img})
        return tuple(_t(o[k]) for k in ["feat", "feat_init", "tgt0", "hr0", "hr1"])


class CMAttention(_ML):
    def __call__(self, tgt0, memory, valid):
        return _t(self.run({"tgt0": _h(tgt0), "memory": _h(memory), "slot_valid": _h(valid)})["pix_feat"])


class CMDecoder:
    def __init__(self, path, cu):
        self.track = _ML(path, cu, "track")
        self.prompt = _ML(path, cu, "prompt")

    def __call__(self, pix, hr0, hr1, coords, labels):
        m = self.prompt if coords.shape[1] == 2 else self.track
        o = m.run({"pix_feat": _h(pix), "hr0": _h(hr0), "hr1": _h(hr1),
                   "point_coords": coords.numpy(), "point_labels": labels.numpy()})
        return tuple(_t(o[k]) for k in ["masks", "iou", "obj_ptr", "obj_score"])


class CMMemEncoder(_ML):
    def __call__(self, feat, mask, binarize):
        return _t(self.run({"feat": _h(feat), "mask_logits": _h(mask), "binarize": _h(binarize)})["memory_tokens"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="EdgeTAM/checkpoints/edgetam.pt")
    ap.add_argument("--models", default="../ios/FaceTrack/Models")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--box", type=float, nargs=4, required=True)
    ap.add_argument("--compute-units", default="ALL", choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"])
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    cfg = json.load(open(os.path.join(args.models, "edgetam_config.json")))
    C, R, P = cfg["num_cond"], cfg["num_recent"], cfg["num_ptr"]
    cu = getattr(ct.ComputeUnit, args.compute_units)

    S = cfg.get("image_size", 1024)
    sam = load_edgetam(args.ckpt, image_size=S)
    ref = ReferenceTracker(EdgeTAMImageEncoder(sam), EdgeTAMMemoryAttention(sam, C, R, P), EdgeTAMMaskDecoder(sam),
                           EdgeTAMMemoryEncoder(sam), C, R, P)
    p = lambda n: os.path.join(args.models, n)
    cml = ReferenceTracker(CMEncoder(p("EdgeTAMImageEncoder.mlpackage"), cu),
                           CMAttention(p("EdgeTAMMemoryAttention.mlpackage"), cu),
                           CMDecoder(p("EdgeTAMMaskDecoder.mlpackage"), cu),
                           CMMemEncoder(p("EdgeTAMMemoryEncoder.mlpackage"), cu), C, R, P)

    files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))
    W, H = Image.open(files[0]).size
    box = [args.box[0] * S / W, args.box[1] * S / H, args.box[2] * S / W, args.box[3] * S / H]
    mem_r = mem_c = None
    ious = []
    for i, f in enumerate(files):
        x = torch.from_numpy(np.array(Image.open(f).convert("RGB").resize((S, S)), np.float32)).permute(2, 0, 1)[None]
        er, ec = ref.encode(x), cml.encode(x)
        if i == 0:
            mem_r, mr, _, _ = ref.add_box(er, box)
            mem_c, mc, _, _ = cml.add_box(ec, box)
        else:
            mr, _, _ = ref.track(er, mem_r)
            mc, _, _ = cml.track(ec, mem_c)
        a, b = mr[0, 0].numpy() > 0, mc[0, 0].numpy() > 0
        iou = (a & b).sum() / (a | b).sum() if (a | b).any() else 1.0
        ious.append(iou)
        print(f"frame {i:3d}  IoU(torch, coreml) {iou:.4f}")
    print(f"\nmean IoU {np.mean(ious):.4f}  min {np.min(ious):.4f}")
    print("\nmean latency on this Mac (first call excluded):")
    for k, v in TIMES.items():
        v = v[1:] or v
        print(f"  {k:45s} {1000 * np.mean(v):7.1f} ms")


if __name__ == "__main__":
    main()
