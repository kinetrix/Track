#!/usr/bin/env python3
"""Check that the re-packaged streaming EdgeTAM (edgetam_coreml_modules +
reference_tracker) reproduces Meta's official SAM2VideoPredictor on a clip.

    python validate_edgetam.py --frames /path/to/jpg_folder --box x1 y1 x2 y2

Runs both trackers from the same box on frame 0 and reports per-frame mask IoU
and max |logit| difference. With the default config (1 cond slot, 6 recent
slots, 16 pointers) the two are the same algorithm, so IoU should be ~1.0.
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

from edgetam_coreml_modules import (EdgeTAMImageEncoder, EdgeTAMMaskDecoder, EdgeTAMMemoryAttention,
                                    EdgeTAMMemoryEncoder, load_edgetam)
from reference_tracker import ReferenceTracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="EdgeTAM/checkpoints/edgetam.pt")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--box", type=float, nargs=4, required=True, help="x1 y1 x2 y2 in video pixels")
    ap.add_argument("--max-frames", type=int, default=48)
    ap.add_argument("--image-size", type=int, default=1024, help="EdgeTAM input size (1024, 768 or 512)")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))[: args.max_frames]
    W, H = Image.open(files[0]).size

    # ---------------- official ----------------
    S = args.image_size
    pred = load_edgetam(args.ckpt, video_predictor=True, image_size=S)
    state = pred.init_state(video_path=args.frames, offload_video_to_cpu=True, offload_state_to_cpu=True)
    pred.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=np.array(args.box, np.float32))
    official = {}
    for fi, _, logits in pred.propagate_in_video(state, max_frame_num_to_track=len(files) - 1):
        official[fi] = state["output_dict_per_obj"][0]
    official_masks = {}
    for fi in range(len(files)):
        d = state["output_dict_per_obj"][0]
        out = d["cond_frame_outputs"].get(fi) or d["non_cond_frame_outputs"].get(fi)
        official_masks[fi] = out["pred_masks"][0, 0].float().numpy()  # (S/4)x(S/4) logits

    # ---------------- ours ----------------
    sam = load_edgetam(args.ckpt, image_size=S)
    trk = ReferenceTracker(EdgeTAMImageEncoder(sam).eval(), EdgeTAMMemoryAttention(sam, 1, 6, 16).eval(),
                           EdgeTAMMaskDecoder(sam).eval(), EdgeTAMMemoryEncoder(sam).eval(), 1, 6, 16)
    sx, sy = S / W, S / H
    box = [args.box[0] * sx, args.box[1] * sy, args.box[2] * sx, args.box[3] * sy]
    ious, diffs = [], []
    mem = None
    for fi, f in enumerate(files):
        img = np.array(Image.open(f).convert("RGB").resize((S, S)), dtype=np.float32)  # same resize as sam2 loader
        x = torch.from_numpy(img).permute(2, 0, 1)[None]
        enc = trk.encode(x)
        if fi == 0:
            mem, mask, _, _ = trk.add_box(enc, box)
        else:
            mask, _, _ = trk.track(enc, mem)
        ours = mask[0, 0].numpy()
        ref = official_masks[fi]
        a, b = ours > 0, ref > 0
        iou = (a & b).sum() / (a | b).sum() if (a | b).any() else 1.0  # both empty = agree
        valid = (np.abs(ref) < 1000) & (np.abs(ours) < 1000)
        ious.append(iou)
        diffs.append(float(np.abs(ours - ref)[valid].max()) if valid.any() else 0.0)
        print(f"frame {fi:3d}  IoU {iou:.4f}  max|dlogit| {diffs[-1]:.4f}  area {a.sum()}/{b.sum()}")
    print(f"\nmean IoU {np.mean(ious):.4f}  min IoU {np.min(ious):.4f}  worst logit diff {max(diffs):.4f}")


if __name__ == "__main__":
    main()
