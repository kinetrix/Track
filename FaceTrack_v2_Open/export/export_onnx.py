#!/usr/bin/env python3
"""Export every model the engine uses to ONNX (opset 17) into ../models.

    python export_onnx.py                      # defaults below
    python export_onnx.py --image-size 512 --det-width 640 --det-height 384

Produces
    yunet_{W}x{H}.onnx           image(1,3,H,W BGR 0..255) -> faces(1,N,15)
    edgetam_image_encoder.onnx   image(1,3,S,S RGB 0..255) -> feat, feat_init, tgt0, hr0, hr1
    edgetam_memory_attention.onnx tgt0, memory(1,K,64), slot_valid(1,C+R+P) -> pix_feat
    edgetam_mask_decoder.onnx    pix_feat, hr0, hr1, point_coords(1,N,2), point_labels(1,N)
                                 -> masks(1,4,S/4,S/4), iou(1,4), ptr(1,4,256), obj_score(1,1)
                                 N = 2 (box prompt) or 1 (tracking, padding only)
    edgetam_memory_encoder.onnx  feat, mask_logits(1,1,S/4,S/4), binarize(1) -> tokens(1,512,64)
    engine_config.json           sizes and memory-slot layout the engine follows

The hand models (palm detector, hand landmarks) are already ONNX in OpenCV Zoo and
are downloaded as is by get_models.sh.
"""
import argparse
import json
import os

import torch

from edgetam_modules import (TOKENS_PER_MEM, TOKENS_PER_PTR, EdgeTAMImageEncoder, EdgeTAMMaskDecoder,
                             EdgeTAMMemoryAttention, EdgeTAMMemoryEncoder, load_edgetam)
from yunet_modules import YuNetDecoded

HERE = os.path.dirname(os.path.abspath(__file__))


def export(module, args, path, inputs, outputs, dynamic=None):
    torch.onnx.export(module, args, path, input_names=inputs, output_names=outputs,
                      opset_version=17, dynamic_axes=dynamic, dynamo=False, do_constant_folding=True)
    print(f"  {os.path.basename(path)}  {os.path.getsize(path) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "EdgeTAM", "checkpoints", "edgetam.pt"))
    ap.add_argument("--yunet", default=os.path.join(HERE, "..", "models", "face_detection_yunet_2023mar.onnx"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "models"))
    ap.add_argument("--image-size", type=int, default=512, choices=[512, 768, 1024])
    ap.add_argument("--det-width", type=int, default=640, help="YuNet input width (landscape webcam)")
    ap.add_argument("--det-height", type=int, default=384)
    ap.add_argument("--num-cond", type=int, default=2)
    ap.add_argument("--num-recent", type=int, default=6)
    ap.add_argument("--num-ptr", type=int, default=16)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda n: os.path.join(args.out_dir, n)  # noqa: E731
    S, C, R, P = args.image_size, args.num_cond, args.num_recent, args.num_ptr
    W, H = args.det_width, args.det_height

    print("YuNet")
    yunet_name = f"yunet_{W}x{H}.onnx"
    export(YuNetDecoded(args.yunet, W, H).eval(), (torch.rand(1, 3, H, W) * 255,), out(yunet_name), ["image"], ["faces"])

    print(f"EdgeTAM at {S}x{S}")
    sam = load_edgetam(args.ckpt, image_size=S)
    enc = EdgeTAMImageEncoder(sam).eval()
    att = EdgeTAMMemoryAttention(sam, C, R, P).eval()
    dec = EdgeTAMMaskDecoder(sam).eval()
    men = EdgeTAMMemoryEncoder(sam).eval()

    img = torch.rand(1, 3, S, S) * 255
    feat, feat_init, tgt0, hr0, hr1 = enc(img)
    export(enc, (img,), out("edgetam_image_encoder.onnx"), ["image"], ["feat", "feat_init", "tgt0", "hr0", "hr1"])

    K = (C + R) * TOKENS_PER_MEM + P * TOKENS_PER_PTR
    mem, valid = torch.randn(1, K, 64), torch.ones(1, C + R + P)
    export(att, (tgt0, mem, valid), out("edgetam_memory_attention.onnx"), ["tgt0", "memory", "slot_valid"], ["pix_feat"])

    coords = torch.tensor([[[100.0, 120.0], [300.0, 360.0]]])
    labels = torch.tensor([[2.0, 3.0]])
    export(dec, (feat_init, hr0, hr1, coords, labels), out("edgetam_mask_decoder.onnx"),
           ["pix_feat", "hr0", "hr1", "point_coords", "point_labels"], ["masks", "iou", "ptr", "obj_score"],
           dynamic={"point_coords": {1: "n"}, "point_labels": {1: "n"}})

    m = torch.randn(1, 1, S // 4, S // 4)
    export(men, (feat, m, torch.ones(1)), out("edgetam_memory_encoder.onnx"), ["feat", "mask_logits", "binarize"], ["tokens"])

    cfg = dict(image_size=S, mask_size=S // 4, num_cond=C, num_recent=R, num_ptr=P,
               tokens_per_mem=TOKENS_PER_MEM, tokens_per_ptr=TOKENS_PER_PTR, mem_dim=64,
               stability_delta=0.05, stability_thresh=0.98,
               yunet=yunet_name, det_width=W, det_height=H)
    with open(out("engine_config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    print("  engine_config.json")


if __name__ == "__main__":
    main()
