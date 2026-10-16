#!/usr/bin/env python3
"""Export EdgeTAM as a *streaming video tracker* for Core ML (iOS 18+).

    python export_edgetam.py --ckpt EdgeTAM/checkpoints/edgetam.pt \
        --num-cond 2 --num-recent 6 --num-ptr 16 --out-dir ../ios/FaceTrack/Models

Produces
    EdgeTAMImageEncoder.mlpackage      image -> feat, feat_init, tgt0, hr0, hr1
    EdgeTAMMemoryAttention.mlpackage   tgt0, memory, slot_valid -> pix_feat
    EdgeTAMMaskDecoder.mlpackage       two functions sharing one weight file:
                                         "track"  (1 point  = padding only)
                                         "prompt" (2 points = box corners)
    EdgeTAMMemoryEncoder.mlpackage     feat, mask_logits, binarize -> memory tokens
    edgetam_config.json                slot layout the app must follow

Input size (--image-size): 1024 is the paper setting; 768 and 512 run the same
checkpoint without retraining and cut compute ~1.8x / ~4x (feature grid S/16,
mask grid S/4). The app reads the size from edgetam_config.json.

Memory-bank presets (quality vs. speed, per tracked face):
    --num-cond 2 --num-recent 6   (default) 4096 spatial + 64 pointer keys
    --num-cond 1 --num-recent 6   exactly the SAM 2 / EdgeTAM paper setting
    --num-cond 1 --num-recent 3   "fast": 2048 + 64 keys, ~half the cross-attention
"""
import argparse
import json
import os
import shutil
import tempfile

import coremltools as ct
import numpy as np

import ct_compat  # noqa: F401  (NumPy 2 fix, must precede conversion)
import torch

from edgetam_coreml_modules import (TOKENS_PER_MEM, TOKENS_PER_PTR, EdgeTAMImageEncoder,
                                    EdgeTAMMaskDecoder, EdgeTAMMemoryAttention, EdgeTAMMemoryEncoder,
                                    load_edgetam)

TARGET = ct.target.iOS18
F16 = np.float16  # tensors passed between graphs stay fp16 end to end: no fp32<->fp16 copies in the app


def T(name, shape=None, dtype=F16):
    return ct.TensorType(name=name, shape=shape, dtype=dtype)


def _convert(module, example, inputs, outputs, desc, precision, compute_units=ct.ComputeUnit.ALL):
    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, example)
    m = ct.convert(traced, inputs=inputs, outputs=outputs, convert_to="mlprogram",
                   minimum_deployment_target=TARGET, compute_precision=precision,
                   compute_units=compute_units)
    m.short_description = desc
    m.author = "EdgeTAM (Meta, Apache-2.0); streaming export for FaceTrack"
    return m


def _maybe_quantize(m, mode):
    if mode == "none":
        return m
    import coremltools.optimize as cto
    if mode == "int8":
        cfg = cto.coreml.OptimizationConfig(global_config=cto.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel"))
        return cto.coreml.linear_quantize_weights(m, cfg)
    if mode == "palette6":
        cfg = cto.coreml.OptimizationConfig(global_config=cto.coreml.OpPalettizerConfig(
            nbits=6, mode="kmeans", granularity="per_grouped_channel", group_size=16))
        return cto.coreml.palettize_weights(m, cfg)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="EdgeTAM/checkpoints/edgetam.pt")
    ap.add_argument("--image-size", type=int, default=512, choices=[512, 768, 1024],
                    help="EdgeTAM input size (same checkpoint at every size; 1024 = paper setting)")
    ap.add_argument("--num-cond", type=int, default=2)
    ap.add_argument("--num-recent", type=int, default=6)
    ap.add_argument("--num-ptr", type=int, default=16)
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--weights", choices=["none", "int8", "palette6"], default="none",
                    help="weight compression (smaller download; int8 can also speed up the ANE)")
    ap.add_argument("--attn-chunk", type=int, default=512,
                    help="queries per attention chunk (0 = unchunked); keeps score blocks in ANE memory")
    ap.add_argument("--out-dir", default="../ios/FaceTrack/Models")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    prec = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    torch.set_grad_enabled(False)

    import edgetam_coreml_modules as mods
    mods.ATTN_Q_CHUNK = args.attn_chunk
    IMG = args.image_size
    Fg, Mg = IMG // 16, IMG // 4  # feature grid, mask grid
    sam = load_edgetam(args.ckpt, image_size=IMG)
    C, R, P = args.num_cond, args.num_recent, args.num_ptr
    S = C + R
    Lk = S * TOKENS_PER_MEM + P * TOKENS_PER_PTR

    def save(m, name):
        m = _maybe_quantize(m, args.weights)
        path = os.path.join(args.out_dir, name)
        if os.path.exists(path):
            shutil.rmtree(path)
        m.save(path)
        print("saved", path)

    # 1) image encoder --------------------------------------------------------
    enc = EdgeTAMImageEncoder(sam)
    m = _convert(enc, torch.rand(1, 3, IMG, IMG) * 255,
                 [ct.ImageType(name="image", shape=(1, 3, IMG, IMG), scale=1.0,
                               color_layout=ct.colorlayout.RGB)],
                 [T(n) for n in ["feat", "feat_init", "tgt0", "hr0", "hr1"]],
                 "EdgeTAM RepViT-M1 encoder + FPN + shared layer-0 self-attention", prec)
    save(m, "EdgeTAMImageEncoder.mlpackage")

    # 2) memory attention -----------------------------------------------------
    attn = EdgeTAMMemoryAttention(sam, C, R, P)
    ex = (torch.randn(1, Fg * Fg, 256), torch.randn(1, Lk, 64), torch.ones(1, S + P))
    m = _convert(attn, ex,
                 [T("tgt0", ex[0].shape), T("memory", ex[1].shape), T("slot_valid", ex[2].shape)],
                 [T("pix_feat")],
                 f"EdgeTAM memory attention, {C} cond + {R} recent slots, {P} object pointers", prec)
    save(m, "EdgeTAMMemoryAttention.mlpackage")

    # 3) mask decoder: two functions (N=1 tracking, N=2 box prompt) in one package
    dec = EdgeTAMMaskDecoder(sam)
    tmp = tempfile.mkdtemp()
    paths = {}
    for fn, n in [("track", 1), ("prompt", 2)]:
        ex = (torch.randn(1, 256, Fg, Fg), torch.randn(1, 32, Mg, Mg), torch.randn(1, 64, Mg // 2, Mg // 2),
              torch.rand(1, n, 2) * IMG, torch.full((1, n), 2.0) if n == 2 else -torch.ones(1, 1))
        m = _convert(dec, ex,
                     [T("pix_feat", ex[0].shape), T("hr0", ex[1].shape), T("hr1", ex[2].shape),
                      T("point_coords", ex[3].shape, np.float32),  # pixel coords up to IMG: keep fp32
                      T("point_labels", ex[4].shape, np.float32)],
                     [T(x) for x in ["masks", "iou", "obj_ptr", "obj_score"]],
                     "EdgeTAM prompt encoder + mask decoder + object-pointer head", prec)
        m = _maybe_quantize(m, args.weights)
        paths[fn] = os.path.join(tmp, f"{fn}.mlpackage")
        m.save(paths[fn])
    desc = ct.utils.MultiFunctionDescriptor()
    desc.add_function(paths["track"], src_function_name="main", target_function_name="track")
    desc.add_function(paths["prompt"], src_function_name="main", target_function_name="prompt")
    desc.default_function_name = "track"
    out = os.path.join(args.out_dir, "EdgeTAMMaskDecoder.mlpackage")
    if os.path.exists(out):
        shutil.rmtree(out)
    ct.utils.save_multifunction(desc, out)
    shutil.rmtree(tmp)
    print("saved", out)

    # 4) memory encoder -------------------------------------------------------
    menc = EdgeTAMMemoryEncoder(sam)
    ex = (torch.randn(1, 256, Fg, Fg), torch.randn(1, 1, Mg, Mg) * 5, torch.zeros(1))
    m = _convert(menc, ex,
                 [T("feat", ex[0].shape), T("mask_logits", ex[1].shape), T("binarize", ex[2].shape)],
                 [T("memory_tokens")],
                 "EdgeTAM memory encoder + 2D spatial perceiver", prec)
    save(m, "EdgeTAMMemoryEncoder.mlpackage")

    cfg = {"num_cond": C, "num_recent": R, "num_ptr": P, "attn_chunk": args.attn_chunk, "tokens_per_mem": TOKENS_PER_MEM,
           "tokens_per_ptr": TOKENS_PER_PTR, "mem_dim": 64, "image_size": IMG, "mask_size": Mg,
           "stability_delta": 0.05, "stability_thresh": 0.98}
    with open(os.path.join(args.out_dir, "edgetam_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(json.dumps(cfg))


if __name__ == "__main__":
    main()
