#!/usr/bin/env python3
"""Check each exported ONNX graph against the PyTorch module it came from
(same inputs, max abs difference), including the decoder with 1 and 2 points."""
import json
import os

import numpy as np
import onnxruntime as ort
import torch

from edgetam_modules import (EdgeTAMImageEncoder, EdgeTAMMaskDecoder, EdgeTAMMemoryAttention,
                             EdgeTAMMemoryEncoder, load_edgetam)
from yunet_modules import YuNetDecoded

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "..", "models")


def run(name, feeds):
    s = ort.InferenceSession(os.path.join(M, name), providers=["CPUExecutionProvider"])
    return s.run(None, {k: v.numpy() if torch.is_tensor(v) else v for k, v in feeds.items()})


def report(tag, ref, got):
    for i, (a, b) in enumerate(zip(ref, got)):
        a = a.numpy()
        d = np.abs(a - b).max()
        print(f"{tag}[{i}] {tuple(a.shape)} max|diff| {d:.2e}  (max|ref| {np.abs(a).max():.1e})")


def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    cfg = json.load(open(os.path.join(M, "engine_config.json")))
    S, C, R, P = cfg["image_size"], cfg["num_cond"], cfg["num_recent"], cfg["num_ptr"]
    W, H = cfg["det_width"], cfg["det_height"]

    x = torch.rand(1, 3, H, W) * 255
    report("yunet", [YuNetDecoded(os.path.join(M, "face_detection_yunet_2023mar.onnx"), W, H).eval()(x)],
           run(cfg["yunet"], {"image": x}))

    sam = load_edgetam(os.path.join(HERE, "EdgeTAM", "checkpoints", "edgetam.pt"), image_size=S)
    enc, att = EdgeTAMImageEncoder(sam).eval(), EdgeTAMMemoryAttention(sam, C, R, P).eval()
    dec, men = EdgeTAMMaskDecoder(sam).eval(), EdgeTAMMemoryEncoder(sam).eval()
    img = torch.rand(1, 3, S, S) * 255
    e = enc(img)
    report("encoder", e, run("edgetam_image_encoder.onnx", {"image": img}))
    feat, feat_init, tgt0, hr0, hr1 = e

    tokens = men(feat, torch.randn(1, 1, S // 4, S // 4) * 4, torch.ones(1))
    K = (C + R) * 512 + P * 4
    memory = torch.cat([tokens] * (C + R) + [torch.randn(1, P * 4, 64)], 1)
    valid = torch.tensor([[1, 0] + [1, 1, 0, 0, 0, 0] + [1] * 5 + [0] * (P - 5)], dtype=torch.float32)
    assert memory.shape[1] == K
    pix = att(tgt0, memory, valid)
    report("mem-attn", [pix], run("edgetam_memory_attention.onnx", {"tgt0": tgt0, "memory": memory, "slot_valid": valid}))

    for coords, labels in ((torch.tensor([[[100.0, 120.0], [300.0, 360.0]]]), torch.tensor([[2.0, 3.0]])),
                           (torch.zeros(1, 1, 2), -torch.ones(1, 1))):
        ref = dec(pix, hr0, hr1, coords, labels)
        report(f"decoder n={coords.shape[1]}", ref, run("edgetam_mask_decoder.onnx", dict(
            pix_feat=pix, hr0=hr0, hr1=hr1, point_coords=coords, point_labels=labels)))

    m = ref[0][:, 1:2]
    for b in (1.0, 0.0):
        report(f"mem-enc bin={b:.0f}", [men(feat, m, torch.tensor([b]))],
               run("edgetam_memory_encoder.onnx", {"feat": feat, "mask_logits": m, "binarize": torch.tensor([b])}))


if __name__ == "__main__":
    main()
