#!/usr/bin/env python3
"""Convert OpenCV-Zoo YuNet (2023mar, MIT) to a Core ML ML Program with the
box decoding folded into the graph.

Output model
    input  : "image"  CVPixelBuffer, BGR, 0..255, size W x H (default 384 x 640, portrait)
    output : "faces"  Float32/16 [1, N, 15]
             per anchor: score, x1, y1, w, h, (lx, ly) x 5 landmarks, in input-pixel units.
             N = sum over strides 8/16/32 of (H/s)*(W/s).

Only thresholding + NMS is left to the app (a few hundred candidates, microseconds).

Usage
    python convert_yunet.py --onnx face_detection_yunet_2023mar.onnx \
        --width 384 --height 640 --out ../ios/FaceTrack/Models/YuNet.mlpackage
"""
import argparse

import coremltools as ct

import ct_compat  # noqa: F401  (NumPy 2 fix, must precede conversion)
import numpy as np
import onnx
import onnx.numpy_helper
import torch
from onnx2torch import convert

STRIDES = (8, 16, 32)


class _Upsample2x(torch.nn.Module):
    """Replaces onnx2torch's OnnxResize (nearest, scale 2): its int casts of the
    shape tensor don't convert to Core ML, and YuNet's scales are constant."""

    def forward(self, x, *unused):
        return torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")


class _Reshape(torch.nn.Module):
    """Replaces OnnxReshape. YuNet only reshapes to [1, -1, C] with constant C."""

    def __init__(self, last_dim: int):
        super().__init__()
        self.last_dim = last_dim

    def forward(self, x, *unused):
        return x.reshape(1, -1, self.last_dim)  # batch is fixed to 1


def _load_traceable(onnx_path: str) -> torch.nn.Module:
    m = onnx.load(onnx_path)
    consts = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
    reshape_dims = {}
    for n in m.graph.node:
        if n.op_type == "Reshape":
            shape = consts[n.input[1]]
            assert len(shape) == 3 and shape[0] == 1 and shape[1] == -1, shape
            reshape_dims[n.name] = int(shape[2])
    gm = convert(m).eval()
    for name, mod in list(gm.named_children()):
        t = type(mod).__name__
        if t == "OnnxResize":
            setattr(gm, name, _Upsample2x())
        elif t == "OnnxReshape":
            setattr(gm, name, _Reshape(reshape_dims[name]))
    return gm


class YuNetDecoded(torch.nn.Module):
    """Raw YuNet heads -> decoded [score, x1, y1, w, h, 10 landmark coords]."""

    def __init__(self, onnx_path: str, width: int, height: int):
        super().__init__()
        assert width % 32 == 0 and height % 32 == 0, "YuNet needs multiples of 32"
        self.net = _load_traceable(onnx_path)
        grids, strides = [], []
        for s in STRIDES:
            rows, cols = height // s, width // s
            r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
            grids.append(np.stack([c.ravel(), r.ravel()], -1))  # (x=col, y=row)
            strides.append(np.full((rows * cols, 1), s))
        self.register_buffer("grid", torch.tensor(np.concatenate(grids), dtype=torch.float32)[None])
        self.register_buffer("stride", torch.tensor(np.concatenate(strides), dtype=torch.float32)[None])

    def forward(self, x):
        o = self.net(x)
        # ONNX output order: cls_8,16,32, obj_8,16,32, bbox_8,16,32, kps_8,16,32
        cls = torch.cat(o[0:3], 1)
        obj = torch.cat(o[3:6], 1)
        box = torch.cat(o[6:9], 1)
        kps = torch.cat(o[9:12], 1)
        score = torch.sqrt(cls.clamp(0, 1) * obj.clamp(0, 1))
        s = self.stride
        cxcy = (box[..., 0:2] + self.grid) * s
        wh = torch.exp(box[..., 2:4]) * s
        xy = cxcy - wh * 0.5
        k = kps.reshape(1, -1, 5, 2)
        k = (k + self.grid[:, :, None, :]) * s[:, :, None, :]
        return torch.cat([score, xy, wh, k.flatten(2)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="face_detection_yunet_2023mar.onnx")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--out", default="../ios/FaceTrack/Models/YuNet.mlpackage")
    args = ap.parse_args()

    model = YuNetDecoded(args.onnx, args.width, args.height).eval()
    example = torch.rand(1, 3, args.height, args.width) * 255
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=example.shape, scale=1.0,
                             color_layout=ct.colorlayout.BGR)],
        outputs=[ct.TensorType(name="faces")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.short_description = "YuNet face detector (OpenCV Zoo 2023mar, MIT) with in-graph decoding"
    mlmodel.user_defined_metadata["input_width"] = str(args.width)
    mlmodel.user_defined_metadata["input_height"] = str(args.height)
    mlmodel.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
