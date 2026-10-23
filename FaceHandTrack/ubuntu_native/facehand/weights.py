"""The published ONNX files of YuNet and the MediaPipe hand models, loaded as
ordinary PyTorch modules (onnx2torch). The weights are used exactly as published
(OpenCV Zoo); only a few shape-computing nodes are swapped for fixed PyTorch ops,
so the modules run on any device (CPU or CUDA) and at any input size.

    YuNet (MIT)                      image (1,3,H,W) BGR 0..255 -> faces (1,N,15) decoded
    MediaPipe palm detector          image (1,192,192,3) RGB 0..1 -> boxes (1,2016,18), scores (1,2016,1)
    MediaPipe hand landmarks         image (1,224,224,3) RGB 0..1 -> landmarks (1,63), presence (1,1),
      (Google, Apache-2.0)                                            right-hand score (1,1), world (1,63)
"""
import numpy as np
import onnx
import onnx.numpy_helper
import torch
import torch.nn.functional as F
from onnx2torch import convert

STRIDES = (8, 16, 32)


class _Resize(torch.nn.Module):
    """Fixed-size (or fixed 2x) resize in place of onnx2torch's OnnxResize, whose
    shape arguments are tensors."""

    def __init__(self, mode: str, size=None):
        super().__init__()
        self.mode, self.size = mode, size

    def forward(self, x, *unused):
        if self.mode == "nearest":
            return F.interpolate(x, scale_factor=2.0, mode="nearest")
        return F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)


class _Reshape(torch.nn.Module):
    """Replaces OnnxReshape; these models only reshape to [1, -1, C] with constant C."""

    def __init__(self, last_dim: int):
        super().__init__()
        self.last_dim = last_dim

    def forward(self, x, *unused):
        return x.reshape(1, -1, self.last_dim)


class _ChannelPad(torch.nn.Module):
    """Replaces OnnxPadDynamic: zero channels appended at the end (the palm model's 'channel_padding')."""

    def __init__(self, extra: int):
        super().__init__()
        self.extra = extra

    def forward(self, x, *unused):
        return F.pad(x, (0, 0, 0, 0, 0, self.extra))


class _PReLU(torch.nn.Module):
    """Replaces OnnxPReLU: the palm model's (1, C, 1, 1) slopes send onnx2torch down a
    boolean-mask path (nonzero / scatter), which is slow everywhere and CPU-only in Core ML."""

    def forward(self, x, slope):
        return F.prelu(x, slope.reshape(-1))


def load_onnx_as_torch(onnx_path: str) -> torch.nn.Module:
    m = onnx.load(onnx_path)
    consts = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
    swap = {}
    for n in m.graph.node:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        if n.op_type == "Reshape":
            shape = consts[n.input[1]]
            assert len(shape) == 3 and shape[0] == 1 and shape[1] == -1, shape
            swap[n.name] = _Reshape(int(shape[2]))
        elif n.op_type == "Resize":
            mode = attrs.get("mode", b"nearest").decode()
            if mode == "nearest":  # YuNet: constant 2x upsampling
                swap[n.name] = _Resize("nearest")
            else:  # palm detector: bilinear, half-pixel, to a constant size
                assert attrs.get("coordinate_transformation_mode") == b"half_pixel", attrs
                swap[n.name] = _Resize("bilinear", tuple(int(v) for v in consts[n.input[3]][2:]))
        elif n.op_type == "PRelu":
            swap[n.name] = _PReLU()
        elif n.op_type == "Pad" and len(n.input) > 1 and n.input[1] in consts:
            pads = consts[n.input[1]]  # NCHW begin..., end...
            if len(pads) == 8 and not pads[[0, 1, 2, 3, 4, 6, 7]].any():
                swap[n.name] = _ChannelPad(int(pads[5]))
    gm = convert(m).eval()
    for name, _ in list(gm.named_children()):
        if name in swap:
            setattr(gm, name, swap[name])
    return gm


class YuNetDecoded(torch.nn.Module):
    """YuNet at any input size (multiple of 32), with the box decoding folded in:
    per anchor [score, x1, y1, w, h, (lx, ly) x 5] in input pixels."""

    def __init__(self, onnx_path: str, width: int, height: int):
        super().__init__()
        assert width % 32 == 0 and height % 32 == 0, "YuNet needs multiples of 32"
        self.net = load_onnx_as_torch(onnx_path)
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
