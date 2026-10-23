#!/usr/bin/env python3
"""Convert the published YuNet and MediaPipe hand models to Core ML for the Mac app.

    python convert_models.py --models ../../onnx/models --out ../FaceHandTrack/Models

Produces (ML Programs, fp16, macOS 14+):
    YuNet.mlpackage          image (BGR, 640x384, 0..255)   -> faces (1, 5040, 15) decoded boxes
    PalmDetector.mlpackage   image (RGB, 192x192, 0..1)     -> boxes (1, 2016, 18), scores (1, 2016, 1)
    HandLandmarks.mlpackage  image (RGB, 224x224, 0..1)     -> landmarks (1, 63), presence (1, 1),
                                                               handedness (1, 1), world (1, 63)
Images are CVPixelBuffer inputs, so the app hands camera-format (BGRA) buffers
straight to Core ML. The four EdgeTAM models are the v1 ones (512 x 512), made
by ../../../FaceTrack_v1_iOS/conversion/export_edgetam.py.

Needs: torch, onnx, onnx2torch, coremltools (any OS; predictions only run on macOS).
"""
import argparse
import os

import coremltools as ct
import torch

import ct_compat  # noqa: F401  (NumPy 2 fix, must precede conversion)
from onnx_weights import YuNetDecoded, load_onnx_as_torch


class NCHWtoNHWC(torch.nn.Module):
    """Core ML image inputs are NCHW; the MediaPipe graphs expect NHWC."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x.permute(0, 2, 3, 1))


def convert(module, example, image, outputs, out_path, description, meta=None):
    with torch.no_grad():
        traced = torch.jit.trace(module.eval(), example)
    m = ct.convert(traced, inputs=[image], outputs=[ct.TensorType(name=n) for n in outputs],
                   convert_to="mlprogram", minimum_deployment_target=ct.target.macOS14,
                   compute_precision=ct.precision.FLOAT16)
    m.short_description = description
    for k, v in (meta or {}).items():
        m.user_defined_metadata[k] = str(v)
    m.save(out_path)
    print("saved", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join("..", "..", "onnx", "models"), help="folder with the ONNX files")
    ap.add_argument("--out", default=os.path.join("..", "FaceHandTrack", "Models"))
    ap.add_argument("--det-width", type=int, default=640)
    ap.add_argument("--det-height", type=int, default=384)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    src = lambda n: os.path.join(args.models, n)  # noqa: E731
    dst = lambda n: os.path.join(args.out, n)  # noqa: E731

    W, H = args.det_width, args.det_height
    convert(YuNetDecoded(src("face_detection_yunet_2023mar.onnx"), W, H), torch.rand(1, 3, H, W) * 255,
            ct.ImageType(name="image", shape=(1, 3, H, W), scale=1.0, color_layout=ct.colorlayout.BGR),
            ["faces"], dst("YuNet.mlpackage"), "YuNet face detector (OpenCV Zoo 2023mar, MIT), decoded boxes",
            {"input_width": W, "input_height": H})

    convert(NCHWtoNHWC(load_onnx_as_torch(src("palm_detection_mediapipe_2023feb.onnx"))), torch.rand(1, 3, 192, 192),
            ct.ImageType(name="image", shape=(1, 3, 192, 192), scale=1 / 255.0, color_layout=ct.colorlayout.RGB),
            ["boxes", "scores"], dst("PalmDetector.mlpackage"),
            "MediaPipe palm detector (Google, Apache-2.0, via OpenCV Zoo 2023feb)")

    convert(NCHWtoNHWC(load_onnx_as_torch(src("handpose_estimation_mediapipe_2023feb.onnx"))),
            torch.rand(1, 3, 224, 224),
            ct.ImageType(name="image", shape=(1, 3, 224, 224), scale=1 / 255.0, color_layout=ct.colorlayout.RGB),
            ["landmarks", "presence", "handedness", "world"], dst("HandLandmarks.mlpackage"),
            "MediaPipe hand landmarks (Google, Apache-2.0, via OpenCV Zoo 2023feb)")


if __name__ == "__main__":
    main()
