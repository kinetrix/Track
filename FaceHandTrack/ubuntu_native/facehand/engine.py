"""The engine: one call per camera frame, returns faces, hands and gestures.

    engine = Engine("models", "third_party/EdgeTAM/checkpoints/edgetam.pt")
    result = engine.process(bgr_frame, t_seconds)   # FrameResult

Same pipeline as the ONNX version, on PyTorch (CUDA GPU when there is one):

    face path                                   hand path
    ---------                                   ---------
    YuNet (every 3rd frame,  || EdgeTAM         follow tracked hands from their
    every frame if no faces)  || image encoder  last landmarks (landmark model)
           \\                    /               palm detector for new hands
            track manager: per face                (every 5th frame, every
            memory attention + decoder              frame if no hands)
            + memory encoder                     gesture rules + smoothing

The two paths run in parallel threads (PyTorch releases Python's GIL inside
its kernels), and inside the face path YuNet runs in parallel with the encoder.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch

from .backend import pick_device
from .faces import EdgeTAM, FaceDetector, FacePolicy, FaceTracker
from .gestures import GestureSmoother, classify
from .hands import HandLandmarker, HandPolicy, HandTracker, PalmDetector
from .types import FrameResult

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CKPT = os.path.join(HERE, "third_party", "EdgeTAM", "checkpoints", "edgetam.pt")


class Engine:
    def __init__(self, model_dir: str, edgetam_ckpt: str = DEFAULT_CKPT, accel: str = "auto", faces: bool = True,
                 hands: bool = True, mirrored_input: bool = True, face_policy: Optional[FacePolicy] = None,
                 hand_policy: Optional[HandPolicy] = None, verbose: bool = True):
        """accel: auto (CUDA GPU if available, else CPU) | cuda | cpu.
        mirrored_input: frames are mirrored like a selfie view (affects Left/Right only)."""
        with open(os.path.join(model_dir, "engine_config.json")) as f:
            self.cfg = json.load(f)
        self.device, self.device_name = pick_device(accel)
        if verbose:
            print(f"[torch] PyTorch {torch.__version__} on {self.device_name}")
        self.enable_faces, self.enable_hands = faces, hands
        p = lambda n: os.path.join(model_dir, n)  # noqa: E731
        if faces:
            if not os.path.exists(edgetam_ckpt):
                raise SystemExit(f"EdgeTAM checkpoint not found at {edgetam_ckpt}. Run ./setup.sh first.")
            c = self.cfg
            self.detector = FaceDetector(p(c["yunet"]), c["det_width"], c["det_height"], self.device)
            self.tam = EdgeTAM(edgetam_ckpt, c, self.device)
            self.faces = FaceTracker(self.tam, face_policy or FacePolicy())
        if hands:
            palm = PalmDetector(p("palm_detection_mediapipe_2023feb.onnx"), self.device)
            lmk = HandLandmarker(p("handpose_estimation_mediapipe_2023feb.onnx"), self.device)
            self.hands = HandTracker(palm, lmk, hand_policy or HandPolicy(), mirrored_input)
            self.smoothers = {}
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.index = 0

    @property
    def backends(self):
        """Where the models run, e.g. {'all models': 'cuda (NVIDIA GeForce RTX 4060 Laptop GPU)'}."""
        return {"all models": self.device_name}

    def reset(self):
        if self.enable_faces:
            self.faces.reset()
        if self.enable_hands:
            self.hands.reset()
            self.smoothers = {}

    # ---- face path ---------------------------------------------------------
    def _face_path(self, bgr, now, events):
        timings = {}
        H, W = bgr.shape[:2]
        pol = self.faces.pol
        self.tam.timings = {}
        det_future = None
        if self.faces.wants_detection(self.index):
            def det():
                t0 = time.perf_counter()
                d = self.detector(bgr, pol.det_score, pol.nms_iou, pol.min_face_px)
                timings["detect"] = (time.perf_counter() - t0) * 1000
                return d
            det_future = self.pool.submit(det)
        enc = self.tam.encode(bgr) if self.faces.tracks else None  # in parallel with YuNet
        dets = det_future.result() if det_future else []
        if enc is None and dets:
            enc = self.tam.encode(bgr)  # faces just appeared: encode now
        if enc is not None:
            self.faces.update(enc, dets, W, H, now, events)
        timings.update(self.tam.timings)
        return self.faces.results(), timings

    # ---- hand path ---------------------------------------------------------
    def _hand_path(self, bgr, events):
        self.hands.timings = {}
        self.hands.update(bgr, self.index, events)
        t0 = time.perf_counter()
        out = self.hands.results()
        worlds = {t.id: t.world for t in self.hands.tracks}
        for h in out:
            h.raw_gesture, h.fingers = classify(h.landmarks, worlds[h.id])
            sm = self.smoothers.setdefault(h.id, GestureSmoother())
            h.gesture = sm(h.raw_gesture)
        live = {h.id for h in out}
        self.smoothers = {k: v for k, v in self.smoothers.items() if k in live}
        timings = dict(self.hands.timings)
        timings["gesture"] = (time.perf_counter() - t0) * 1000
        return out, timings

    def process(self, bgr: np.ndarray, now: float) -> FrameResult:
        """bgr: HxWx3 uint8 frame. now: time in seconds (track timeouts are in seconds,
        so they behave the same at any frame rate)."""
        t0 = time.perf_counter()
        res = FrameResult(self.index, now)
        face_events, hand_events = [], []
        hand_future = self.pool.submit(self._hand_path, bgr, hand_events) if self.enable_hands else None
        if self.enable_faces:
            res.faces, ft = self._face_path(bgr, now, face_events)
            res.timings.update(ft)
        if hand_future:
            res.hands, ht = hand_future.result()
            res.timings.update(ht)
        res.events = face_events + hand_events
        res.timings["total"] = (time.perf_counter() - t0) * 1000
        self.index += 1
        return res

    def close(self):
        self.pool.shutdown(wait=False)
