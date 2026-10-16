"""The engine: one call per camera frame, returns faces, hands and gestures.

    engine = Engine("models")
    result = engine.process(bgr_frame, t_seconds)   # FrameResult

It knows nothing about cameras, windows or platforms. Any interface (the Mac
viewer in app/, a web server, a mobile app re-implementing it) feeds frames in
and draws FrameResult. Per frame:

    face path                                   hand path
    ---------                                   ---------
    YuNet (every 3rd frame,  || EdgeTAM         follow tracked hands from their
    every frame if no faces)  || image encoder  last landmarks (landmark model)
           \\                    /               palm detector for new hands
            track manager: per face                (every 5th frame, every
            memory attention + decoder              frame if no hands)
            + memory encoder                     gesture rules + smoothing

The two paths run in parallel threads (ONNX Runtime releases Python's GIL while
a model runs), and inside the face path YuNet runs in parallel with the encoder.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from .faces import EdgeTAM, FaceDetector, FacePolicy, FaceTracker
from .gestures import GestureSmoother, classify
from .hands import HandLandmarker, HandPolicy, HandTracker, PalmDetector
from .runtime import Runtime
from .types import FrameResult


class Engine:
    def __init__(self, model_dir: str, accel: str = "auto", faces: bool = True, hands: bool = True,
                 mirrored_input: bool = True, face_policy: Optional[FacePolicy] = None,
                 hand_policy: Optional[HandPolicy] = None, face_threads: int = 0, hand_threads: int = 2,
                 verbose: bool = True):
        """mirrored_input: frames are mirrored like a selfie view (affects Left/Right only).
        face_threads / hand_threads: CPU threads per model (0 = all cores)."""
        with open(os.path.join(model_dir, "engine_config.json")) as f:
            self.cfg = json.load(f)
        self.rt = Runtime(accel, verbose)
        self.enable_faces, self.enable_hands = faces, hands
        p = lambda n: os.path.join(model_dir, n)  # noqa: E731
        if faces:
            c = self.cfg
            self.detector = FaceDetector(self.rt, p(c["yunet"]), c["det_width"], c["det_height"], face_threads)
            self.tam = EdgeTAM(self.rt, model_dir, c, face_threads)
            self.faces = FaceTracker(self.tam, face_policy or FacePolicy())
        if hands:
            palm = PalmDetector(self.rt, p("palm_detection_mediapipe_2023feb.onnx"), hand_threads)
            lmk = HandLandmarker(self.rt, p("handpose_estimation_mediapipe_2023feb.onnx"), hand_threads)
            self.hands = HandTracker(palm, lmk, hand_policy or HandPolicy(), mirrored_input)
            self.smoothers = {}
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.index = 0

    @property
    def backends(self):
        """Which accelerator each model ended up on, e.g. {'yunet_640x384.onnx': 'coreml'}."""
        return dict(self.rt.chosen)

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
