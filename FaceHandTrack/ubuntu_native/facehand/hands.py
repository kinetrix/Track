"""Hand detection, 21-point hand landmarks and hand tracking.

Models: MediaPipe palm detector + hand landmark model (Google, Apache-2.0), as
published in OpenCV Zoo (palm_detection_mediapipe_2023feb,
handpose_estimation_mediapipe_2023feb), run as PyTorch modules on the chosen
device (weights.py). Identical to the ONNX version apart from the runtime.
Pre/post-processing is ported from OpenCV Zoo's mp_palmdet.py / mp_handpose.py
(Apache-2.0), with two changes:
  * anchors are generated instead of being a 2016-row table;
  * tracking: like MediaPipe, a tracked hand's next crop comes from its own
    landmarks of the previous frame, so the palm detector only runs to find
    *new* hands (every frame while no hand is tracked, every N frames otherwise).
"""
import time
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

import torch

from .backend import TorchModel
from .faces import iou, nms
from .types import Hand
from .weights import load_onnx_as_torch

# hand landmarks that correspond to the palm detector's 7 keypoints
PALM_FROM_HAND = [0, 5, 9, 13, 17, 1, 2]  # wrist, index/middle/ring/pinky MCP, thumb CMC, thumb MCP


def _palm_anchors():
    """SSD anchors of the 192x192 palm model: a 24x24 grid with 2 anchors per
    cell (stride 8) followed by a 12x12 grid with 6 per cell (stride 16 x 3 layers)."""
    a = []
    for g, n in ((24, 2), (12, 6)):
        for y in range(g):
            for x in range(g):
                a += [[(x + 0.5) / g, (y + 0.5) / g]] * n
    return np.array(a, np.float32)  # (2016, 2)


class PalmDetector:
    SIZE = 192

    def __init__(self, path: str, device: torch.device, score: float = 0.5, nms_iou: float = 0.3):
        self.net = TorchModel(load_onnx_as_torch(path), device)
        self.anchors = _palm_anchors()
        self.score, self.nms_iou = score, nms_iou

    def __call__(self, bgr: np.ndarray):
        """Returns a list of (palm, score); palm = 18 floats: box x1,y1,x2,y2 then 7 keypoints (x,y)."""
        H, W = bgr.shape[:2]
        S = self.SIZE
        ratio = min(S / H, S / W)
        nh, nw = int(H * ratio), int(W * ratio)
        img = cv2.resize(bgr, (nw, nh))
        left, top = (S - nw) // 2, (S - nh) // 2
        img = cv2.copyMakeBorder(img, top, S - nh - top, left, S - nw - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255.0  # NHWC, 0..1
        reg, logit = self.net(np.ascontiguousarray(x))
        pad = np.array([left, top], np.float32) / ratio  # letterbox offset in frame pixels
        scale = max(H, W)

        score = 1.0 / (1.0 + np.exp(-logit[0, :, 0].astype(np.float64)))
        keep = np.nonzero(score >= self.score)[0]
        if len(keep) == 0:
            return []
        reg, score, anc = reg[0, keep], score[keep], self.anchors[keep]
        cxy = reg[:, 0:2] / S + anc
        wh = reg[:, 2:4] / S
        boxes = np.concatenate([(cxy - wh / 2) * scale - pad, (cxy + wh / 2) * scale - pad], 1)
        kps = (reg[:, 4:18].reshape(-1, 7, 2) / S + anc[:, None, :]) * scale - pad
        idx = nms(boxes, score, self.nms_iou)
        return [(np.concatenate([boxes[i], kps[i].ravel()]).astype(np.float32), float(score[i])) for i in idx]


class HandLandmarker:
    SIZE = 224
    # crop geometry from MediaPipe (via OpenCV Zoo)
    PALM_BOX_PRE_ENLARGE = 4.0
    PALM_BOX_SHIFT = np.array([0.0, -0.4])
    PALM_BOX_ENLARGE = 3.0
    HAND_BOX_SHIFT = np.array([0.0, -0.1])
    HAND_BOX_ENLARGE = 1.65

    def __init__(self, path: str, device: torch.device):
        self.net = TorchModel(load_onnx_as_torch(path), device)

    @staticmethod
    def _crop_pad(image, box, shift, enlarge, square_diag):
        """Shift + enlarge a box, crop it and pad to a square. Returns crop, box, offset."""
        wh = box[1] - box[0]
        box = box + shift * wh
        c = box.sum(0) / 2
        half = (box[1] - box[0]) * enlarge / 2
        box = np.array([c - half, c + half]).astype(np.int32)
        box[:, 0] = np.clip(box[:, 0], 0, image.shape[1])
        box[:, 1] = np.clip(box[:, 1], 0, image.shape[0])
        crop = image[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        side = int(np.linalg.norm(crop.shape[:2]) if square_diag else max(crop.shape[:2]))
        ph, pw = side - crop.shape[0], side - crop.shape[1]
        left, top = pw // 2, ph // 2
        crop = cv2.copyMakeBorder(crop, top, ph - top, left, pw - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return crop, box, box[0] - np.array([left, top])

    def __call__(self, bgr: np.ndarray, palm: np.ndarray):
        """palm: 18 floats (box + 7 keypoints) from the palm detector or from the
        previous frame's landmarks. Returns (landmarks (21,3), world (21,3),
        right_prob, presence) in frame pixels, or None if the crop is empty."""
        # 1) generous crop around the palm, padded so rotation never cuts corners
        crop, cbox, bias = self._crop_pad(bgr, palm[0:4].reshape(2, 2), np.zeros(2), self.PALM_BOX_PRE_ENLARGE, True)
        if crop.size == 0 or min(crop.shape[:2]) < 8:
            return None
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        kps = palm[4:18].reshape(7, 2) - bias
        # 2) rotate so the wrist -> middle-finger-base direction points up
        p1, p2 = kps[0], kps[2]
        rad = np.pi / 2 - np.arctan2(-(p2[1] - p1[1]), p2[0] - p1[0])
        rad = rad - 2 * np.pi * np.floor((rad + np.pi) / (2 * np.pi))
        angle = np.rad2deg(rad)
        center = (cbox - bias).sum(0) / 2  # rotation centre: middle of the cropped region
        rot = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), float(angle), 1.0)
        rotated = cv2.warpAffine(crop, rot, (crop.shape[1], crop.shape[0]))
        rk = np.c_[kps, np.ones(7)] @ rot.T
        # 3) final crop from the rotated keypoints' box, resized to 224
        hand_crop, rbox, _ = self._crop_pad(rotated, np.array([rk.min(0), rk.max(0)]), self.PALM_BOX_SHIFT,
                                            self.PALM_BOX_ENLARGE, False)
        if hand_crop.size == 0 or min(hand_crop.shape[:2]) < 8:
            return None
        x = cv2.resize(hand_crop, (self.SIZE, self.SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)[None] / 255.0
        lm, presence, right, world = self.net(np.ascontiguousarray(x))

        # 4) map landmarks back: crop -> rotated image -> frame
        lm = lm[0].reshape(21, 3).astype(np.float64)
        world = world[0].reshape(21, 3).astype(np.float64)
        scale = max((rbox[1] - rbox[0]) / self.SIZE)
        lm[:, :2] = (lm[:, :2] - self.SIZE / 2) * scale
        lm[:, 2] *= scale
        back = cv2.getRotationMatrix2D((0, 0), float(angle), 1.0)
        lm[:, :2] = lm[:, :2] @ back[:, :2]
        world[:, :2] = world[:, :2] @ back[:, :2]
        R, t = rot[:, :2], rot[:, 2]
        inv = np.c_[R.T, -R.T @ t]  # inverse of the rotation about `center`
        c = np.append((rbox[0] + rbox[1]) / 2, 1.0)
        lm[:, :2] += inv @ c + bias
        return lm.astype(np.float32), world.astype(np.float32), float(right[0, 0]), float(presence[0, 0])


def hand_box(lm: np.ndarray):
    """Box around the landmarks, shifted/enlarged like MediaPipe's hand rect."""
    b = np.array([lm[:, :2].min(0), lm[:, :2].max(0)], np.float64)
    wh = b[1] - b[0]
    b = b + HandLandmarker.HAND_BOX_SHIFT * wh
    c, half = b.sum(0) / 2, (b[1] - b[0]) * HandLandmarker.HAND_BOX_ENLARGE / 2
    return np.concatenate([c - half, c + half]).astype(np.float32)


def palm_from_landmarks(lm: np.ndarray):
    """Build the palm-detector-style input (square box + 7 keypoints) from a
    hand's landmarks, so a tracked hand needs no palm detection."""
    k = lm[PALM_FROM_HAND, :2]
    c = (k.min(0) + k.max(0)) / 2
    half = (k.max(0) - k.min(0)).max() / 2
    return np.concatenate([c - half, c + half, k.ravel()]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Hand tracker
# --------------------------------------------------------------------------- #
@dataclass
class HandPolicy:
    max_hands: int = 2
    palm_interval: int = 5  # palm detector every N frames while hands are tracked (every frame when none)
    min_presence: float = 0.6  # landmark model's hand-presence score to keep a hand
    min_presence_new: float = 0.8  # ... to start a new hand
    overlap_iou: float = 0.3  # a detected palm overlapping a tracked hand's palm this much is that hand
    duplicate_iou: float = 0.5  # two tracked hands overlapping this much -> drop the younger


@dataclass
class _HandTrack:
    id: int
    landmarks: np.ndarray
    world: np.ndarray
    right_prob: float
    presence: float
    age: int = 0
    labels: list = field(default_factory=list)  # recent raw gesture labels (smoothing)
    stable: str = ""


class HandTracker:
    def __init__(self, palm: PalmDetector, lmk: HandLandmarker, pol: HandPolicy = HandPolicy(),
                 mirrored_input: bool = True):
        self.palm, self.lmk, self.pol = palm, lmk, pol
        self.mirrored = mirrored_input
        self.tracks: List[_HandTrack] = []
        self.next_id = 1
        self.timings = {}

    def reset(self):
        self.tracks, self.next_id = [], 1

    def _t(self, key, t0):
        self.timings[key] = self.timings.get(key, 0.0) + (time.perf_counter() - t0) * 1000

    def update(self, bgr: np.ndarray, frame_index: int, events: list):
        pol = self.pol
        # 1) follow every tracked hand from its previous landmarks
        alive = []
        for t in self.tracks:
            t0 = time.perf_counter()
            r = self.lmk(bgr, palm_from_landmarks(t.landmarks))
            self._t("landmarks", t0)
            if r is None or r[3] < pol.min_presence:
                events.append(f"hand drop #{t.id} lost (" + ("crop outside frame" if r is None else
                                                                  f"presence {r[3]:.2f}") + ")")
                continue
            t.landmarks, t.world, t.right_prob, t.presence = r
            t.age += 1
            alive.append(t)
        self.tracks = alive

        # 2) look for new hands
        if len(self.tracks) < pol.max_hands and (not self.tracks or frame_index % pol.palm_interval == 0):
            t0 = time.perf_counter()
            palms = self.palm(bgr)
            self._t("palm", t0)
            for p, _ in palms:
                if len(self.tracks) >= pol.max_hands:
                    break
                if any(self._same_hand(p, t) for t in self.tracks):
                    continue
                t0 = time.perf_counter()
                r = self.lmk(bgr, p)
                self._t("landmarks", t0)
                if r is None or r[3] < pol.min_presence_new:
                    continue
                t = _HandTrack(self.next_id, *r)
                self.next_id += 1
                self.tracks.append(t)
                events.append(f"hand new #{t.id}")

        # 3) two tracks on the same hand -> keep the older one
        keep = []
        for t in sorted(self.tracks, key=lambda t: -t.age):
            if any(iou(hand_box(t.landmarks), hand_box(k.landmarks)) > pol.duplicate_iou for k in keep):
                events.append(f"hand drop #{t.id} duplicate")
                continue
            keep.append(t)
        self.tracks = sorted(keep, key=lambda t: t.id)

    def _same_hand(self, palm: np.ndarray, t: _HandTrack) -> bool:
        """A detected palm belongs to a tracked hand if its centre lies inside that
        hand's box, or the two palm boxes overlap (a palm box is much smaller than a
        hand box, so plain IoU against the hand box would miss it)."""
        hb = hand_box(t.landmarks)
        cx, cy = (palm[0] + palm[2]) / 2, (palm[1] + palm[3]) / 2
        if hb[0] <= cx <= hb[2] and hb[1] <= cy <= hb[3]:
            return True
        return iou(palm[:4], palm_from_landmarks(t.landmarks)[:4]) > self.pol.overlap_iou

    def handedness(self, t: _HandTrack) -> str:
        """The person's real hand. This ONNX model's score is "right hand" for an
        ordinary, un-mirrored camera image (as in OpenCV Zoo's own demo). A mirrored
        (selfie) image is a reflection, which swaps left and right, so flip it back."""
        right = t.right_prob > 0.5
        if self.mirrored:
            right = not right
        return "Right" if right else "Left"

    def results(self) -> List[Hand]:
        return [Hand(t.id, self.handedness(t), t.presence, hand_box(t.landmarks), t.landmarks.copy())
                for t in self.tracks]
