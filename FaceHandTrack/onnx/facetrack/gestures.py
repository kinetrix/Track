"""Static hand gestures from the 21 landmarks, with hand-written rules.

No model is needed: each finger is classified as extended or folded, and a
gesture is a pattern of extended fingers plus a few distances/directions.
Thresholds were tuned on the 24 labelled HaGRID sample images in
tests/hand_images (see tests/test_gestures.py).

Finger state uses the landmark model's *world* landmarks (3-D, metric, centred
on the hand), so it does not change with hand size, distance or in-plane
rotation. Pinch distance and pointing direction use the image landmarks,
because they are about what the camera sees.
"""
from collections import Counter, deque

import numpy as np

FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}

# thresholds
EXT_RATIO = 1.15  # fingertip at least this much farther from the wrist than the PIP joint
EXT_BEND = 80.0  # and the finger bends less than this (degrees, PIP + DIP)
THUMB_OUT = 0.85  # thumb tip distance from ring-finger base, in palm lengths
THUMB_BEND = 60.0  # thumb bends less than this (degrees, MCP + IP)
PINCH = 0.25  # thumb tip - index tip distance, in palm lengths
INDEX_NOT_FOLDED = 0.88  # pinch needs the index finger only partly bent (a fist also brings the tips together)
INDEX_CLEAR = 1.25  # "I love you" needs the index clearly straight; a half-bent index stays "Call me"

# the gestures this module can return
GESTURES = ["Open palm", "Fist", "Thumbs up", "Thumbs down", "Pointing up", "Pointing down", "Pointing left",
            "Pointing right", "Victory", "Three", "Rock", "I love you", "Call me", "OK", "Pinch"]


def _angle(a, b) -> float:
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def finger_states(world: np.ndarray) -> dict:
    """Extended (True) or folded (False) for thumb, index, middle, ring, pinky."""
    w = world
    palm = np.linalg.norm(w[9] - w[0]) + 1e-9
    out = {}
    s = np.diff(w[[2, 3, 4]], axis=0)
    out["thumb"] = (np.linalg.norm(w[4] - w[13]) / palm > THUMB_OUT and _angle(s[0], s[1]) +
                    _angle(w[2] - w[1], s[0]) < THUMB_BEND)
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        s = np.diff(w[[mcp, pip, dip, tip]], axis=0)
        bend = _angle(s[0], s[1]) + _angle(s[1], s[2])
        ratio = np.linalg.norm(w[tip] - w[0]) / (np.linalg.norm(w[pip] - w[0]) + 1e-9)
        out[name] = bool(ratio > EXT_RATIO and bend < EXT_BEND)
    return out


def _direction(v) -> str:
    """Screen direction of an image-space vector (y grows downwards)."""
    if abs(v[1]) >= abs(v[0]):
        return "up" if v[1] < 0 else "down"
    return "left" if v[0] < 0 else "right"


def classify(landmarks: np.ndarray, world: np.ndarray):
    """landmarks: (21,3) image pixels (+ relative z); world: (21,3) metric.
    Returns (gesture or "", number of extended fingers)."""
    st = finger_states(world)
    count = sum(st.values())
    T, I, M, R, P = (st[k] for k in ("thumb", "index", "middle", "ring", "pinky"))
    lm = landmarks
    palm_img = np.linalg.norm(lm[9] - lm[0]) + 1e-9
    index_ratio = np.linalg.norm(world[8] - world[0]) / (np.linalg.norm(world[6] - world[0]) + 1e-9)
    pinch = np.linalg.norm(lm[4] - lm[8]) / palm_img < PINCH and index_ratio > INDEX_NOT_FOLDED

    if pinch:
        return ("OK" if M and R and P else "Pinch"), count
    if I and M and R and P:
        return "Open palm", count
    if not (I or M or R or P):
        if not T:
            return "Fist", count
        d = _direction(lm[4, :2] - lm[2, :2])
        if d in ("up", "down"):
            return ("Thumbs up" if d == "up" else "Thumbs down"), count
        return "", count
    if I and not (M or R or P):
        return "Pointing " + _direction(lm[8, :2] - lm[5, :2]), count
    if I and M and not (R or P) and not T:
        return "Victory", count
    if I and M and R and not P and not T:
        return "Three", count
    if I and P and not (M or R):
        if not T:
            return "Rock", count
        return ("I love you" if index_ratio > INDEX_CLEAR else "Call me"), count
    if T and P and not (I or M or R):
        return "Call me", count
    return "", count


class GestureSmoother:
    """Reports a gesture only after it won the majority of the last `window`
    frames, so labels don't flicker while the hand moves between poses."""

    def __init__(self, window: int = 5, need: int = 3):
        self.need = need
        self.hist = deque(maxlen=window)
        self.stable = ""

    def __call__(self, label: str) -> str:
        self.hist.append(label)
        top, n = Counter(self.hist).most_common(1)[0]
        if n >= self.need:
            self.stable = top
        return self.stable
