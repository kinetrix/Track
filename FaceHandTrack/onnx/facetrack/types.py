"""Results the engine returns for each frame.

Plain data only (numbers, strings, numpy arrays), so any interface can consume
them: draw them in a window (app/mac_viewer.py), send them as JSON to a web page
(FrameResult.to_dict), or re-implement the same structs in Swift/Kotlin/JS.
All coordinates are in pixels of the frame that was passed to Engine.process.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# MediaPipe's 21 hand landmarks
HAND_LANDMARKS = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
# bones to draw a hand skeleton
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


@dataclass
class Face:
    id: int
    box: np.ndarray  # x1, y1, x2, y2
    visible: bool  # False while the face is hidden (track kept alive for re-appearance)
    quality: float  # EdgeTAM's own mask-quality estimate (predicted IoU, 0..1)
    mask: Optional[np.ndarray] = None  # (M, M) logits over the whole frame squashed to a square; > 0 = face

    def to_dict(self):
        return {"id": self.id, "box": [round(float(v), 1) for v in self.box], "visible": self.visible,
                "quality": round(self.quality, 3)}


@dataclass
class Hand:
    id: int
    handedness: str  # "Left" / "Right": the person's real hand, whether or not the view is mirrored
    score: float  # hand-presence confidence
    box: np.ndarray  # x1, y1, x2, y2
    landmarks: np.ndarray  # (21, 3): x, y in pixels, z relative depth (wrist = 0)
    gesture: str = ""  # stable gesture label after smoothing ("" = none)
    fingers: int = 0  # number of extended fingers
    raw_gesture: str = ""  # this frame's label before smoothing

    def to_dict(self):
        return {"id": self.id, "handedness": self.handedness, "score": round(self.score, 3),
                "box": [round(float(v), 1) for v in self.box],
                "landmarks": [[round(float(v), 1) for v in p] for p in self.landmarks],
                "gesture": self.gesture, "fingers": self.fingers}


@dataclass
class FrameResult:
    index: int
    time: float  # seconds, as passed to Engine.process
    faces: List[Face] = field(default_factory=list)
    hands: List[Hand] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)  # milliseconds per stage
    events: List[str] = field(default_factory=list)  # track births/deaths this frame

    def to_dict(self):
        return {"index": self.index, "time": self.time, "faces": [f.to_dict() for f in self.faces],
                "hands": [h.to_dict() for h in self.hands],
                "timings": {k: round(v, 2) for k, v in self.timings.items()}, "events": self.events}
