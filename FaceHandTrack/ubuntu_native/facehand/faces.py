"""Face detection (YuNet) and face tracking (EdgeTAM streaming tracker) in PyTorch.

Same algorithm, policy and constants as the ONNX version (../onnx/facetrack/faces.py)
and the iOS app; here every model is a PyTorch module on the chosen device, and
EdgeTAM is built from Meta's own code and checkpoint. Per-face memory stays on
the device; only the 128 x 128 mask of each face comes back to the CPU.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import torch

from . import edgetam_modules as etm
from .backend import Timer, TorchModel
from .types import Face
from .weights import YuNetDecoded


# --------------------------------------------------------------------------- #
# Small geometry helpers (shared with hands.py)
# --------------------------------------------------------------------------- #
def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> List[int]:
    """Greedy non-maximum suppression. boxes (N,4) x1y1x2y2. Returns kept indices."""
    order = list(np.argsort(-scores))
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) <= thr]
    return keep


# --------------------------------------------------------------------------- #
# Face detector
# --------------------------------------------------------------------------- #
class FaceDetector:
    """YuNet at a fixed input size, frame letterboxed into the top-left corner."""

    def __init__(self, onnx_path: str, width: int, height: int, device: torch.device):
        self.w, self.h = width, height
        self.net = TorchModel(YuNetDecoded(onnx_path, width, height), device)

    def __call__(self, bgr: np.ndarray, min_score: float, nms_iou: float, min_px: float):
        H, W = bgr.shape[:2]
        s = min(self.w / W, self.h / H)
        canvas = np.zeros((self.h, self.w, 3), np.uint8)
        r = cv2.resize(bgr, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_LINEAR)
        canvas[: r.shape[0], : r.shape[1]] = r
        x = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]).astype(np.float32)  # BGR, 0..255, NCHW
        out = self.net(x)[0][0]  # (N, 15)
        out = out[(out[:, 0] >= min_score) & (np.minimum(out[:, 3], out[:, 4]) >= min_px)]
        if len(out) == 0:
            return []
        boxes = np.stack([out[:, 1], out[:, 2], out[:, 1] + out[:, 3], out[:, 2] + out[:, 4]], 1)
        keep = nms(boxes, out[:, 0], nms_iou)
        return [(boxes[i] / s, float(out[i, 0]), out[i, 5:15].reshape(5, 2) / s) for i in keep]


# --------------------------------------------------------------------------- #
# EdgeTAM streaming tracker
# --------------------------------------------------------------------------- #
@dataclass
class TrackMemory:
    """Per-face memory bank (tensors on the device). Slots are fixed in number."""

    cond_mem: list = field(default_factory=list)  # anchor frames, oldest first, <= C
    cond_ptr: list = field(default_factory=list)
    recent_mem: list = field(default_factory=list)  # newest first, <= R
    recent_ptr: list = field(default_factory=list)  # newest first, <= P - C


class EdgeTAM:
    def __init__(self, ckpt: str, cfg: dict, device: torch.device):
        self.cfg, self.device = cfg, device
        self.S, self.M = cfg["image_size"], cfg["mask_size"]
        self.C, self.R, self.P = cfg["num_cond"], cfg["num_recent"], cfg["num_ptr"]
        self.TM, self.TP, self.D = cfg["tokens_per_mem"], cfg["tokens_per_ptr"], cfg["mem_dim"]
        if device.type == "cuda":
            etm.ATTN_Q_CHUNK = 0  # one attention matrix per call; chunking only helps small memories
        sam = etm.load_edgetam(ckpt, image_size=self.S)
        self.enc = etm.EdgeTAMImageEncoder(sam).to(device).eval()
        self.att = etm.EdgeTAMMemoryAttention(sam, self.C, self.R, self.P).to(device).eval()
        self.dec = etm.EdgeTAMMaskDecoder(sam).to(device).eval()
        self.men = etm.EdgeTAMMemoryEncoder(sam).to(device).eval()
        K = (self.C + self.R) * self.TM + self.P * self.TP
        self._memory = torch.zeros(1, K, self.D, device=device)
        self._valid = torch.zeros(1, self.C + self.R + self.P, device=device)
        self._box_labels = torch.tensor([[2.0, 3.0]], device=device)  # box corners
        self._no_prompt = (torch.zeros(1, 1, 2, device=device), -torch.ones(1, 1, device=device))
        self._one, self._zero = torch.ones(1, device=device), torch.zeros(1, device=device)
        self.timer = Timer(device)

    @property
    def timings(self):
        return self.timer.timings

    @timings.setter
    def timings(self, value):
        self.timer.timings = value

    # once per frame, shared by all faces
    @torch.inference_mode()
    def encode(self, bgr: np.ndarray) -> dict:
        t0 = self.timer.start()
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.S, self.S), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)[None].float()  # whole frame squashed to S x S
        out = dict(zip(("feat", "feat_init", "tgt0", "hr0", "hr1"), self.enc(x)))
        self.timer.stop("enc", t0)
        return out

    def _assemble(self, mem: TrackMemory):
        """Fill the fixed [cond | recent | pointers] key layout; empty slots are
        flagged in slot_valid and masked out inside memory attention."""
        C, R, P, TM, TP = self.C, self.R, self.P, self.TM, self.TP
        m, v = self._memory, self._valid
        m.zero_()
        v.zero_()
        for i, tok in enumerate(mem.cond_mem[:C]):
            m[0, i * TM:(i + 1) * TM] = tok[0]
            v[0, i] = 1
        for i, tok in enumerate(mem.recent_mem[:R]):
            m[0, (C + i) * TM:(C + i + 1) * TM] = tok[0]
            v[0, C + i] = 1
        ptrs = list(mem.cond_ptr) + list(mem.recent_ptr[: P - len(mem.cond_ptr)])
        base = (C + R) * TM
        for i, ptr in enumerate(ptrs[:P]):
            m[0, base + i * TP: base + (i + 1) * TP] = ptr.reshape(TP, self.D)  # 256 -> 4 x 64 tokens
            v[0, C + R + i] = 1
        return m, v

    def _attend(self, enc, mem):
        t0 = self.timer.start()
        m, v = self._assemble(mem)
        pix = self.att(enc["tgt0"], m, v)
        self.timer.stop("attn", t0)
        return pix

    def _decode(self, enc, pix, coords, labels):
        t0 = self.timer.start()
        out = self.dec(pix, enc["hr0"], enc["hr1"], coords, labels)
        self.timer.stop("dec", t0)
        return out  # masks (1,4,M,M), iou (1,4), ptr (1,4,256), obj_score (1,1)

    def _memorize(self, enc, mask, binarize):
        t0 = self.timer.start()
        tok = self.men(enc["feat"], mask, binarize)
        self.timer.stop("menc", t0)
        return tok

    def _stability(self, mask) -> float:
        d = self.cfg["stability_delta"]
        union = (mask > -d).sum()
        return float((mask > d).sum() / union) if union > 0 else 1.0

    @torch.inference_mode()
    def start(self, enc, box_S, mem: Optional[TrackMemory] = None):
        """Start a track (mem=None) or re-anchor one from a detector box.
        box_S: x1, y1, x2, y2 in model-input pixels (0..S)."""
        new = mem is None
        mem = mem or TrackMemory()
        b = [float(v) for v in box_S]
        coords = torch.tensor([[[b[0], b[1]], [b[2], b[3]]]], device=self.device)
        pix = enc["feat_init"] if new else self._attend(enc, mem)
        masks, iou_pred, ptr, score = self._decode(enc, pix, coords, self._box_labels)
        # single-mask output, falling back to the best multimask output when unstable
        k = 0 if self._stability(masks[0, 0]) >= self.cfg["stability_thresh"] else 1 + int(torch.argmax(iou_pred[0, 1:]))
        mask = masks[:, k:k + 1]
        mem.cond_mem.append(self._memorize(enc, mask, self._one))
        mem.cond_ptr.append(ptr[:, 0])  # the single-mask token always gives the pointer
        if len(mem.cond_mem) > self.C:  # keep the first anchor + the newest ones
            del mem.cond_mem[1]
            del mem.cond_ptr[1]
        return mem, mask[0, 0].float().cpu().numpy(), float(iou_pred[0, k]), float(score[0, 0])

    @torch.inference_mode()
    def track(self, enc, mem: TrackMemory):
        pix = self._attend(enc, mem)
        masks, iou_pred, ptr, score = self._decode(enc, pix, *self._no_prompt)
        k = 1 + int(torch.argmax(iou_pred[0, 1:]))  # multimask for tracking
        mask = masks[:, k:k + 1]
        mem.recent_mem.insert(0, self._memorize(enc, mask, self._zero))
        mem.recent_ptr.insert(0, ptr[:, k])
        del mem.recent_mem[self.R:]
        del mem.recent_ptr[self.P - 1:]
        return mask[0, 0].float().cpu().numpy(), float(iou_pred[0, k]), float(score[0, 0])


# --------------------------------------------------------------------------- #
# Track manager (policy identical to v1 and the ONNX version)
# --------------------------------------------------------------------------- #
@dataclass
class FacePolicy:
    det_score: float = 0.80  # detector confidence to start a track
    nms_iou: float = 0.30
    det_interval: int = 3  # detector every N frames while faces are tracked (every frame when none)
    match_iou: float = 0.30  # detection <-> track association
    reanchor_iou: float = 0.50  # matched but overlap below this -> re-prompt with the detection box
    reanchor_min_gap: float = 0.5  # seconds between re-anchors of one track
    min_track_iou_pred: float = 0.60  # EdgeTAM's quality estimate below this -> re-anchor on next detection
    max_missing: float = 1.0  # seconds with no mask -> drop
    confident_iou_pred: float = 0.80
    max_unconfirmed: float = 5.0  # seconds without a detector match (not confident) -> drop
    max_unconfirmed_confident: float = 30.0  # ... while confident (detector misses tilted/profile faces)
    duplicate_iou: float = 0.60  # two tracks overlapping this much -> drop the younger
    max_tracks: int = 4
    min_face_px: float = 24.0  # in detector input pixels


@dataclass
class _Track:
    id: int
    mem: TrackMemory
    box: np.ndarray
    mask: np.ndarray
    age: int = 0
    last_seen: float = 0.0
    last_confirmed: float = 0.0
    last_anchor: float = 0.0
    iou_pred: float = 1.0
    visible: bool = True


def mask_box(mask: np.ndarray):
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


class FaceTracker:
    def __init__(self, tam: EdgeTAM, pol: FacePolicy = FacePolicy()):
        self.tam, self.pol = tam, pol
        self.tracks: List[_Track] = []
        self.next_id = 1

    def reset(self):
        self.tracks, self.next_id = [], 1

    def wants_detection(self, frame_index: int) -> bool:
        return not self.tracks or frame_index % self.pol.det_interval == 0

    def update(self, enc, dets, W: int, H: int, now: float, events: list):
        pol, S = self.pol, self.tam.S
        to_S = np.array([S / W, S / H, S / W, S / H], np.float32)

        # 1) associate detections with last-frame track boxes (greedy by IoU)
        pairs = sorted(((iou(t.box, d[0]), ti, di) for ti, t in enumerate(self.tracks) for di, d in enumerate(dets)),
                       reverse=True)
        t_used, d_used, match = set(), set(), {}
        for v, ti, di in pairs:
            if v < pol.match_iou or ti in t_used or di in d_used:
                continue
            t_used.add(ti)
            d_used.add(di)
            match[ti] = di

        # 2) propagate or re-anchor every live track
        for ti, t in enumerate(self.tracks):
            t.age += 1
            di = match.get(ti)
            if di is not None:
                t.last_confirmed = now
                want = iou(t.box, dets[di][0]) < pol.reanchor_iou or t.iou_pred < pol.min_track_iou_pred
                if want and now - t.last_anchor >= pol.reanchor_min_gap:
                    _, t.mask, t.iou_pred, score = self.tam.start(enc, dets[di][0] * to_S, t.mem)
                    t.last_anchor = now
                    events.append(f"face re-anchor #{t.id}")
                    self._update_box(t, W, H, score, now)
                    continue
            t.mask, t.iou_pred, score = self.tam.track(enc, t.mem)
            self._update_box(t, W, H, score, now)

        # 3) start tracks for unmatched detections
        for di, (box, _, _) in enumerate(dets):
            if di in d_used or len(self.tracks) >= pol.max_tracks:
                continue
            if any(iou(box, t.box) > pol.match_iou for t in self.tracks):
                continue
            mem, m, ip, score = self.tam.start(enc, box * to_S)
            t = _Track(self.next_id, mem, box.copy(), m, iou_pred=ip, last_seen=now, last_confirmed=now,
                       last_anchor=now)
            self.next_id += 1
            self._update_box(t, W, H, score, now)
            self.tracks.append(t)
            events.append(f"face new #{t.id}")

        # 4) retire lost / unconfirmed / duplicate tracks
        keep = []
        for t in sorted(self.tracks, key=lambda t: -t.age):
            limit = pol.max_unconfirmed_confident if t.iou_pred >= pol.confident_iou_pred else pol.max_unconfirmed
            if now - t.last_seen > pol.max_missing:
                events.append(f"face drop #{t.id} lost")
                continue
            if now - t.last_confirmed > limit:
                events.append(f"face drop #{t.id} unconfirmed")
                continue
            if t.visible and any(k.visible and iou(t.box, k.box) > pol.duplicate_iou for k in keep):
                events.append(f"face drop #{t.id} duplicate")
                continue
            keep.append(t)
        self.tracks = sorted(keep, key=lambda t: t.id)

    @staticmethod
    def _update_box(t: _Track, W, H, score, now):
        b = mask_box(t.mask) if score > 0 else None
        t.visible = b is not None
        if b is None:
            return
        t.last_seen = now
        g = t.mask.shape[0]
        t.box = b * np.array([W / g, H / g, W / g, H / g], np.float32)

    def results(self) -> List[Face]:
        return [Face(t.id, t.box.copy(), t.visible, t.iou_pred, t.mask) for t in self.tracks]
