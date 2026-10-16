"""v1 Python reference pipeline (PyTorch modules), kept only as the test oracle
for tests/test_faces_vs_v1.py. Merged from v1 conversion/reference_tracker.py and
conversion/pipeline_reference.py; the detector uses the decoded YuNet module."""
import cv2
from yunet_modules import YuNetDecoded

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from edgetam_modules import TOKENS_PER_MEM, TOKENS_PER_PTR

STABILITY_DELTA = 0.05
STABILITY_THRESH = 0.98


@dataclass
class TrackMemory:
    """Per-face memory bank. Slots are fixed so graphs keep static shapes."""

    cond_mem: List[torch.Tensor] = field(default_factory=list)  # newest last, <= C
    cond_ptr: List[torch.Tensor] = field(default_factory=list)
    recent_mem: List[torch.Tensor] = field(default_factory=list)  # newest first, <= R
    recent_ptr: List[torch.Tensor] = field(default_factory=list)  # newest first, <= P - C


def stability(mask: torch.Tensor) -> float:
    inter = (mask > STABILITY_DELTA).sum().item()
    union = (mask > -STABILITY_DELTA).sum().item()
    return inter / union if union > 0 else 1.0


class ReferenceTracker:
    def __init__(self, enc, attn, dec, menc, num_cond, num_recent, num_ptr, image_size=1024):
        self.enc, self.attn, self.dec, self.menc = enc, attn, dec, menc
        self.C, self.R, self.P = num_cond, num_recent, num_ptr
        self.image_size = image_size

    # ---- per frame, shared by all faces --------------------------------------
    @torch.no_grad()
    def encode(self, image: torch.Tensor):
        """image: (1,3,S,S) float RGB in 0..255, S = EdgeTAM input size."""
        return self.enc(image)

    # ---- memory bank assembly --------------------------------------------------
    def assemble(self, mem: TrackMemory):
        C, R, P = self.C, self.R, self.P
        zero_m = torch.zeros(1, TOKENS_PER_MEM, 64)
        zero_p = torch.zeros(1, 256)
        slots, valid = [], []
        for i in range(C):
            ok = i < len(mem.cond_mem)
            slots.append(mem.cond_mem[i] if ok else zero_m)
            valid.append(float(ok))
        for i in range(R):
            ok = i < len(mem.recent_mem)
            slots.append(mem.recent_mem[i] if ok else zero_m)
            valid.append(float(ok))
        ptrs = list(mem.cond_ptr) + list(mem.recent_ptr[: P - len(mem.cond_ptr)])
        for i in range(P):
            ok = i < len(ptrs)
            p = ptrs[i] if ok else zero_p
            slots.append(p.reshape(1, TOKENS_PER_PTR, 64))  # 256 -> 4 x 64 tokens
            valid.append(float(ok))
        return torch.cat(slots, 1), torch.tensor([valid])

    # ---- start a track from a detector box (conditioning frame) ---------------
    @torch.no_grad()
    def add_box(self, enc_out, box_1024, mem: Optional[TrackMemory] = None):
        """box_1024: x1, y1, x2, y2 in model-input pixels (0..S; the name is historical)."""
        feat, feat_init, tgt0, hr0, hr1 = enc_out
        new = mem is None
        mem = mem or TrackMemory()
        coords = torch.tensor([[[box_1024[0], box_1024[1]], [box_1024[2], box_1024[3]]]], dtype=torch.float32)
        labels = torch.tensor([[2.0, 3.0]])
        if new:
            pix = feat_init  # first frame of the track: no memory, add no_mem_embed
        else:
            # re-anchoring an existing track: condition on its memory, as SAM 2
            # does when a correction prompt lands on a non-initial frame
            m, v = self.assemble(mem)
            pix = self.attn(tgt0, m, v)
        masks, iou, ptr, score = self.dec(pix, hr0, hr1, coords, labels)
        # single-mask output with dynamic fallback to the best multimask output
        if stability(masks[0, 0]) >= STABILITY_THRESH:
            k = 0
        else:
            k = 1 + int(torch.argmax(iou[0, 1:]))
        mask = masks[:, k : k + 1]
        obj_ptr = ptr[:, 0]  # the single-mask token always gives the pointer
        tokens = self.menc(feat, mask, torch.ones(1))  # binarised: mask came from a prompt
        mem.cond_mem.append(tokens)
        mem.cond_ptr.append(obj_ptr)
        if len(mem.cond_mem) > self.C:  # keep the first anchor + the newest ones
            del mem.cond_mem[1]
            del mem.cond_ptr[1]
        return mem, mask, float(iou[0, k]), float(score)

    # ---- propagate a track one frame -----------------------------------------
    @torch.no_grad()
    def track(self, enc_out, mem: TrackMemory):
        feat, feat_init, tgt0, hr0, hr1 = enc_out
        m, v = self.assemble(mem)
        pix = self.attn(tgt0, m, v)
        coords = torch.zeros(1, 1, 2)
        labels = -torch.ones(1, 1)
        masks, iou, ptr, score = self.dec(pix, hr0, hr1, coords, labels)
        k = 1 + int(torch.argmax(iou[0, 1:]))  # multimask for tracking
        mask = masks[:, k : k + 1]
        obj_ptr = ptr[:, k]
        tokens = self.menc(feat, mask, torch.zeros(1))
        mem.recent_mem.insert(0, tokens)
        mem.recent_ptr.insert(0, obj_ptr)
        del mem.recent_mem[self.R :]
        del mem.recent_ptr[self.P - 1 :]
        return mask, float(iou[0, k]), float(score)


def mask_box(mask: np.ndarray, thresh: float = 0.0):
    """Tight box (x1, y1, x2, y2) of a low-res mask, in mask-grid pixels."""
    ys, xs = np.nonzero(mask > thresh)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


@dataclass
class Policy:  # keep in sync with TrackManager.Policy (Swift)
    det_score: float = 0.80  # detector confidence to start a track
    nms_iou: float = 0.30
    det_interval: int = 3  # run detector every N frames while tracks exist (every frame when none)
    match_iou: float = 0.30  # detection <-> track association
    reanchor_iou: float = 0.50  # matched but overlap below this -> re-prompt with the detection box
    reanchor_min_gap: float = 0.5  # seconds between re-anchors of one track
    min_track_iou_pred: float = 0.60  # SAM's own IoU estimate below this -> re-anchor when a detection is available
    # time limits in seconds (frame-rate independent)
    max_missing: float = 1.0  # object score <= 0 this long -> drop
    confident_iou_pred: float = 0.80  # SAM mask-quality estimate that counts as confident
    max_unconfirmed: float = 5.0  # no detector match this long while not confident -> drop
    max_unconfirmed_confident: float = 30.0  # ... while confident (detector misses tilted/profile faces)
    duplicate_iou: float = 0.60  # two tracks overlapping this much -> drop the younger
    max_tracks: int = 4
    min_face_px: float = 24.0  # in detector input pixels


@dataclass
class Track:
    tid: int
    mem: TrackMemory
    box: np.ndarray  # x1,y1,x2,y2 in frame pixels
    mask: np.ndarray = None  # (S/4)x(S/4) logits over the squashed frame
    age: int = 0
    last_seen: float = 0.0
    last_confirmed: float = 0.0
    last_anchor: float = 0.0
    iou_pred: float = 1.0
    visible: bool = True


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def nms(d, thr):
    order = list(np.argsort(-d[:, 0]))
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        bi = [d[i, 1], d[i, 2], d[i, 1] + d[i, 3], d[i, 2] + d[i, 4]]
        order = [j for j in order if iou(bi, [d[j, 1], d[j, 2], d[j, 1] + d[j, 3], d[j, 2] + d[j, 4]]) <= thr]
    return d[keep]


class Detector:
    """YuNet at a fixed input size with top-left letterboxing (same as the app)."""

    def __init__(self, onnx_path, w=384, h=640):
        self.w, self.h = w, h
        self.net = YuNetDecoded(onnx_path, w, h).eval()

    @torch.no_grad()
    def __call__(self, bgr, pol: Policy):
        H, W = bgr.shape[:2]
        s = min(self.w / W, self.h / H)
        canvas = np.zeros((self.h, self.w, 3), np.uint8)
        r = cv2.resize(bgr, (int(round(W * s)), int(round(H * s))))
        canvas[: r.shape[0], : r.shape[1]] = r
        out = self.net(torch.from_numpy(canvas).permute(2, 0, 1)[None].float())[0].numpy()
        out = out[(out[:, 0] >= pol.det_score) & (np.minimum(out[:, 3], out[:, 4]) >= pol.min_face_px)]
        out = nms(out, pol.nms_iou) if len(out) else out
        return [(np.array([d[1], d[2], d[1] + d[3], d[2] + d[4]]) / s, float(d[0])) for d in out]


class Pipeline:
    def __init__(self, trk: ReferenceTracker, det: Detector, pol=Policy(), fps: float = 30.0):
        self.trk, self.det, self.pol, self.fps = trk, det, pol, fps
        self.tracks, self.next_id, self.frame = [], 1, 0

    def step(self, bgr):
        pol, H, W = self.pol, *bgr.shape[:2]
        now = self.frame / self.fps
        run_det = not self.tracks or self.frame % pol.det_interval == 0
        dets = self.det(bgr, pol) if run_det else []
        self.frame += 1
        if not self.tracks and not dets:
            return []  # nothing to track: the EdgeTAM encoder is not run at all

        S = self.trk.image_size
        rgb = cv2.cvtColor(cv2.resize(bgr, (S, S)), cv2.COLOR_BGR2RGB)
        enc = self.trk.encode(torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)[None])
        to1024 = np.array([S / W, S / H, S / W, S / H])  # frame -> model-input pixels

        # 1) associate detections with last-frame track boxes (greedy by IoU)
        pairs = sorted(((iou(t.box, d[0]), ti, di) for ti, t in enumerate(self.tracks) for di, d in enumerate(dets)), reverse=True)
        t_used, d_used, match = set(), set(), {}
        for v, ti, di in pairs:
            if v < pol.match_iou or ti in t_used or di in d_used:
                continue
            t_used.add(ti); d_used.add(di); match[ti] = di

        # 2) propagate or re-anchor every live track
        for ti, t in enumerate(self.tracks):
            t.age += 1
            di = match.get(ti)
            if di is not None:
                t.last_confirmed = now
                want = (iou(t.box, dets[di][0]) < pol.reanchor_iou or t.iou_pred < pol.min_track_iou_pred)
                if want and now - t.last_anchor >= pol.reanchor_min_gap:
                    _, m, t.iou_pred, score = self.trk.add_box(enc, dets[di][0] * to1024, t.mem)
                    t.mask, t.last_anchor = m[0, 0].numpy(), now
                    self._update_box(t, W, H, score, now)
                    continue
            m, t.iou_pred, score = self.trk.track(enc, t.mem)
            t.mask = m[0, 0].numpy()
            self._update_box(t, W, H, score, now)

        # 3) spawn tracks for unmatched detections
        for di, (box, _) in enumerate(dets):
            if di in d_used or len(self.tracks) >= pol.max_tracks:
                continue
            if any(iou(box, t.box) > pol.match_iou for t in self.tracks):
                continue
            mem, m, ip, score = self.trk.add_box(enc, box * to1024)
            t = Track(self.next_id, mem, box.copy(), m[0, 0].numpy(), iou_pred=ip,
                      last_seen=now, last_confirmed=now, last_anchor=now)
            self.next_id += 1
            self._update_box(t, W, H, score, now)
            self.tracks.append(t)

        # 4) retire lost / unconfirmed / duplicate tracks
        keep = []
        for t in sorted(self.tracks, key=lambda t: -t.age):
            limit = pol.max_unconfirmed_confident if t.iou_pred >= pol.confident_iou_pred else pol.max_unconfirmed
            if now - t.last_seen > pol.max_missing or now - t.last_confirmed > limit:
                continue
            if t.visible and any(k.visible and iou(t.box, k.box) > pol.duplicate_iou for k in keep):
                continue
            keep.append(t)
        self.tracks = sorted(keep, key=lambda t: t.tid)
        return self.tracks

    @staticmethod
    def _update_box(t, W, H, score, now):
        b = mask_box(t.mask) if score > 0 else None
        t.visible = b is not None
        if b is None:
            return
        t.last_seen = now
        g = t.mask.shape[0]
        t.box = b * np.array([W / g, H / g, W / g, H / g])


