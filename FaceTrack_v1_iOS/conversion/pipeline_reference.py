#!/usr/bin/env python3
"""Desktop reference of the whole on-device pipeline (YuNet + EdgeTAM + track
manager). Same policy and constants as ios/FaceTrack/TrackManager.swift, so it
can be used to tune thresholds on recorded clips before touching the phone.

    python pipeline_reference.py --frames clip/ --out annotated.mp4
"""
import argparse
import glob
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from convert_yunet import YuNetDecoded
from edgetam_coreml_modules import (EdgeTAMImageEncoder, EdgeTAMMaskDecoder, EdgeTAMMemoryAttention,
                                    EdgeTAMMemoryEncoder, load_edgetam)
from reference_tracker import ReferenceTracker, TrackMemory, mask_box


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="EdgeTAM/checkpoints/edgetam.pt")
    ap.add_argument("--yunet", default="face_detection_yunet_2023mar.onnx")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="annotated.mp4")
    ap.add_argument("--num-cond", type=int, default=2)
    ap.add_argument("--num-recent", type=int, default=6)
    ap.add_argument("--num-ptr", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=1024, help="EdgeTAM input size (1024, 768 or 512)")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate of the clip (policy limits are in seconds)")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    sam = load_edgetam(args.ckpt, image_size=args.image_size)
    C, R, P = args.num_cond, args.num_recent, args.num_ptr
    trk = ReferenceTracker(EdgeTAMImageEncoder(sam), EdgeTAMMemoryAttention(sam, C, R, P), EdgeTAMMaskDecoder(sam),
                           EdgeTAMMemoryEncoder(sam), C, R, P, args.image_size)
    pipe = Pipeline(trk, Detector(args.yunet), fps=args.fps)
    files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))
    first = cv2.imread(files[0])
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 10, (first.shape[1], first.shape[0]))
    colors = [(66, 135, 245), (80, 200, 120), (245, 170, 66), (200, 80, 200)]
    for i, f in enumerate(files):
        img = cv2.imread(f)
        t0 = time.perf_counter()
        tracks = pipe.step(img)
        dt = time.perf_counter() - t0
        for t in tracks:
            c = colors[t.tid % len(colors)]
            if t.visible:
                m = cv2.resize(t.mask, (img.shape[1], img.shape[0])) > 0
                img[m] = (0.55 * img[m] + 0.45 * np.array(c)).astype(np.uint8)
                x1, y1, x2, y2 = t.box.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
                cv2.putText(img, f"#{t.tid}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        vw.write(img)
        print(f"frame {i:3d}  tracks {[(t.tid, 'vis' if t.visible else 'occ', round(t.iou_pred, 2)) for t in tracks]}  {dt * 1000:.0f} ms (CPU)")
    vw.release()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
