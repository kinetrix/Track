"""Python reference of the on-device tracker, built only from the four exported
graphs in edgetam_coreml_modules.py. The Swift `EdgeTAMTracker` mirrors this
file line by line, so it doubles as executable documentation and as the test
oracle for the Core ML export (see validate_edgetam.py).
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from edgetam_coreml_modules import TOKENS_PER_MEM, TOKENS_PER_PTR

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
