"""Runtime-neutral re-packaging of EdgeTAM (Meta, Apache-2.0) for *video tracking*.
Carried over from v1 (where it fed Core ML); in v2 it feeds ONNX.

The official EdgeTAM exports ship only image encoder / prompt encoder / mask
decoder, i.e. single-image SAM, and never use the memory bank. This file exports
the full SAM-2-style streaming tracker, split into four static-shape graphs:

    All shapes follow the input size S (1024 = paper setting; 768 / 512 run the
    same checkpoint without retraining). Feature grid F = S/16, mask grid M = S/4.

    EdgeTAMImageEncoder   once per frame (shared by all tracked faces)
        image(S x S RGB) -> feat, feat_init, tgt0, hr0, hr1
        * ImageNet normalisation folded in
        * conv_s0/conv_s1 high-res projections folded in
        * layer-0 self-attention of memory attention folded in: it only sees the
          current frame's features, so it is identical for every object and is
          computed once per frame instead of once per face.

    EdgeTAMMemoryAttention   once per face per frame
        tgt0, memory bank, slot validity -> memory-conditioned features
        * fixed number of memory slots; empty slots are masked out of the softmax
          (mathematically identical to not concatenating them)
        * complex-number RoPE rewritten with real cos/sin tables
        * all positional encodings (perceiver + temporal) are constants -> baked in

    EdgeTAMMaskDecoder   once per face per frame
        features, hr0, hr1, points(1 or 2), labels -> 4 mask logits, 4 IoUs,
        4 object pointers, object score
        * prompt encoder folded in (box corners = labels 2/3, padding = -1)
        * occlusion gating (NO_OBJ_SCORE / no_obj_ptr) folded in

    EdgeTAMMemoryEncoder   once per face per frame
        raw feat, chosen low-res mask, binarize flag -> 512 x 64 memory tokens
        * bilinear upsample to S, sigmoid*20-10 (or binarise) folded in
        * spatial perceiver folded in
"""
import math
import os
import sys
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

EDGETAM_ROOT = os.environ.get("EDGETAM_ROOT", os.path.join(os.path.dirname(__file__), "EdgeTAM"))
sys.path.insert(0, EDGETAM_ROOT)

NO_OBJ_SCORE = -1024.0
IMG_SIZE = 1024  # default input size; any multiple of 256 works (see load_edgetam)
TOKENS_PER_MEM = 512  # 256 global perceiver latents + 16x16 2-D latents
TOKENS_PER_PTR = 4  # 256-d object pointer split into 4 x 64-d tokens
MASK_NEG = -1.0e4  # additive attention bias for empty slots (fp16 safe)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_edgetam(ckpt: str, video_predictor: bool = False, image_size: int = IMG_SIZE):
    """Build EdgeTAM exactly as `build_sam2_video_predictor` does (same
    post-processing overrides) without touching the network: timm's
    'pretrained=True' download is disabled because the checkpoint holds all
    backbone weights anyway.

    image_size: 1024 (paper) or a smaller multiple of 256. Only the input size
    and the RoPE grid sizes change (grid = image_size / 16); the checkpoint is
    the same, no retraining (as in the faster-edgetam fork's 512/768 configs)."""
    assert image_size % 256 == 0, "image_size must be a multiple of 256 (perceiver windows)"
    import timm.models
    import sam2.modeling.backbones.timm as sam_timm
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam2.build_sam import _load_checkpoint

    _orig = timm.models.create_model

    def _no_download(*a, **k):
        k["pretrained"] = False
        return _orig(*a, **k)

    sam_timm.create_model = _no_download

    overrides = [
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        "++model.binarize_mask_from_pts_for_mem_enc=true",
        "++model.fill_hole_area=0",  # CUDA-only op; not used on device either
    ]
    grid = image_size // 16
    overrides += [
        f"++model.image_size={image_size}",
        f"++model.memory_attention.layer.self_attention.feat_sizes=[{grid},{grid}]",
        f"++model.memory_attention.layer.cross_attention.q_sizes=[{grid},{grid}]",
    ]
    if video_predictor:
        overrides.insert(0, "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor")
    else:
        overrides = [o for o in overrides if "fill_hole_area" not in o]

    GlobalHydra.instance().clear()
    cfg_dir = os.path.join(EDGETAM_ROOT, "sam2", "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="edgetam", overrides=overrides)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt)
    return model.eval()


# --------------------------------------------------------------------------- #
# Real-valued RoPE + masked attention
# --------------------------------------------------------------------------- #
def axial_angles(dim: int, end_x: int, end_y: int, theta: float = 10000.0) -> torch.Tensor:
    """Angles equivalent to sam2 `compute_axial_cis` (which returns polar(1, angle))."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: dim // 4].float() / dim))
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x, t_y = t % end_x, torch.div(t, end_x, rounding_mode="floor")
    return torch.cat([torch.outer(t_x, freqs), torch.outer(t_y, freqs)], -1)  # (N, dim/2)


def rope_tables(angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(N, D/2) angles -> interleaved (N, D) cos / sin tables."""
    return angles.cos().repeat_interleave(2, -1), angles.sin().repeat_interleave(2, -1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, N, D) with interleaved complex pairs. Rotation (a,b)->(a c - b s, a s + b c)."""
    B, N, D = x.shape
    pairs = x.reshape(B, N, D // 2, 2)
    rot = torch.stack([-pairs[..., 1], pairs[..., 0]], -1).reshape(B, N, D)
    return x * cos + rot * sin


# Queries per attention chunk. A full 4096 x 4160 fp16 score matrix is ~34 MB, far
# larger than the Neural Engine's on-chip memory, so the unchunked op is memory-bound
# (measured: 44 ms for memory attention on iPhone 17). Splitting the queries keeps each
# score block at ~4 MB. Mathematically identical: softmax is per query row.
# 0 disables chunking. Set via export_edgetam.py --attn-chunk.
ATTN_Q_CHUNK = 512


def attend(attn: nn.Module, q_in, k_in, v_in, cos_q=None, sin_q=None, cos_k=None, sin_k=None, bias=None):
    """Single-head attention using an sam2 Attention module's weights (all
    EdgeTAM memory-attention blocks are single-head), chunked over queries."""
    assert attn.num_heads == 1
    q = attn.q_proj(q_in)
    k = attn.k_proj(k_in)
    v = attn.v_proj(v_in)
    if cos_q is not None:
        q = apply_rope(q, cos_q, sin_q)
    if cos_k is not None:
        k = apply_rope(k, cos_k, sin_k)
    q = q * (1.0 / math.sqrt(q.shape[-1]))  # scale the small tensor, not the score matrix
    kt = k.transpose(1, 2)
    n = q.shape[1]
    step = ATTN_Q_CHUNK if 0 < ATTN_Q_CHUNK < n else n
    outs = []
    for i in range(0, n, step):
        scores = torch.matmul(q[:, i:i + step], kt)
        if bias is not None:
            scores = scores + bias
        outs.append(torch.matmul(torch.softmax(scores, -1), v))
    out = outs[0] if len(outs) == 1 else torch.cat(outs, 1)
    return attn.out_proj(out)


# --------------------------------------------------------------------------- #
# 1) Image encoder (+ shared first self-attention)
# --------------------------------------------------------------------------- #
class EdgeTAMImageEncoder(nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.sam = sam
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255.0)
        S, Fg = sam.image_size, sam.image_size // 16
        with torch.no_grad():
            out = sam.forward_image(torch.zeros(1, 3, S, S))
        pos = out["vision_pos_enc"][-1]  # (1,256,F,F), depends on shape only
        self.register_buffer("curr_pos", pos.flatten(2).transpose(1, 2).clone())  # (1,F*F,256)
        cq, sq = rope_tables(axial_angles(256, Fg, Fg))
        self.register_buffer("cos_q", cq)
        self.register_buffer("sin_q", sq)

    def forward(self, image):
        x = (image - self.mean) / self.std
        fpn = self.sam.forward_image(x)["backbone_fpn"]
        hr0, hr1, feat = fpn[0], fpn[1], fpn[2]
        feat_init = feat + self.sam.no_mem_embed.view(1, -1, 1, 1)
        # memory attention: input pos-enc add + layer-0 self-attention (object independent)
        ma = self.sam.memory_attention
        tgt = feat.flatten(2).transpose(1, 2) + 0.1 * self.curr_pos
        layer0 = ma.layers[0]
        t2 = layer0.norm1(tgt)
        tgt0 = tgt + attend(layer0.self_attn, t2, t2, t2, self.cos_q, self.sin_q, self.cos_q, self.sin_q)
        return feat, feat_init, tgt0, hr0, hr1


# --------------------------------------------------------------------------- #
# 2) Memory attention with a fixed-size, masked memory bank
# --------------------------------------------------------------------------- #
class EdgeTAMMemoryAttention(nn.Module):
    """Memory layout (keys), all 64-d:
         [cond slot 0 .. cond slot C-1 | recent slot 0 (newest) .. recent slot R-1 | P pointer slots]
       each spatial slot = 512 tokens (256 global latents, then 256 2-D latents),
       each pointer slot = 4 tokens.
       slot_valid: (1, C+R+P) with 1 = filled, 0 = empty."""

    def __init__(self, sam, num_cond: int, num_recent: int, num_ptr: int):
        super().__init__()
        assert 1 <= num_recent <= sam.num_maskmem - 1
        self.sam = sam
        self.C, self.R, self.P = num_cond, num_recent, num_ptr
        self.S = num_cond + num_recent
        self.Lk = self.S * TOKENS_PER_MEM + num_ptr * TOKENS_PER_PTR
        self.Fg = Fg = sam.image_size // 16

        # perceiver output positional encoding: zeros for global latents, sine for 2-D latents
        with torch.no_grad():
            dummy = torch.zeros(1, 64, Fg, Fg)
            pos_hw = sam.memory_encoder.position_encoding(dummy)
            _, per_pos = sam.spatial_perceiver(dummy, pos_hw)  # (1,512,64)
        tpos = sam.maskmem_tpos_enc[:, 0, 0, :]  # (num_maskmem, 64)
        nm = sam.num_maskmem
        slot_pos = []
        for _ in range(num_cond):  # t_pos = 0  -> index nm-1
            slot_pos.append(per_pos[0] + tpos[nm - 1])
        for k in range(num_recent):  # newest has t_pos = nm-1 -> index 0
            slot_pos.append(per_pos[0] + tpos[k])
        ptr_pos = torch.zeros(num_ptr * TOKENS_PER_PTR, 64)  # add_tpos_enc_to_obj_ptrs = False
        self.register_buffer("memory_pos", torch.cat(slot_pos + [ptr_pos], 0)[None].clone())

        # RoPE tables. queries: F x F grid. keys: 16x16 grid on the 2-D latents of each
        # spatial slot; global latents and pointer tokens are not rotated (cos=1, sin=0).
        cq, sq = rope_tables(axial_angles(256, Fg, Fg))
        self.register_buffer("cos_q", cq)
        self.register_buffer("sin_q", sq)
        ck2d, sk2d = rope_tables(axial_angles(256, 16, 16))
        ones, zeros = torch.ones(256, 256), torch.zeros(256, 256)
        ck = torch.cat([torch.cat([ones, ck2d]) for _ in range(self.S)] + [torch.ones(num_ptr * TOKENS_PER_PTR, 256)])
        sk = torch.cat([torch.cat([zeros, sk2d]) for _ in range(self.S)] + [torch.zeros(num_ptr * TOKENS_PER_PTR, 256)])
        self.register_buffer("cos_k", ck)
        self.register_buffer("sin_k", sk)

    def key_bias(self, slot_valid):
        spatial = slot_valid[:, : self.S].unsqueeze(-1).expand(-1, -1, TOKENS_PER_MEM).reshape(1, -1)
        ptr = slot_valid[:, self.S :].unsqueeze(-1).expand(-1, -1, TOKENS_PER_PTR).reshape(1, -1)
        valid = torch.cat([spatial, ptr], 1)  # (1, Lk)
        return ((1.0 - valid) * MASK_NEG).unsqueeze(1)  # (1, 1, Lk)

    def cross(self, layer, tgt, memory, bias):
        t2 = layer.norm2(tgt)
        k_in = memory + self.memory_pos  # pos_enc_at_cross_attn_keys=True, queries: no pos
        return tgt + attend(layer.cross_attn_image, t2, k_in, memory, self.cos_q, self.sin_q, self.cos_k, self.sin_k, bias)

    @staticmethod
    def mlp(layer, tgt):
        t2 = layer.norm3(tgt)
        return tgt + layer.linear2(layer.activation(layer.linear1(t2)))

    def forward(self, tgt0, memory, slot_valid):
        ma = self.sam.memory_attention
        bias = self.key_bias(slot_valid)
        l0, l1 = ma.layers[0], ma.layers[1]
        x = self.cross(l0, tgt0, memory, bias)  # layer-0 self-attn already done in encoder
        x = self.mlp(l0, x)
        t2 = l1.norm1(x)
        x = x + attend(l1.self_attn, t2, t2, t2, self.cos_q, self.sin_q, self.cos_q, self.sin_q)
        x = self.cross(l1, x, memory, bias)
        x = self.mlp(l1, x)
        x = ma.norm(x)
        return x.transpose(1, 2).reshape(1, 256, self.Fg, self.Fg)


# --------------------------------------------------------------------------- #
# 3) Prompt encoder + mask decoder + object pointer head
# --------------------------------------------------------------------------- #
class EdgeTAMMaskDecoder(nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.sam = sam
        pe = sam.sam_prompt_encoder
        with torch.no_grad():
            self.register_buffer("image_pe", pe.get_dense_pe().clone())
            Fg = sam.image_size // 16
            dense = pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, -1, Fg, Fg)
            self.register_buffer("dense", dense.clone())
        self.register_buffer("point_emb", torch.cat([e.weight for e in pe.point_embeddings], 0).detach().clone())  # (4,256)

    def embed_points(self, coords, labels):
        """Same as PromptEncoder._embed_points(pad=True) but written with
        where() instead of boolean index assignment (clean exported graph)."""
        pe = self.sam.sam_prompt_encoder
        coords = torch.cat([coords + 0.5, torch.zeros(1, 1, 2, dtype=coords.dtype)], 1)
        labels = torch.cat([labels, -torch.ones(1, 1, dtype=labels.dtype)], 1).unsqueeze(-1)  # (1,N+1,1)
        enc = pe.pe_layer._pe_encoding(coords / float(self.sam.image_size))
        emb = enc
        for i in range(4):
            emb = emb + (labels == float(i)).to(emb.dtype) * self.point_emb[i]
        return torch.where(labels == -1.0, pe.not_a_point_embed.weight.view(1, 1, -1), emb)

    def forward(self, pix_feat, hr0, hr1, point_coords, point_labels):
        sam = self.sam
        sparse = self.embed_points(point_coords, point_labels)
        masks, iou, mask_tokens, obj_score = sam.sam_mask_decoder.predict_masks(
            image_embeddings=pix_feat,
            image_pe=self.image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=self.dense,
            repeat_image=False,
            high_res_features=[hr0, hr1],
        )
        present = (obj_score > 0).to(obj_score.dtype)  # (1,1)
        masks = torch.where(present.view(1, 1, 1, 1) > 0, masks, torch.full_like(masks, NO_OBJ_SCORE))
        ptr = sam.obj_ptr_proj(mask_tokens)  # (1,4,256) one pointer per mask token
        ptr = present.view(1, 1, 1) * ptr + (1.0 - present.view(1, 1, 1)) * sam.no_obj_ptr.view(1, 1, -1)
        return masks, iou, ptr, obj_score


# --------------------------------------------------------------------------- #
# 4) Memory encoder + spatial perceiver
# --------------------------------------------------------------------------- #
class EdgeTAMMemoryEncoder(nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.sam = sam
        with torch.no_grad():
            Fg = sam.image_size // 16
            self.register_buffer("pos_hw", sam.memory_encoder.position_encoding(torch.zeros(1, 64, Fg, Fg)).clone())

    def forward(self, feat, mask_logits, binarize):
        sam = self.sam
        S = sam.image_size
        hi = F.interpolate(mask_logits, size=(S, S), mode="bilinear", align_corners=False)
        b = binarize.view(1, 1, 1, 1)
        m = b * (hi > 0).to(hi.dtype) + (1.0 - b) * torch.sigmoid(hi)
        m = m * sam.sigmoid_scale_for_mem_enc + sam.sigmoid_bias_for_mem_enc
        me = sam.memory_encoder
        x = me.pix_feat_proj(feat) + me.mask_downsampler(m)
        x = me.out_proj(me.fuser(x))
        tokens, _ = sam.spatial_perceiver(x, self.pos_hw)
        return tokens  # (1, 512, 64)
