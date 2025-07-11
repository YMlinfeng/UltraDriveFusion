
'''
https://poe.com/chat/pfy977zt5w434rf2g4


'''
import os
import logging

DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from transformers import PretrainedConfig, PreTrainedModel

from magicdrivedit.acceleration.checkpoint import auto_grad_checkpoint
from magicdrivedit.acceleration.communications import gather_forward_split_backward, split_forward_gather_backward
from magicdrivedit.acceleration.parallel_states import get_sequence_parallel_group
from magicdrivedit.models.layers.blocks import (
    Attention,
    CaptionEmbedder,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    PositionEmbedding2D,
    MultiHeadAttention,
    SeqParallelMultiHeadAttention,
    SeqParallelMultiHeadCrossAttention,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_layernorm,
    t2i_modulate,
)
from magicdrivedit.registry import MODELS
from magicdrivedit.utils.ckpt_utils import load_checkpoint
from magicdrivedit.utils.misc import warn_once

from .embedder import MapControlTempEmbedding
from .utils import zero_module, load_module

def encode_box(self, bboxes, drop_mask):  # changed
        # >>> NEW: drop_mask 现在形状 [B, CT]
        B, CT, seq_len = bboxes['bboxes'].shape[:3]
        bbox_embedder_kwargs = {k: v.clone() for k, v in bboxes.items()}

        # 将帧 / 视角的保留信息传播到每个 box token
        drop_mask = repeat(drop_mask, "B CT -> B CT S", S=seq_len)  # [B, CT, seq_len]

        _null_mask = torch.ones_like(bbox_embedder_kwargs["masks"])
        _null_mask[bbox_embedder_kwargs["masks"] == 0] = 0
        _mask = torch.ones_like(bbox_embedder_kwargs["masks"])
        _mask[bbox_embedder_kwargs["masks"] == -1] = 0
        _mask[torch.logical_and(
            bbox_embedder_kwargs["masks"] == 1,
            drop_mask == 0,            # 仅对真实存在的 box 做随机丢弃
        )] = 0

        bbox_emb = self.bbox_embedder(
            bboxes=bbox_embedder_kwargs['bboxes'],
            classes=bbox_embedder_kwargs["classes"].type(torch.int32),
            null_mask=_null_mask,
            mask=_mask,
            box_latent=bbox_embedder_kwargs.get('box_latent', None),
        )                               # [B, CT, seq_len, hidden]
        return bbox_emb


def encode_cam(self, cam, embedder, drop_mask):
        # cam       : [B, CT, S, ...]
        # drop_mask : [B, CT]
        B, CT, S = cam.shape[:3]
        cam_flat  = rearrange(cam, "B CT S ... -> (B CT S) ...")   # 展平
        mask_flat = repeat(drop_mask, "B CT -> (B CT S)", S=S)     # 同步展平

        cam_emb, _ = embedder.embed_cam(cam_flat, mask_flat, T=CT, S=S)
        cam_emb    = rearrange(cam_emb, "(B CT S) ... -> B CT S ...", B=B, CT=CT, S=S)
        return cam_emb                                               # [B, CT, S, hidden]


def encode_cond_sequence(self, bbox, cams, rel_pos, y, mask,
                             drop_cond_mask, drop_frame_mask):
        """
        新版本：
        - 输入帧维 CT = NC × T_original (=48)
        - 返回形状 [B, CT, L_cond, hidden]
        """
        B, CT = cams.shape[:2]
        T_orig = drop_frame_mask.size(1)
        NC = CT // T_orig if T_orig > 0 else 1                     # 由 CT 与原始 T 推断

        cond = []

        # ---------- 文本 ----------
        y, _ = self.encode_text(y, mask, drop_cond_mask)           # [B, L_txt, D]
        y = repeat(y, "B L D -> B CT L D", CT=CT)                  # 按 CT 帧复制
        cond.append(y)

        # ---------- 3D Box ----------
        if bbox is not None:
            drop_box_mask = repeat(drop_cond_mask, "B -> B CT") & \
                            repeat(drop_frame_mask, "B T -> B (T NC)", NC=NC)
            bbox_emb = self.encode_box(bbox, drop_mask=drop_box_mask)   # [B, CT, L_box, D]
            bbox_emb = self.base_token[None, None, None] + bbox_emb
            cond.append(bbox_emb)

        # ---------- Camera & Frame ----------
        cam_emb = self.encode_cam(cams[:, :1], self.camera_embedder,     # 仅第一帧
                                  repeat(drop_cond_mask, "B -> B 1"))    # [B, 1, S, D]
        frame_emb = self.encode_cam(rel_pos, self.frame_embedder,
                                    repeat(drop_frame_mask, "B T -> B (T NC)", NC=NC))
        cam_emb   = self.base_token[None, None, None] + cam_emb
        frame_emb = self.base_token[None, None, None] + frame_emb

        cam_emb   = repeat(cam_emb, 'B 1 S ... -> B CT S ...', CT=CT)    # time-align
        cond.extend([frame_emb, cam_emb])

        # ---------- 合并 ----------
        cond = torch.cat(cond, dim=2)    # [B, CT, L_all, hidden]
        return cond, None


def forward(self, x, timestep, y, maps, bbox, cams, rel_pos, fps,
                height, width, drop_cond_mask=None, drop_frame_mask=None,
                mv_order_map=None, t_order_map=None, mask=None, x_mask=None,
                **kwargs):

        dtype = self.x_embedder.proj.weight.dtype
        B, _, real_T, _, _ = x.size()                    # >>> NEW: 不含 NC
        NC = 1                                           # >>> NEW: 已将多目折叠到时间维

        # 保持旧的默认 mask 行为
        if drop_cond_mask is None:
            drop_cond_mask = torch.ones((B), device=x.device, dtype=x.dtype)
        if drop_frame_mask is None:
            drop_frame_mask = torch.ones((B, real_T), device=x.device, dtype=x.dtype)

        x = x.to(dtype)                                  # [B, C, T', H, W]
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        # === 1.1 无需再拆分 NC ===
        # 旧行: x = rearrange(x, "B (C NC) T ... -> (B NC) C T ...", NC=NC)

        # 下面流程（patchify/位置嵌入/Transformer）保持不变
        ...
        # (其余代码无需改动，直到 FinalLayer 调用处)

        # === 3.1 Final Layer ===
        x = self.final_layer(
            x,
            t,                                           # >>> NEW: 不再 repeat
            x_mask,
            t0 if x_mask is not None else None,
            T, S,
        )
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

        x = x.to(torch.float32)

        # 旧行: x = rearrange(x, "(B NC) C T ... -> B (C NC) T ...", NC=NC)
        # >>> NEW: 直接返回 [B, C_out, T', H, W]
        return x

