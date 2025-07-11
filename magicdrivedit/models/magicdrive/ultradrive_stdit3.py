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


class MultiViewSTDiT3Block(nn.Module):
    """
    Adapt PixArt & STDiT3 block for multiview generation in MagicDrive.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flash_attn=False,
        enable_xformers=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        sequence_parallelism_temporal=True,
        # stdit3
        rope=None,
        qk_norm=False,
        temporal=False,
        # multiview params
        is_control_block=False,
        use_st_cross_attn=False,
        skip_cross_view=False,
    ):
        super().__init__()
        self.temporal = temporal
        self.is_control_block = is_control_block
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn
        self.enable_sequence_parallelism = enable_sequence_parallelism

        assert not use_st_cross_attn, "STDiT3 have temporal downsample, this means nothing."
        if use_st_cross_attn:
            assert not enable_sequence_parallelism or not sequence_parallelism_temporal
        self.use_st_cross_attn = use_st_cross_attn
        self.skip_cross_view = skip_cross_view or self.temporal
        # `attn_cls` is for self-attn (only one input).
        if enable_sequence_parallelism:
            attn_cls = fmha_cls = SeqParallelMultiHeadAttention
            mha_cls = SeqParallelMultiHeadCrossAttention
        else:
            attn_cls = fmha_cls = MultiHeadAttention
            mha_cls = MultiHeadCrossAttention

        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        if temporal:
            _this_attn_cls = attn_cls if sequence_parallelism_temporal else Attention
        else:
            _this_attn_cls = fmha_cls if use_st_cross_attn else attn_cls
        self.attn = _this_attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            rope=rope,
            enable_flash_attn=enable_flash_attn,
            enable_xformers=enable_xformers,
            is_cross_attention=use_st_cross_attn,
        )

        # TODO: if split on T, we should also split conditions.
        # splits on `head_num` for conditions is performed in `SeqParallelMultiHeadCrossAttention`
        _this_attn_cls = MultiHeadCrossAttention if sequence_parallelism_temporal else mha_cls
        self.cross_attn = _this_attn_cls(hidden_size, num_heads)

        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )

        if not self.skip_cross_view:
            self.norm3 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
            # if split T, this is local attn; if split S, need full parallel.
            _this_attn_cls = Attention if sequence_parallelism_temporal else fmha_cls
            self.cross_view_attn = _this_attn_cls(
                hidden_size,
                num_heads=num_heads,
                qk_norm=True,
                enable_flash_attn=enable_flash_attn,
                enable_xformers=enable_xformers,
                is_cross_attention=True,
            )
            self.mva_proj = zero_module(nn.Linear(hidden_size, hidden_size))
        else:
            self.mva_proj = None

        # other helpers
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)
        if not self.skip_cross_view:
            self.scale_shift_table_mva = nn.Parameter(torch.randn(3, hidden_size) / hidden_size**0.5)
        if is_control_block:
            self.after_proj = zero_module(nn.Linear(hidden_size, hidden_size))
        else:
            self.after_proj = None

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def _construct_attn_input_from_map(self, h, order_map: dict, cat_seq=False):
        """
        Produce the inputs for the cross-view attention layer.

        Args:
            h (torch.Tensor): The hidden state of shape: [B, N, THW, self.hidden_size],
                              where T is the number of time frames and N the number of cameras.
            order_map (dict): key for query index, values for kv indexes.
            cat_seq (bool): if True, cat kv in seq length rather than batch size.
        Returns:
            h_q (torch.Tensor): The hidden state for the target views
            h_kv (torch.Tensor): The hidden state for the neighboring views
            back_order (torch.Tensor): The camera index for each of target camera in h_q
        """
        B = len(h)
        h_q, h_kv, back_order = [], [], []

        for target, values in order_map.items():
            if cat_seq:
                h_q.append(h[:, target])
                h_kv.append(torch.cat([h[:, value] for value in values], dim=1))
                back_order += [target] * B
            else:
                for neighbor in values:
                    h_q.append(h[:, target])
                    h_kv.append(h[:, neighbor])
                    back_order += [target] * B

        h_q = torch.cat(h_q, dim=0)
        h_kv = torch.cat(h_kv, dim=0)
        back_order = torch.LongTensor(back_order)

        return h_q, h_kv, back_order

    def forward(
        self,
        x,
        y,
        t,  # this t
        mask=None,  # text mask
        x_mask=None,  # temporal mask
        t0=None,  # t with timestamp=0, for x_mask
        # dim param, we need them for dynamic input size
        T=None,  # number of frames
        S=None,  # number of pixel patches
        NC=None,  # number of cameras
        # attn indexes, we need them for dynamic camera num/T
        mv_order_map=None,
        t_order_map=None,
    ):

        B, N, C = x.shape  # [6, 350, 1152]
        assert (N == T * S) and (B % NC == 0)
        b = B // NC

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = repeat(
            self.scale_shift_table[None] + t.reshape(b, 6, -1),
            "b ... -> (b NC) ...", NC=NC,
        ).chunk(6, dim=1)
        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = repeat(
                self.scale_shift_table[None] + t0.reshape(b, 6, -1),
                "b ... -> (b NC) ...", NC=NC,
            ).chunk(6, dim=1)

        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero, scale_msa_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        ######################
        # attention
        ######################
        if self.temporal:
            x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
            x_m = self.attn(x_m)
            x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
        else:
            if self.use_st_cross_attn:
                # "(b f n) d c -> (b n) f d c",
                x_st = rearrange(x_m, "B (T S) C -> B T S C", T=T, S=S)
                # this index is for kv pair, your dataloader should make it consistent.
                x_q, x_kv, back_order = self._construct_attn_input_from_map(
                    x_st, t_order_map, cat_seq=True)
                st_attn_raw_output = self.attn(x_q, x_kv)
                st_attn_output = torch.zeros_like(x_st)
                for frame_i in range(T):
                    attn_out_mt = rearrange(
                        st_attn_raw_output[back_order == frame_i],
                        '(n b) ... -> b n ...', b=B)
                    st_attn_output[:, frame_i] = torch.sum(attn_out_mt, dim=1)
                x_m = rearrange(st_attn_output, "B T S C -> B (T S) C")
            else:
                x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
                x_m = self.attn(x_m)
                x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)

        # modulate (attention)
        x_m_s = gate_msa * x_m
        if x_mask is not None:
            x_m_s_zero = gate_msa_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + self.drop_path(x_m_s)

        ######################
        # cross attn
        ######################
        assert mask is None
        if y.shape[1] == 1:
            x_c = self.cross_attn(x, y[:, 0], mask)
        elif y.shape[1] == T:
            x_c = rearrange(x, "B (T S) C -> (B T) S C", T=T, S=S)
            y_c = rearrange(y, "B T L C -> (B T) L C", T=T)
            x_c = self.cross_attn(x_c, y_c, mask)
            x_c = rearrange(x_c, "(B T) S C -> B (T S) C", T=T, S=S)
        else:
            raise RuntimeError(f"unsupported y.shape[1] = {y.shape[1]}")

        # residual, we skip drop_path here
        x = x + x_c

        ######################
        # multi-view cross attention
        ######################
        if not self.skip_cross_view:
            assert mv_order_map is not None
            # here we re-use the first 3 parameters from t and t0
            shift_mva, scale_mva, gate_mva = repeat(
                self.scale_shift_table_mva[None] + t[:, :3].reshape(b, 3, -1),
                "b ... -> (b NC) ...", NC=NC,
            ).chunk(3, dim=1)
            if x_mask is not None:
                shift_mva_zero, scale_mva_zero, gate_mva_zero = repeat(
                    self.scale_shift_table_mva[None] + t0[:, :3].reshape(b, 3, -1),
                    "b ... -> (b NC) ...", NC=NC,
                ).chunk(3, dim=1)

            x_v = t2i_modulate(self.norm3(x), shift_mva, scale_mva)
            if x_mask is not None:
                x_v_zero = t2i_modulate(self.norm3(x), shift_mva_zero, scale_mva_zero)
                x_v = self.t_mask_select(x_mask, x_v, x_v_zero, T, S)

            # Prepare inputs for multiview cross attention
            x_mv = rearrange(x_v, "(B NC) (T S) C -> (B T) NC S C", NC=NC, T=T)
            x_targets, x_neighbors, cam_order = self._construct_attn_input_from_map(
                x_mv, mv_order_map, cat_seq=False)
            # multi-view cross attention forward with batched neighbors
            cross_view_attn_output_raw = self.cross_view_attn(
                x_targets, x_neighbors)
            # arrange output tensor for sum over neighbors
            cross_view_attn_output = torch.zeros_like(x_mv)

            # cross_view_attn_output_raw [400, 350, 1152] t=20 b=1 ， c=1152
            for cam_i in range(NC):
                attn_out_mv = rearrange(
                    cross_view_attn_output_raw[cam_order == cam_i],
                    "(n_neighbors b) ... -> b n_neighbors ...",
                    b=B // NC * T,
                )
                cross_view_attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
            cross_view_attn_output = rearrange(
                cross_view_attn_output, "(B T) NC S C -> (B NC) (T S) C", T=T)

            # modulate (cross-view attention)
            x_v_s = gate_mva * cross_view_attn_output
            if x_mask is not None:
                x_v_s_zero = gate_mva_zero * cross_view_attn_output
                x_v_s = self.t_mask_select(x_mask, x_v_s, x_v_s_zero, T, S)

            # residual
            x_v_s = self.mva_proj(self.drop_path(x_v_s))
            x = x + x_v_s

        ######################
        # MLP
        ######################
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # MLP
        x_m = self.mlp(x_m)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m
        if x_mask is not None:
            x_m_s_zero = gate_mlp_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + self.drop_path(x_m_s)

        if self.is_control_block:
            x_skip = self.after_proj(x)
            return x, x_skip
        else:
            return x


class UltraDriveSTDiT3Config(PretrainedConfig):
    model_type = "MagicDriveSTDiT3"

    def __init__(
        self,
        input_size=(1, 32, 32),
        input_sq_size=512,
        force_pad_h_for_sp_size=None,
        simulate_sp_size=[],
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path: float = 0.0,
        caption_channels=4096,
        model_max_length=300,
        qk_norm=True,
        enable_flash_attn=False,
        enable_xformers=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        freeze_y_embedder=False,
        # magicdrive
        with_temp_block=True,
        freeze_x_embedder=False,
        freeze_old_embedder=False,
        freeze_temporal_blocks=False,
        freeze_old_params=False,
        zero_and_train_embedder=None,
        only_train_base_blocks=False,
        only_train_temp_blocks=False,
        qk_norm_trainable=False,
        sequence_parallelism_temporal=False,
        control_depth=13,
        use_x_control_embedder=False,
        use_st_cross_attn=False,
        uncond_cam_in_dim=(3, 7),
        cam_encoder_cls=None,
        cam_encoder_param={},
        bbox_embedder_cls=None,
        bbox_embedder_param={},
        map_embedder_cls=None,
        map_embedder_param={},
        map_embedder_downsample_rate=4,
        micro_frame_size=17,
        control_skip_cross_view=True,
        control_skip_temporal=True,
        **kwargs,
    ):
        self.input_size = input_size
        self.input_sq_size = input_sq_size
        self.force_pad_h_for_sp_size = force_pad_h_for_sp_size
        self.simulate_sp_size = simulate_sp_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.pred_sigma = pred_sigma
        self.drop_path = drop_path
        self.caption_channels = caption_channels
        self.model_max_length = model_max_length
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.freeze_y_embedder = freeze_y_embedder
        # magicdrive
        self.with_temp_block = with_temp_block
        self.freeze_x_embedder = freeze_x_embedder
        self.freeze_old_embedder = freeze_old_embedder
        self.freeze_temporal_blocks = freeze_temporal_blocks
        self.freeze_old_params = freeze_old_params
        self.zero_and_train_embedder = zero_and_train_embedder
        self.only_train_base_blocks = only_train_base_blocks
        self.only_train_temp_blocks = only_train_temp_blocks
        self.qk_norm_trainable = qk_norm_trainable
        self.enable_xformers = enable_xformers
        self.sequence_parallelism_temporal = sequence_parallelism_temporal
        self.control_depth = control_depth
        self.use_x_control_embedder = use_x_control_embedder
        self.use_st_cross_attn = use_st_cross_attn
        self.uncond_cam_in_dim = uncond_cam_in_dim
        self.cam_encoder_cls = cam_encoder_cls
        self.cam_encoder_param = cam_encoder_param
        self.bbox_embedder_cls = bbox_embedder_cls
        self.bbox_embedder_param = bbox_embedder_param
        self.map_embedder_cls = map_embedder_cls
        self.map_embedder_param = map_embedder_param
        self.map_embedder_downsample_rate = map_embedder_downsample_rate
        self.micro_frame_size = micro_frame_size
        self.control_skip_cross_view = control_skip_cross_view
        self.control_skip_temporal = control_skip_temporal
        super().__init__(**kwargs)


class UltraDriveSTDiT3(PreTrainedModel):

    config_class = UltraDriveSTDiT3Config

    def __init__(self, config: UltraDriveSTDiT3Config):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.pred_sigma else config.in_channels

        # model size related
        self.depth = config.depth
        self.control_depth = config.control_depth
        self.mlp_ratio = config.mlp_ratio
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        # computation related
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_xformers = config.enable_xformers
        self.enable_layernorm_kernel = config.enable_layernorm_kernel
        self.enable_sequence_parallelism = config.enable_sequence_parallelism
        self.sequence_parallelism_temporal = config.sequence_parallelism_temporal

        # input size related
        self.patch_size = config.patch_size
        self.input_sq_size = config.input_sq_size
        self.pos_embed = PositionEmbedding2D(self.hidden_size)
        self.rope = RotaryEmbedding(dim=self.hidden_size // self.num_heads)
        self.force_pad_h_for_sp_size = config.force_pad_h_for_sp_size
        self.simu_sp_size = config.simulate_sp_size

        # embedding
        self.x_embedder = PatchEmbed3D(self.patch_size, self.in_channels, self.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=config.hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.model_max_length,
        )
        self.fps_embedder = SizeEmbedder(self.hidden_size)

        if config.use_x_control_embedder:
            self.x_control_embedder = PatchEmbed3D(self.patch_size, self.in_channels, self.hidden_size)
        else:
            self.x_control_embedder = None
        # base_token, should not be trainable
        self.register_buffer("base_token", torch.randn(self.hidden_size))
        # init camera encoder
        self.camera_embedder = load_module(config.cam_encoder_cls)(
            out_dim=config.hidden_size, **config.cam_encoder_param)
        # init frame encoder
        self.frame_embedder = load_module(config.frame_emb_cls)(
            out_dim=config.hidden_size, **config.frame_emb_param)
        # init bbox encoder
        self.bbox_embedder = load_module(config.bbox_embedder_cls)(
            **config.bbox_embedder_param)
        # init map 2D encoder
        self.controlnet_cond_embedder = load_module(config.map_embedder_cls)(
            conditioning_embedding_channels=self.hidden_size // 2,
            **config.map_embedder_param,
        )
        self.micro_frame_size = config.micro_frame_size  # should be the same as vae
        self.controlnet_cond_embedder_temp = MapControlTempEmbedding(
            self.hidden_size, config.map_embedder_downsample_rate)
        self.controlnet_cond_patchifier = PatchEmbed3D(self.patch_size, self.hidden_size, self.hidden_size)

        # base blocks
        drop_path = [x.item() for x in torch.linspace(0, config.drop_path, self.depth)]
        self.base_blocks_s = nn.ModuleList(
            [
                MultiViewSTDiT3Block(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flash_attn=self.enable_flash_attn,
                    enable_xformers=self.enable_xformers,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    enable_sequence_parallelism=self.enable_sequence_parallelism,
                    sequence_parallelism_temporal=self.sequence_parallelism_temporal,
                    # stdit3
                    qk_norm=config.qk_norm,
                    # multiview params
                    use_st_cross_attn=config.use_st_cross_attn,
                    # skip_cross_view=True,  # just for debug
                )
                for i in range(self.depth)
            ]
        )
        if config.with_temp_block:
            self.base_blocks_t = nn.ModuleList(
                [
                    MultiViewSTDiT3Block(
                        hidden_size=self.hidden_size,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        drop_path=drop_path[i],
                        enable_flash_attn=self.enable_flash_attn,
                        enable_xformers=self.enable_xformers,
                        enable_layernorm_kernel=self.enable_layernorm_kernel,
                        enable_sequence_parallelism=self.enable_sequence_parallelism,
                        sequence_parallelism_temporal=self.sequence_parallelism_temporal,
                        # stdit3
                        qk_norm=config.qk_norm,
                        temporal=True,
                        rope=self.rope.rotate_queries_or_keys,
                    )
                    for i in range(self.depth)
                ]
            )
        else:
            self.base_blocks_t = None

        # control blocks
        self.before_proj = zero_module(nn.Linear(self.hidden_size, self.hidden_size))
        drop_path = [x.item() for x in torch.linspace(0, config.drop_path, self.control_depth)]
        self.control_blocks_s = nn.ModuleList(
            [
                MultiViewSTDiT3Block(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flash_attn=self.enable_flash_attn,
                    enable_xformers=self.enable_xformers,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    enable_sequence_parallelism=self.enable_sequence_parallelism,
                    sequence_parallelism_temporal=self.sequence_parallelism_temporal,
                    # stdit3
                    qk_norm=config.qk_norm,
                    # multiview params
                    is_control_block=True,
                    use_st_cross_attn=config.use_st_cross_attn,
                    skip_cross_view=config.control_skip_cross_view,
                )
                for i in range(self.control_depth)
            ]
        )
        if config.control_skip_temporal:
            self.control_blocks_t = None
        else:
            self.control_blocks_t = nn.ModuleList(
                [
                    MultiViewSTDiT3Block(
                        hidden_size=self.hidden_size,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        drop_path=drop_path[i],
                        enable_flash_attn=self.enable_flash_attn,
                        enable_xformers=self.enable_xformers,
                        enable_layernorm_kernel=self.enable_layernorm_kernel,
                        enable_sequence_parallelism=self.enable_sequence_parallelism,
                        sequence_parallelism_temporal=self.sequence_parallelism_temporal,
                        # stdit3
                        qk_norm=config.qk_norm,
                        temporal=True,
                        rope=self.rope.rotate_queries_or_keys,
                        # multiview params
                        is_control_block=True,
                    )
                    for i in range(self.control_depth)
                ]
            )

        # final layer
        self.final_layer = T2IFinalLayer(self.hidden_size, np.prod(self.patch_size), self.out_channels)

        self.initialize_weights()

        # set training status
        if config.freeze_y_embedder:
            for param in self.y_embedder.parameters():
                param.requires_grad = False
        if config.freeze_x_embedder:
            for param in self.x_embedder.parameters():
                param.requires_grad = False
        if config.freeze_old_embedder:
            for param in self.t_embedder.parameters():
                param.requires_grad = False
            for param in self.t_block.parameters():
                param.requires_grad = False
            for param in self.fps_embedder.parameters():
                param.requires_grad = False
        if config.freeze_temporal_blocks:
            for block in self.base_blocks_t:
                # freeze all
                for param in block.parameters():
                    param.requires_grad = False
                # but train cross_attn! NOTE: we may not need this.
                # for param in block.cross_attn.parameters():
                #     param.requires_grad = True
                
            if self.control_blocks_t is not None:
                for block in self.control_blocks_t:
                    for param in block.parameters():
                        param.requires_grad = False
                    # for param in block.cross_attn.parameters():
                    #     param.requires_grad = True

        # from magicdrive to video
        if config.only_train_temp_blocks:
            if not config.only_train_base_blocks:
                logging.warning("`only_train_temp_blocks` is only usable with `only_train_base_blocks`.")
        if config.only_train_base_blocks:
            # first freeze all
            for param in self.parameters():
                param.requires_grad = False
            
            # then open some
            if not config.only_train_temp_blocks:
                for param in self.base_blocks_s.parameters():
                    param.requires_grad = True
            if self.base_blocks_t is not None:
                for param in self.base_blocks_t.parameters():
                    param.requires_grad = True

            if self.control_blocks_t is not None:
                for param in self.control_blocks_t.parameters():
                    param.requires_grad = True
            
            # embedders
            # NOTE: embedder changed, do we need to change cross-attn in control
            # blocks? 
            for mod in [
                # self.camera_embedder,
                self.frame_embedder,
                self.bbox_embedder,
                self.controlnet_cond_embedder,
                self.controlnet_cond_embedder_temp,
                self.controlnet_cond_patchifier,
                self.before_proj,
                # self.x_control_embedder,
            ]:
                if mod is None:
                    continue
                for param in mod.parameters():
                    param.requires_grad = True

            assert config.zero_and_train_embedder is None
            assert not config.qk_norm_trainable
            assert not config.freeze_old_params
            return # ignore all others

        if config.freeze_old_params:
            for param in self.parameters():
                param.requires_grad = False

        # from pretrain to magicdrive control
        if config.zero_and_train_embedder is not None:
            for emb in config.zero_and_train_embedder:
                zero_module(getattr(self, emb).mlp[-1])
                for param in getattr(self, emb).parameters():
                    param.requires_grad = True

        if config.qk_norm_trainable:
            for name, param in self.named_parameters():
                if "q_norm" in name or "k_norm" in name:
                    logging.info(f"set {name} to trainable")
                    param.requires_grad = True

        # make sure all new parameters require grad
        # cross view attn
        for block in self.base_blocks_s:
            if hasattr(block, "cross_view_attn"):
                for param in block.norm3.parameters():
                    param.requires_grad = True
                for param in block.cross_view_attn.parameters():
                    param.requires_grad = True
                for param in block.mva_proj.parameters():
                    param.requires_grad = True
                block.scale_shift_table_mva.requires_grad = True

        # control blocks        
        for param in self.control_blocks_s.parameters():
            param.requires_grad = True
        if self.control_blocks_t is not None:
            for param in self.control_blocks_t.parameters():
                param.requires_grad = True
        
        # embedders
        for mod in [
            self.camera_embedder,
            self.frame_embedder,
            self.bbox_embedder,
            self.controlnet_cond_embedder,
            self.controlnet_cond_embedder_temp,
            self.controlnet_cond_patchifier,
            self.before_proj,
            self.x_control_embedder,
        ]:
            if mod is None:
                continue
            for param in mod.parameters():
                param.requires_grad = True

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # NOTE: some proj layers are zero-initialized on creating.
        def _zero_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.constant_(module.weight, 0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # new block in base
        for block in self.base_blocks_s:
            _zero_init(block.mva_proj)
            assert block.after_proj == None

        if self.base_blocks_t is not None:
            for block in self.base_blocks_t:
                assert block.mva_proj == None
                assert block.after_proj == None
                # Initialize temporal blocks
                _zero_init(block.attn.proj)
                _zero_init(block.cross_attn.proj)
                _zero_init(block.mlp.fc2.weight)
            logging.info("Your base_blocks_t uses zero init!")

        # control block
        for block in self.control_blocks_s:
            _zero_init(block.mva_proj)
            _zero_init(block.after_proj)
        if self.control_blocks_t is not None:
            for block in self.control_blocks_t:
                assert block.mva_proj == None
                _zero_init(block.after_proj)

        # self
        _zero_init(self.before_proj)

        # zero init embedder proj
        _zero_init(self.bbox_embedder.final_proj)
        _zero_init(self.camera_embedder.after_proj)
        _zero_init(self.frame_embedder.final_proj)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d): cr. PixArt
        w = self.controlnet_cond_patchifier.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize caption embedding MLP: cr. PixArt
        nn.init.normal_(self.bbox_embedder.mlp.fc1.weight, std=0.02)
        nn.init.normal_(self.bbox_embedder.mlp.fc2.weight, std=0.02)
        nn.init.normal_(self.frame_embedder.mlp.fc1.weight, std=0.02)
        nn.init.normal_(self.frame_embedder.mlp.fc2.weight, std=0.02)
        nn.init.normal_(self.camera_embedder.emb2token.weight, std=0.02)

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def sample_box_latent(self, n_boxes, generator=None):
        if self.bbox_embedder.mean_var is None:
            latent = None
        else:
            latent = torch.randn(
                (n_boxes, self.bbox_embedder.box_latent_shape[1]),
                generator=generator,
            )
        return latent

    def encode_text(self, y, mask=None, drop_cond_mask=None):
        # NOTE: we do not use y mask, but keep the batch dim.
        # NOTE: we do not use drop in y_embedder
        if drop_cond_mask is not None:
            y = self.y_embedder(y, False, force_drop_ids=1 - drop_cond_mask)  # [B, 1, N_token, C]
        else:
            y = self.y_embedder(y, False)  # [B, 1, N_token, C]
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            y_lens = [i + 1 for i in mask.sum(dim=1).tolist()]
            max_len = int(min(max(y_lens), y.shape[2]))  # we need min because of +1
            if drop_cond_mask is not None and not drop_cond_mask.all():  # on any drop, this should be the max
                assert max_len == y.shape[2]
            # y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.hidden_size)
            y = y.squeeze(1)[:, :max_len]
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1)
        return y, y_lens

    def encode_box(self, bboxes, drop_mask):  #! changed 兼容 CT=NC×T 的新时间维
        # B, T, seq_len = bboxes['bboxes'].shape[:3]
        # bbox_embedder_kwargs = {}
        # for k, v in bboxes.items():
        #     bbox_embedder_kwargs[k] = v.clone()

        # >>> NEW: drop_mask 现在形状 [B, CT]
        B, CT, seq_len = bboxes['bboxes'].shape[:3]
        bbox_embedder_kwargs = {k: v.clone() for k, v in bboxes.items()}
        # each key should have dim like: (b, seq_len, dim...)
        # bbox_embedder_kwargs["masks"]: 0 -> null, -1 -> mask, 1 -> keep
        # drop_mask: 0 -> mask, 1 -> keep

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


    def encode_cam(self, cam, embedder, drop_mask): #! 不再把 NC 打到 batch 维
        # B, T, S = cam.shape[:3]
        # NC = B // drop_mask.shape[0]
        # mask = repeat(drop_mask, "b T -> (b NC T S)", NC=NC, S=S)
        # cam = rearrange(cam, "B T S ... -> (B T S) ...")
        # cam_emb, _ = embedder.embed_cam(cam, mask, T=T, S=S)  # changed here
        # # cam_emb = rearrange(cam_emb, "(B T S) ... -> B T S ...", B=B, T=T, S=S)
        # return cam_emb
        
        # cam       : [B, CT, S, ...]
        # drop_mask : [B, CT]
        B, CT, S = cam.shape[:3]
        cam_flat  = rearrange(cam, "B CT S ... -> (B CT S) ...")   # 展平
        mask_flat = repeat(drop_mask, "B CT -> (B CT S)", S=S)     # 同步展平

        cam_emb, _ = embedder.embed_cam(cam_flat, mask_flat, T=CT, S=S)
        cam_emb    = rearrange(cam_emb, "(B CT S) ... -> B CT S ...", B=B, CT=CT, S=S)
        return cam_emb                                               # [B, CT, S, hidden]

    def encode_cond_sequence( #! 彻底去掉 (B·NC) 展开
            self, 
            bbox, 
            cams, 
            rel_pos, 
            y, 
            mask, 
            drop_cond_mask, 
            drop_frame_mask):  
        '''
        encode -> repeat -> concatenate
        """
        新版本：
        - 输入帧维 CT = NC × T_original (=48)
        - 返回形状 [B, CT, L_cond, hidden]
        """
        '''
        B, CT = cams.shape[:2]
        T_orig = drop_frame_mask.size(1) # # 仍是 8，NC 推断正确
        NC = CT // T_orig if T_orig > 0 else 1        # 由 CT 与原始 T 推断 NC = 6
        cond = []

        # encode y 文本
        # y, _ = self.encode_text(y, mask, drop_cond_mask)  # b, seq_len, dim = 2,38,1152
        y, _ = self.encode_text(y, mask, drop_cond_mask)           # [B, L_txt, D]
        y = repeat(y, "B L D -> B CT L D", CT=CT)                  # 按 CT 帧复制
        cond.append(y) # todo 原始文件中这里是注释掉的，奇怪

        # ---------- 3D Box ----------
        if bbox is not None:
            # drop_box_mask = repeat(drop_cond_mask, "B -> B CT", CT=CT) & \
            #                 repeat(drop_frame_mask, "B T -> B (T NC)", NC=NC)
            drop_cond_mask = repeat(drop_cond_mask, "B -> B CT", CT=CT)
            drop_frame_mask = repeat(drop_frame_mask, "B T -> B (T NC)", NC=NC)
            drop_box_mask = torch.logical_and(drop_cond_mask, drop_frame_mask) # 先转换成bool再按位与，太妙了
            bbox_emb = self.encode_box(bbox, drop_mask=drop_box_mask)   # [B, CT, L_box, D]
            bbox_emb = self.base_token[None, None, None] + bbox_emb
            cond.append(bbox_emb)

        # # encode cam, just take from first frame
        # cam_emb = self.encode_cam(
        #     # cams, self.camera_embedder, repeat(drop_cond_mask, "b -> b T", T=T))
        #     cams[:, 0:1], self.camera_embedder, repeat(drop_cond_mask, "b -> b T", T=1))
        # frame_emb = self.encode_cam(rel_pos, self.frame_embedder, drop_frame_mask)
        # cam_emb = rearrange(cam_emb, "(B 1 S) ... -> B 1 S ...", S=cams.shape[2])
        # # frame_emb = frame_emb.mean(1)  # pooled token
        # # zero proj on base token
        # cam_emb = self.base_token[None, None, None] + cam_emb
        # frame_emb = self.base_token[None, None, None] + frame_emb # 编码摄像头参数和 frame_emb（相对位置）分别通过 encode_cam 得到摄像头和帧嵌入，并加上 base_token

        # cam_emb = repeat(cam_emb, 'B 1 S ... -> B T S ...', T=frame_emb.shape[1])
        # y = repeat(y, "B ... -> B T ...", T=frame_emb.shape[1])
        # cond = [frame_emb, cam_emb, y] + cond

        # ---------- Camera & Frame ----------
        cam_emb = self.encode_cam(cams[:, :1], self.camera_embedder,     # 仅第一帧
                                  repeat(drop_cond_mask, "B -> B 1"))    # [B, 1, S, D]
        frame_emb = self.encode_cam(rel_pos, self.frame_embedder,
                                    repeat(drop_frame_mask, "B T -> B (T NC)", NC=NC))
        cam_emb   = self.base_token[None, None, None] + cam_emb
        frame_emb = self.base_token[None, None, None] + frame_emb

        cam_emb   = repeat(cam_emb, 'B 1 S ... -> B CT S ...', CT=CT)    # time-align
        cond.extend([frame_emb, cam_emb])

        # merge to one
        # # cond = torch.cat([frame_emb, cam_emb, y, bbox_emb], dim=2)  # B, T, len, dim
        # # # change me!
        # # cond = torch.cat([y, frame_emb, cam_emb], dim=1)  # B, len, dim
        # # return rearrange(cond, '(b NC) ... -> b NC ...', NC=NC)[:, 0], None
        # # cond = torch.cat(cond, dim=1)  # B, len, dim

        # cond = torch.cat(cond, dim=2)  # B=12, T=3, len=78, dim=1152 # 将所有条件（文本、bbox、摄像头、frame_emb）拼接在一起，通常输出一个张量形状为 [B, T, L, hidden_size]，这里 L 为各个条件 tokens 之和
        # return cond, 

        # ---------- 合并 ----------
        cond = torch.cat(cond, dim=2)    # [B, CT, L_all, hidden]
        return cond, None


    def encode_map(self, maps, NC, h_pad_size, x_shape): #! 貌似是可以兼容的
        """
        新数据流：NC 已经在 VAE 阶段折叠进时间维，因此
        这里只需返回形状   [B, C_embed, T_z, H_z, W_z]  即可。

        参数说明:
            maps       : [B, T_orig, C_map, H_in, W_in]
            NC         : 保留旧接口但仅支持 NC==1（若>1则报错）
            h_pad_size : 为 sequence-parallel 可能追加的高度补边 (patch 单位)
            x_shape    : 用于和视觉 latent 对齐的目标三维尺寸
        """
        if NC != 1:
            raise ValueError(
                "在“六目合并压缩”模式下，encode_map 不再支持 NC>1；"
                "多视角信息已在时间维展开。")

        # b, T = maps.shape[:2]
        B, T_orig = maps.shape[:2]
        
        # maps = rearrange(maps, "b T ... -> (b T) ...")
        # controlnet_cond = self.controlnet_cond_embedder(maps)
        # # map patchifier reshapes and forward -> format expected by nn.Conv3D
        # controlnet_cond = rearrange(controlnet_cond, "(b T) C ... -> b C T ...", T=T)
        
        # ---------------- Step-1: 空间 CNN + 下采样 ----------------
        maps_flat = rearrange(maps, "B T ... -> (B T) ...")               # (B·T), C, H, W
        controlnet_cond = self.controlnet_cond_embedder(maps_flat)        # ↓ 通道至 hidden
        controlnet_cond = rearrange(
            controlnet_cond,
            "(B T) C ... -> B C T ...",
            B=B, T=T_orig
        )   # [B, C, T, H', W']

        
        
        # if self.micro_frame_size is None:
        #     controlnet_cond = self.controlnet_cond_embedder_temp(controlnet_cond)
        # else:
        #     z_list = []
        #     for i in range(0, controlnet_cond.shape[2], self.micro_frame_size):
        #         x_z_bs = controlnet_cond[:, :, i: i + self.micro_frame_size]
        #         z = self.controlnet_cond_embedder_temp(x_z_bs)
        #         z_list.append(z)
        #     controlnet_cond = torch.cat(z_list, dim=2)
        
        # ---------------- Step-2: Temporal Downsample ------------------
        if self.micro_frame_size is None:
            controlnet_cond = self.controlnet_cond_embedder_temp(controlnet_cond)
        else:
            z_list = []
            for i in range(0, controlnet_cond.shape[2], self.micro_frame_size):
                z_chunk = controlnet_cond[:, :, i: i + self.micro_frame_size]
                z_list.append(self.controlnet_cond_embedder_temp(z_chunk))
            controlnet_cond = torch.cat(z_list, dim=2)                    # [B, C, T_z, H', W']
        
        
        # if controlnet_cond.shape[-3:] != x_shape[-3:]:
        #     # [-3:] for (T, H, W)
        #     warn_once(
        #         f"For x_shape = {x_shape[-3:]}, we interpolate map cond from "
        #         f"{controlnet_cond.shape[-3:]}"
        #     )
        #     if np.prod(x_shape[-3:]) > np.prod([33, 106, 200]) and controlnet_cond.shape[0] > 1:
        #         # slice batch
        #         _controlnet_cond = []
        #         for ci in range(controlnet_cond.shape[0]):
        #             _controlnet_cond.append(
        #                 F.interpolate(controlnet_cond[ci:ci + 1], x_shape[-3:])
        #             )
        #         controlnet_cond = torch.cat(_controlnet_cond, dim=0)
        #     else:
        #         if np.prod(x_shape[-3:]) > np.prod([33, 106, 200]):
        #             warn_once(f"shape={controlnet_cond.shape} cannot be splitted!")
        #         controlnet_cond = F.interpolate(controlnet_cond, x_shape[-3:])

        # ---------------- Step-3: 与视觉 latent 尺寸对齐 ----------------
        if controlnet_cond.shape[-3:] != x_shape[-3:]:
            warn_once(
                f"controlnet_cond {controlnet_cond.shape[-3:]} "
                f"≠ latent {x_shape[-3:]}, 将进行三线性插值对齐。")
            controlnet_cond = F.interpolate(controlnet_cond, x_shape[-3:])
        
        # ---------------- Step-4: Sequence-Parallel 填充 ----------------
        if h_pad_size > 0:
            hx_pad_size = h_pad_size * self.patch_size[1]
            # pad c along the H dimension
            controlnet_cond = F.pad(controlnet_cond, (0, 0, 0, hx_pad_size))
        
        # ---------------- Step-5: Patchify 到 token 维 ------------------
        controlnet_cond = self.controlnet_cond_patchifier(controlnet_cond)  # [B, N_token, hidden]
       
        controlnet_cond = repeat(controlnet_cond, "b ... -> (b NC) ...", NC=NC) # 现在 NC==1，直接返回即可
        return controlnet_cond

    def prepare_text_embedding(self, text_encoder):
        @torch.no_grad()
        def text_to_embedding(text):
            ret = text_encoder.encode(text)
            hidden_state, _ = self.encode_text(ret['y'], mask=None)
            return hidden_state[:, :int(ret['mask'].sum(dim=1))]
        _training = self.training
        self.training = False
        self.bbox_embedder.prepare(text_to_embedding)
        self.base_token[:] = text_to_embedding("").squeeze()
        self.training = _training

    def forward(self, x, timestep, y, maps, bbox, cams, rel_pos, fps,
                height, width, drop_cond_mask=None, drop_frame_mask=None,
                mv_order_map=None, t_order_map=None, mask=None, x_mask=None,
                **kwargs):
        """
        Forward pass of MagicDrive.
        假设输入参数说明如下：
            x：经过 VAE 编码、形状大致为 [B, C×NC, T, H, W]
            B：batch 大小
            NC：摄像头数量
            C：每个摄像头的 latent 通道数
            T：帧数
            H, W：空间尺寸（已经下采样）
            timestep：时间步（用于生成噪声权重），标量或 [B] 张量
            y：文本描述（未经过编码），后续通过 CaptionEmbedder 编码为 [B, 1, N_token, hidden_size]
            maps：地图或 BEV 信息，形状 [B, T, ...]
            bbox、cams、rel_pos：分别是 3D 边界框、相机参数（内外参）和帧间变换，用于条件约束
            fps、height、width：视频帧率、原始图像高度和宽度
            另外还有 drop_cond_mask、drop_frame_mask、mv_order_map、t_order_map、mask、x_mask 等辅助条件和 mask 信息
        """
        dtype = self.x_embedder.proj.weight.dtype
        B, _, real_T, _, _ = x.size()                    # >>> NEW: 不含 NC
        NC = 1                                           # >>> NEW: 已将多目折叠到时间维

        if drop_cond_mask is None:  # camera
            drop_cond_mask = torch.ones((B), device=x.device, dtype=x.dtype)
        if drop_frame_mask is None:  # box & rel_pos
            drop_frame_mask = torch.ones((B, real_T), device=x.device, dtype=x.dtype)
        if False:
        # if mv_order_map is None:
            NC = 1
        else:
            NC = len(mv_order_map)

        x = x.to(dtype)
        # HACK: to use scheduler, we never assume NC with C

        # x = rearrange(x, "B (C NC) T ... -> (B NC) C T ...", NC=NC)
        timestep = timestep.to(dtype)
        y = y.to(dtype)


        # === 1.1 分patch无需再拆分 NC ===
        # 下面流程（patchify/位置嵌入/Transformer）保持不变
        # (其余代码无需改动，直到 FinalLayer 调用处)
        _, _, Tx, Hx, Wx = x.size()
        x_in_shape = x.shape  # before pad 原始输入 x 尺寸，用于后续 unpatchify 调整
        T, H, W = self.get_dynamic_size(x) # 根据输入 x 得到下采样后的 T、H、W 值，然后通过 pos_embed 计算位置嵌入，再与 x 的空间嵌入结合
        S = H * W

        # adjust for sequence parallelism 这一大坨都不用改
        # we need to ensure H * W is divisible by sequence parallel size
        # for simplicity, we can adjust the height to make it divisible
        h_pad_size = 0
        if self.training:
            _simu_sp_size = self.simu_sp_size
        else:
            if len(self.simu_sp_size) > 0:
                warn_once(f"We will ignore `simu_sp_size` if not training.")
            _simu_sp_size = []
        if self.force_pad_h_for_sp_size is not None:
            if S % self.force_pad_h_for_sp_size != 0:
                h_pad_size = self.force_pad_h_for_sp_size - H % self.force_pad_h_for_sp_size
                warn_once(
                    f"Your input shape {x.shape} was rounded into {(T, H, W)}. "
                    f"With force_pad_h_for_sp_size={self.force_pad_h_for_sp_size}, "
                    f"it is padded by H with {h_pad_size}. "
                )
        elif len(_simu_sp_size) > 0:
            if self.enable_sequence_parallelism and not self.sequence_parallelism_temporal:
                # make sure the simulated is greater than real sp_size
                sp_size = dist.get_world_size(get_sequence_parallel_group())
                possible_sp_size = []
                for _sp_size in _simu_sp_size:
                    if _sp_size >= sp_size:
                        possible_sp_size.append(_sp_size)
            else:
                possible_sp_size = _simu_sp_size
            # random pick one
            simu_sp_size = random.choice(possible_sp_size)
            if S % simu_sp_size != 0:
                h_pad_size = simu_sp_size - H % simu_sp_size
            if h_pad_size > 0:
                warn_once(
                    f"Your input shape {x.shape} was rounded into {(T, H, W)}. "
                    f"For simu_sp_size={simu_sp_size} out of {possible_sp_size}, "
                    f"it is padded by H with {h_pad_size}. "
                    "Please pay attention to potential mismatch between w/ and w/o sp."
                )
        elif self.enable_sequence_parallelism and not self.sequence_parallelism_temporal:
            sp_size = dist.get_world_size(get_sequence_parallel_group())
            if S % sp_size != 0:
                h_pad_size = sp_size - H % sp_size
            if h_pad_size > 0:
                warn_once(
                    f"Your input shape {x.shape} was rounded into {(T, H, W)}. "
                    f"For sp_size={sp_size}, it is padded by H with {h_pad_size}. "
                    "Please pay attention to potential mismatch between w/ and w/o sp."
                )

        if h_pad_size > 0:
            # pad x along the H dimension
            hx_pad_size = h_pad_size * self.patch_size[1]
            x = F.pad(x, (0, 0, 0, hx_pad_size))
            # adjust parameters
            H += h_pad_size
            S = H * W
            if self.enable_sequence_parallelism and not self.sequence_parallelism_temporal:
                sp_size = dist.get_world_size(get_sequence_parallel_group())
                assert S % sp_size == 0, f"S={S} should be divisible by {sp_size}!"

        # 1.2 PE 计算
        base_size = round(S**0.5)
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size) # ([1, 1350, C=1152])???

        # === 1.3 get timestep embed ===
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]=[2, 1152],t_embedder 将 time step 嵌入为隐藏状态，再加上 fps 嵌入补充运动信息
        fps = self.fps_embedder(fps.unsqueeze(1), B)  # [B, C]=[2, 1152]
        t = t + fps
        t_mlp = self.t_block(t)# 生成扩展后的 t_mlp，形状为 [B, 6×hidden_size],t_block （一个线性+激活）将 t 转换为用于控制每层的“shift/scale/gate”参数，这个 t_mlp 会在后续 Transformer 块中拆分为 6 部分
        t0 = t0_mlp = None
        if x_mask is not None: # 如果存在 x_mask，则对全 0 的 t（即 t0）也进行嵌入，用于在 mask 部分替换原特征
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block(t0)

        # 1.4. 条件信息编码（文本、Bounding Box、摄像头、frame_emb）
        # === get y embed ===
        # we need to remove the T dim in y
        # rel_pos & bbox: T -> 1
        # cam: just take first frame
        y, y_lens = self.encode_cond_sequence(
            bbox, cams, rel_pos, y, mask, drop_cond_mask, drop_frame_mask)  # (B, L, D)
        if y.shape[1] != T and y.shape[1] > 1:
            warn_once(f"Got y length {y.shape[1]}, will interpolate to {T}.")
            seq_len = y.shape[2]
            y = rearrange(y, "B T L D -> B (L D) T")
            y = F.interpolate(y, T)
            y = rearrange(y, "B (L D) T -> B T L D", L=seq_len)

        # 1.5. 地图条件编码
        c = self.encode_map(maps, NC, h_pad_size, x_in_shape)
        c = rearrange(c, "B (T S) C -> B T S C", T=T) # ([12, 3, 1350, 1152])

        # === 1.6. 图像嵌入（x_embedder）与控制条件融合 get x embed ===
        x_b = self.x_embedder(x)  # # 得到视觉 token, 原始形状：[B, (T S), hidden_size]=[12, 16, 3, 53, 100]->[B, N, C]
        x_b = rearrange(x_b, "B (T S) C -> B T S C", T=T, S=S)
        x_b = x_b + pos_emb # ([12, 3, 1350, 1152]) #! 这里的PE不用变

        if self.x_control_embedder is None:
            x_c = x_b
        else:
            x_c = self.x_control_embedder(x)  # controlnet has another embedder!
            x_c = rearrange(x_c, "B (T S) C -> B T S C", T=T, S=S)
            x_c = x_c + pos_emb

        c = x_c + self.before_proj(c)  # first block connection # 将地图条件经过投影，与控制条件相加   这里 before_proj 仅做一个线性映射，其设计目的是将地图条件与图像条件维度对齐，不需要详细讨论其内部参数初始化
        x = x_b # [12, 3, 1350, 1152]

        # shard over the sequence dim if sp is enabled
        # 根据是否启用了 sequence parallelism 对 x 和 c 在序列维度上做拆分和重排，目的是为了在多设备上分布式计算
        if self.enable_sequence_parallelism:
            assert not self.sequence_parallelism_temporal, "not support!"
            x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="down")
            c = split_forward_gather_backward(c, get_sequence_parallel_group(), dim=2, grad_scale="down")
            S = S // dist.get_world_size(get_sequence_parallel_group())
        # c = torch.randn_like(x)  # change me!
        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
        c = rearrange(c, "B T S C -> B (T S) C", T=T, S=S)

        # === blocks ===
        '''
        进入 Transformer 块之前，x 和 c 都已经整理成形状 [B, (T×S), hidden_size]。接下来的代码主要分为两部分：

        第一部分：对于前 control_depth 个 block，同时处理基准块和控制块。
        第二部分：对于后续的基准块（base_blocks_s/t），只处理 x 分支。

        
        '''
        if x_mask is not None: # ([12, 3])
            x_mask = repeat(x_mask, "b ... -> (b NC) ...", NC=NC)
        
        # 2.1. 控制块循环（for block_i in range(0, self.control_depth)）
        for block_i in range(0, self.control_depth):
            x = auto_grad_checkpoint( # auto_grad_checkpoint 包裹调用用于节省显存，同时允许在前向不完整保存中间激活，后向再重新计算
                self.base_blocks_s[block_i], 
                x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC, mv_order_map, t_order_map)
            c, c_skip = auto_grad_checkpoint( # 这部分 skip 信息携带了条件（例如多视图信息）对生成结果的修正作用，然后用残差加到 x 上
                self.control_blocks_s[block_i],
                c, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC, mv_order_map, t_order_map)
            x = x + c_skip  # connection
            if self.base_blocks_t is not None:
                x = auto_grad_checkpoint(
                    self.base_blocks_t[block_i],
                    x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC, mv_order_map, t_order_map)
            if self.control_blocks_t is not None:
                c, c_skip = auto_grad_checkpoint(
                    self.control_blocks_t[block_i],
                    c, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC, mv_order_map, t_order_map)
                x = x + c_skip  # connection

        # 2.2. 后续基块循环（for block_i in range(self.control_depth, self.depth)）
        for block_i in range(self.control_depth, self.depth):# 13-28
            x = auto_grad_checkpoint(
                self.base_blocks_s[block_i],
                x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC, mv_order_map, t_order_map)
            if self.base_blocks_t is not None:
                x = auto_grad_checkpoint(
                    self.base_blocks_t[block_i],
                    x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC, mv_order_map, t_order_map)

        if self.enable_sequence_parallelism:
            x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
            x = gather_forward_split_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="up")
            S = S * dist.get_world_size(get_sequence_parallel_group())
            x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # 3.1. Final Layer 与 Unpatchify
        # final_layer 和 unpatchify 内部只是对张量形状做转换和线性映射，与条件注入逻辑无关，所以不展开
        # x = self.final_layer( # final_layer（T2IFinalLayer）对经过 Transformer 块处理后的 x 进行最终的线性映射，获得输出的 latent 表示
        #     x, repeat(t, "b d -> (b NC) d", NC=NC),
        #     x_mask, repeat(t0, "b d -> (b NC) d", NC=NC) if t0 is not None else None,
        #     T, S,
        # )
        x = self.final_layer(
            x,
            t,                                           # >>> NEW: 不再 repeat
            x_mask,
            t0 if x_mask is not None else None,
            T, S,
        )
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx) # 调用 unpatchify 将 patch 序列还原成视频帧形式
        # x:12, 4050, 1152->[12, 4050, 64]

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        # HACK: to use scheduler, we never assume NC with C
        # 旧行: x = rearrange(x, "(B NC) C T ... -> B (C NC) T ...", NC=NC)
        # >>> NEW: 直接返回 [B, C_out, T', H, W]
        return x

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x


# def load_from_stdit3_pretrained(model, from_pretrained):
#     from ..stdit import STDiT3
#     base_model = STDiT3.from_pretrained(from_pretrained)

#     # helper modules
#     (m, u) = model.load_state_dict(base_model.state_dict(), strict=False)
#     if model.x_control_embedder is not None:
#         model.x_control_embedder.load_state_dict(base_model.x_embedder.state_dict())
#     _m, _u = [], []
#     for key in m:
#         if key.startswith("base_blocks_") or key.startswith("control_blocks_"):
#             pass
#         else:
#             _m.append(key)
#     for key in u:
#         if key.startswith("spatial_blocks") or key.startswith("temporal_blocks"):
#             pass
#         else:
#             _u.append(key)
#     logging.info(f"1st, Load from {from_pretrained} with \nmissing={_m}, \nunexpected={_u}")

#     # main blocks
#     base_m, base_u, control_m, control_u = [], [], [], []
#     (m, u) = model.base_blocks_s.load_state_dict(base_model.spatial_blocks.state_dict(), strict=False)
#     base_m.append(m)
#     base_u.append(u)
#     if model.base_blocks_t is not None:
#         (m, u) = model.base_blocks_t.load_state_dict(base_model.temporal_blocks.state_dict(), strict=False)
#         base_m.append(m)
#         base_u.append(u)
#     logging.info(f"2nd, Load base from {from_pretrained} with \nmissing={base_m}, \nunexpected={base_u}")

#     # control blocks
#     (m, u) = model.control_blocks_s.load_state_dict(base_model.spatial_blocks.state_dict(), strict=False)
#     control_m.append(m)
#     control_u.append(u)
#     if model.control_blocks_t is not None:
#         (m, u) = model.control_blocks_t.load_state_dict(base_model.temporal_blocks.state_dict(), strict=False)
#         control_m.append(m)
#         control_u.append(u)
#     logging.info(f"3nd, Load control from {from_pretrained} with \nmissing={control_m}, \nunexpected={control_u}")
#     return model


# def load_from_pixart_pretrained(model: MagicDriveSTDiT3, pretrained):
#     from ..pixart import PixArt_XL_2
#     base_model = PixArt_XL_2(from_pretrained=pretrained)

#     # helper modules
#     (m, u) = model.load_state_dict(base_model.state_dict(), strict=False)
#     if model.x_control_embedder is not None:
#         model.x_control_embedder.load_state_dict(base_model.x_embedder.state_dict())
#     _m, _u = [], []
#     for key in m:
#         if key.startswith("base_blocks_") or key.startswith("control_blocks_"):
#             pass
#         else:
#             _m.append(key)
#     for key in u:
#         if key.startswith("blocks"):
#             pass
#         else:
#             _u.append(key)
#     logging.info(f"1st, Load from {pretrained} with \nmissing={_m}, \nunexpected={_u}")

#     base_m, base_u, control_m, control_u = [], [], [], []
#     # main blocks
#     (m, u) = model.base_blocks_s.load_state_dict(base_model.blocks.state_dict(), strict=False)
#     base_m.append(m)
#     base_u.append(u)
#     logging.info(f"2nd, Load base from {pretrained} with \nmissing={base_m}, \nunexpected={base_u}")

#     # control blocks
#     (m, u) = model.control_blocks_s.load_state_dict(base_model.blocks[:len(model.control_blocks_s)].state_dict(), strict=False)
#     control_m.append(m)
#     control_u.append(u)
#     logging.info(f"3nd, Load control from {pretrained} with \nmissing={control_m}, \nunexpected={control_u}")

#     return model


@MODELS.register_module("UltraDriveSTDiT3-XL/2")
def UltraDriveSTDiT3_XL_2(from_pretrained=None, force_huggingface=False, **kwargs):
    if from_pretrained is not None and not (os.path.exists(from_pretrained)):
        model = UltraDriveSTDiT3.from_pretrained(from_pretrained, **kwargs)
    else:
        from_pretrained_pixart = kwargs.pop("from_pretrained_pixart", None)
        config = UltraDriveSTDiT3Config(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
        model = UltraDriveSTDiT3(config)
        if from_pretrained is not None and force_huggingface:  # load from hf stdit3 model
            raise ValueError(f"from_pretrained is not supported for UltraDriveSTDiT3.")
            # load_from_stdit3_pretrained(model, from_pretrained)
        elif from_pretrained is not None:
            load_checkpoint(model, from_pretrained, strict=True)
        elif from_pretrained_pixart is not None:
            raise ValueError(f"from_pretrained_pixart is not supported for UltraDriveSTDiT3.")
            # load_from_pixart_pretrained(model, from_pretrained_pixart)
        else: # 走这条路
            logging.info(f"Your model does not use any pre-trained model.")
    return model
