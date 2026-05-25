# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Eleos config registry and builder functions.

Named configurations
--------------------
* ``debugmodel``  — tiny (dim=256, 6 layers) for fast iteration / CI.
* ``16B``         — 16B parameter scale.
* ``236B``        — 236B parameter scale.
* ``671B``        — 671B parameter scale.

Per-layer hemisphere split
--------------------------
``logical_head_fractions`` is a list of floats, one per layer, controlling the
fraction of attention heads devoted to logical (left-brain) reasoning in that
layer.  A value of 0.5 gives an equal split; values closer to 1.0 make a layer
more analytically oriented; values closer to 0.0 make it more generative/creative.

The builder helpers accept a single default fraction (applied uniformly to all
layers) **and** an optional override list so callers can set any pattern they want.

Moral loss weight
-----------------
``moral_loss_weight`` defaults to ``1e-3`` (same order of magnitude as the MoE
load-balance auxiliary loss).  It can be overridden per config.
"""

from collections.abc import Callable
from functools import partial
from typing import Literal

import torch.nn as nn

import torch
from torchtitan.components.loss import build_cross_entropy_loss, cross_entropy_loss, IGNORE_INDEX
from torchtitan.config import CompileConfig
from torchtitan.tools.logging import logger
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear, RMSNorm, RoPE, TransformerBlock
from torchtitan.models.common.attention import FlexAttention, ScaledDotProductAttention
from torchtitan.models.common.config_utils import (
    make_experts_config,
    make_ffn_config,
    make_moe_config,
    make_router_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.protocols.model_spec import ModelSpec

from .model import CorpusCallosum, DualHemisphereAttention, EleosModel, EleosTransformerBlock
from .parallelize import parallelize_eleos
from .state_dict_adapter import EleosStateDictAdapter

__all__ = [
    "parallelize_eleos",
    "EleosModel",
    "CorpusCallosum",
    "eleos_configs",
]


# ---------------------------------------------------------------------------
# Parameter init helpers (identical conventions to DeepSeek V3)
# ---------------------------------------------------------------------------

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim ** -0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _depth_experts_init(layer_id: int) -> dict[str, Callable]:
    return {
        "w1": partial(nn.init.trunc_normal_, std=0.02),
        "w2": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }


# ---------------------------------------------------------------------------
# DualHemisphereAttention config builder
# ---------------------------------------------------------------------------

def _make_dha_config(
    *,
    layer_id: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    mscale: float = 1.0,
    inner_attention=None,
    mask_type: str = "causal",
    logical_head_fraction: float = 0.5,
    moral_gate_hidden: int | None = None,
    moral_override_alpha: float = 0.85,
    adversarial_threshold: float = 0.5,
    corpus_callosum: CorpusCallosum.Config | None = None,
) -> DualHemisphereAttention.Config:
    """Build a fully-specified DualHemisphereAttention.Config for one layer."""
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # Resolve head split
    n_logical = max(1, round(n_heads * logical_head_fraction))
    n_gen = n_heads - n_logical

    if q_lora_rank == 0:
        # ---- Full Q projections ----
        wq_logical = Linear.Config(
            in_features=dim,
            out_features=n_logical * qk_head_dim,
            param_init=_LINEAR_INIT,
        )
        wq_gen = Linear.Config(
            in_features=dim,
            out_features=n_gen * qk_head_dim,
            param_init=_LINEAR_INIT,
        )
        wq_a_logical = wq_b_logical = None
        wq_a_gen = wq_b_gen = None
        q_norm_logical = RMSNorm.Config(normalized_shape=1, param_init=_NORM_INIT)
        q_norm_gen = RMSNorm.Config(normalized_shape=1, param_init=_NORM_INIT)
    else:
        # ---- LoRA Q projections ----
        wq_logical = wq_gen = None
        wq_a_logical = Linear.Config(
            in_features=dim,
            out_features=q_lora_rank,
            param_init=_LINEAR_INIT,
        )
        wq_b_logical = Linear.Config(
            in_features=q_lora_rank,
            out_features=n_logical * qk_head_dim,
            param_init=_LINEAR_INIT,
        )
        wq_a_gen = Linear.Config(
            in_features=dim,
            out_features=q_lora_rank,
            param_init=_LINEAR_INIT,
        )
        wq_b_gen = Linear.Config(
            in_features=q_lora_rank,
            out_features=n_gen * qk_head_dim,
            param_init=_LINEAR_INIT,
        )
        q_norm_logical = RMSNorm.Config(normalized_shape=q_lora_rank, param_init=_NORM_INIT)
        q_norm_gen = RMSNorm.Config(normalized_shape=q_lora_rank, param_init=_NORM_INIT)

    return DualHemisphereAttention.Config(
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        mscale=mscale,
        # Q projections
        wq_logical=wq_logical,
        wq_a_logical=wq_a_logical,
        wq_b_logical=wq_b_logical,
        wq_gen=wq_gen,
        wq_a_gen=wq_a_gen,
        wq_b_gen=wq_b_gen,
        q_norm_logical=q_norm_logical,
        q_norm_gen=q_norm_gen,
        # Shared KV
        wkv_a=Linear.Config(
            in_features=dim,
            out_features=kv_lora_rank + qk_rope_head_dim,
            param_init=_LINEAR_INIT,
        ),
        kv_norm=RMSNorm.Config(normalized_shape=kv_lora_rank, param_init=_NORM_INIT),
        wkv_b=Linear.Config(
            in_features=kv_lora_rank,
            # wkv_b must accommodate the largest hemisphere for the nope projection;
            # we use the full n_heads set here (shared matrix, the hemisphere splits
            # happen in the Q path).  At runtime the hemispherical KV calls each
            # pass their own n_heads count to _compute_kv so the view is consistent.
            out_features=n_heads * (qk_nope_head_dim + v_head_dim),
            param_init=_LINEAR_INIT,
        ),
        # Output projections (one per hemisphere, sized to their head count)
        wo_logical=Linear.Config(
            in_features=n_logical * v_head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
        wo_gen=Linear.Config(
            in_features=n_gen * v_head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
        inner_attention=(
            inner_attention
            if inner_attention is not None
            else ScaledDotProductAttention.Config()
        ),
        mask_type=mask_type,
        # Hemisphere split
        logical_head_fraction=logical_head_fraction,
        # Corpus callosum (None = segregated mode for this layer)
        corpus_callosum=corpus_callosum,
        # Moral governor
        moral_gate_hidden=moral_gate_hidden,
        moral_override_alpha=moral_override_alpha,
        adversarial_threshold=adversarial_threshold,
    )


# ---------------------------------------------------------------------------
# Layer list builder
# ---------------------------------------------------------------------------

def _build_eleos_layers(
    *,
    n_layers: int,
    n_dense_layers: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    mscale: float,
    dense_hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    num_shared_experts: int,
    router_top_k: int,
    router_score_func: Literal["sigmoid", "softmax"],
    router_num_expert_groups: int | None = None,
    router_num_limited_groups: int | None = None,
    router_route_scale: float = 1.0,
    router_route_norm: bool = False,
    score_before_experts: bool = False,
    inner_attention=None,
    mask_type: str = "causal",
    # --- hemisphere split ---
    logical_head_fractions: list[float] | None = None,
    default_logical_head_fraction: float = 0.5,
    # --- moral governor ---
    moral_gate_hidden: int | None = None,
    moral_override_alpha: float = 0.85,
    adversarial_threshold: float = 0.5,
    # --- corpus callosum ---
    callosal_layers: list[int] | None = None,
    callosal_fiber_rank_thin: int | None = None,
    callosal_fiber_rank_thick: int | None = None,
    callosal_thick_layers: list[int] | None = None,
    callosal_gate_hidden: int | None = None,
) -> list[TransformerBlock.Config]:
    """Build per-layer EleosTransformerBlock configs.

    Args:
        logical_head_fractions: Optional list of per-layer fractions (length must
            equal n_layers).  If provided, each layer uses its own fraction.
            If ``None``, every layer uses ``default_logical_head_fraction``.
        default_logical_head_fraction: Fallback fraction used when
            ``logical_head_fractions`` is ``None``.
        callosal_layers: Layer indices that receive a CorpusCallosum module.
            Corresponds to anatomical regions: genu (early), midbody (middle),
            isthmus (mid-late), splenium (final). If ``None``, no callosal
            connections are added (pure segregated mode).
        callosal_fiber_rank_thin: Rank for genu/splenium (associative) layers.
            Defaults to ``max(16, dim // 8)``.
        callosal_fiber_rank_thick: Rank for midbody (sensorimotor) layers.
            Defaults to ``max(32, dim // 4)``.
        callosal_thick_layers: Subset of ``callosal_layers`` that use the
            thick-fiber (midbody) rank.  Defaults to the middle 40% of
            ``callosal_layers``.
        callosal_gate_hidden: Hidden dim of the callosal gate MLP.
            Defaults to ``max(16, dim // 8)``.  Smaller values are preferred
            since the gate only needs to compute a scalar switch signal.
    """
    if logical_head_fractions is not None:
        assert len(logical_head_fractions) == n_layers, (
            f"logical_head_fractions length ({len(logical_head_fractions)}) "
            f"must equal n_layers ({n_layers})"
        )
    else:
        logical_head_fractions = [default_logical_head_fraction] * n_layers

    # --- Resolve callosal fiber ranks ---
    _thin_rank = callosal_fiber_rank_thin or max(16, dim // 8)
    _thick_rank = callosal_fiber_rank_thick or max(32, dim // 4)
    _callosal_set = set(callosal_layers) if callosal_layers else set()
    # Layers that use the thick (midbody) fiber rank
    if callosal_thick_layers is not None:
        _thick_set = set(callosal_thick_layers)
    elif callosal_layers:
        # Default: middle 40% of callosal layers are midbody (thick)
        sorted_cal = sorted(callosal_layers)
        n_cal = len(sorted_cal)
        lo = n_cal // 4
        hi = 3 * n_cal // 4
        _thick_set = set(sorted_cal[lo:hi])
    else:
        _thick_set = set()

    layers = []
    for layer_id in range(n_layers):
        # Resolve corpus callosum config for this layer
        callosum_cfg: CorpusCallosum.Config | None = None
        if layer_id in _callosal_set:
            fiber_rank = _thick_rank if layer_id in _thick_set else _thin_rank
            callosum_cfg = CorpusCallosum.Config(
                dim=dim,
                fiber_rank=fiber_rank,
                gate_hidden=callosal_gate_hidden,
            )

        attn_cfg = _make_dha_config(
            layer_id=layer_id,
            dim=dim,
            n_heads=n_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            mscale=mscale,
            inner_attention=inner_attention,
            mask_type=mask_type,
            logical_head_fraction=logical_head_fractions[layer_id],
            moral_gate_hidden=moral_gate_hidden,
            moral_override_alpha=moral_override_alpha,
            adversarial_threshold=adversarial_threshold,
            corpus_callosum=callosum_cfg,
        )

        if layer_id < n_dense_layers:
            ffn_cfg = make_ffn_config(
                dim=dim,
                hidden_dim=dense_hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            )
            moe_cfg = None
        else:
            ffn_cfg = None
            moe_cfg = make_moe_config(
                num_experts=num_experts,
                score_before_experts=score_before_experts,
                router=make_router_config(
                    dim=dim,
                    num_experts=num_experts,
                    gate_param_init=_depth_init(layer_id),
                    top_k=router_top_k,
                    score_func=router_score_func,
                    num_expert_groups=router_num_expert_groups,
                    num_limited_groups=router_num_limited_groups,
                    route_scale=router_route_scale,
                    route_norm=router_route_norm,
                ),
                experts=make_experts_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim,
                    num_experts=num_experts,
                    param_init=_depth_experts_init(layer_id),
                ),
                shared_experts=make_ffn_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim * num_shared_experts,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )

        layers.append(
            EleosTransformerBlock.Config(
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
                feed_forward=ffn_cfg,
                moe=moe_cfg,
            )
        )
    return layers


# ---------------------------------------------------------------------------
# Named config functions
# ---------------------------------------------------------------------------

def _rope_config(rope_dim: int, seq_len: int = 4096, factor: float = 40.0) -> RoPE.Config:
    return RoPE.Config(
        dim=rope_dim,
        max_seq_len=seq_len * 4,
        theta=10000.0,
        backend="complex",
        scaling="yarn",
        rope_factor=factor,
        beta_fast=32.0,
        beta_slow=1.0,
        original_seq_len=seq_len,
    )


def _debugmodel() -> EleosModel.Config:
    """Tiny debug model for fast iteration (dim=256, 6 layers).

    Uses a graduated per-layer split: early layers (0-1) are 70% logical,
    middle layers (2-3) are 50/50, final layers (4-5) are 70% generative.
    """
    dim, n_layers, vocab_size = 256, 6, 2048
    n_heads = 16
    rope_dim = 64

    # Per-layer fractions: more logical early, more generative late
    fractions = [0.70, 0.65, 0.50, 0.50, 0.35, 0.30]

    layers = _build_eleos_layers(
        n_layers=n_layers,
        n_dense_layers=1,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=1024,
        moe_hidden_dim=256,
        num_experts=8,
        num_shared_experts=2,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
        logical_head_fractions=fractions,
        # Corpus callosum — 6 layers (dim=256), topographic:
        #   genu=0       (thin)  → planning / higher-order
        #   midbody=2-3  (thick) → integration hub
        #   splenium=5   (thin)  → perceptual synthesis
        # fiber_rank_thin = 24  (~9% of dim, associative fibers)
        # fiber_rank_thick = 48 (~19% of dim, fast sensorimotor fibers)
        # gate_hidden = 16      (minimal — gate is binary switch only)
        callosal_layers=[0, 2, 3, 5],
        callosal_thick_layers=[2, 3],
        callosal_fiber_rank_thin=24,
        callosal_fiber_rank_thick=48,
        callosal_gate_hidden=16,
    )
    return EleosModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        moral_loss_weight=1e-3,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=_rope_config(rope_dim),
        layers=layers,
    )


def _debugmodel_flex_attn() -> EleosModel.Config:
    """Debug model with FlexAttention."""
    dim, n_layers, vocab_size = 256, 6, 2048
    n_heads = 16
    rope_dim = 64
    fractions = [0.70, 0.65, 0.50, 0.50, 0.35, 0.30]

    layers = _build_eleos_layers(
        n_layers=n_layers,
        n_dense_layers=1,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=1024,
        moe_hidden_dim=256,
        num_experts=8,
        num_shared_experts=2,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
        inner_attention=FlexAttention.Config(),
        mask_type="block_causal",
        logical_head_fractions=fractions,
        # Corpus callosum — 6 layers (dim=256), topographic:
        #   genu=0       (thin)  → planning / higher-order
        #   midbody=2-3  (thick) → integration hub
        #   splenium=5   (thin)  → perceptual synthesis
        # Same parameters as _debugmodel for consistency
        callosal_layers=[0, 2, 3, 5],
        callosal_thick_layers=[2, 3],
        callosal_fiber_rank_thin=24,
        callosal_fiber_rank_thick=48,
        callosal_gate_hidden=16,
    )
    return EleosModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        moral_loss_weight=1e-3,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=_rope_config(rope_dim),
        layers=layers,
    )


def _16b() -> EleosModel.Config:
    """16B parameter Eleos model.

    Head split follows a graduated pattern:
    - Dense layers (0):      0.70 logical (heavily analytical)
    - MoE layers 1-9:        0.60 logical  (analytically leaning)
    - MoE layers 10-18:      0.50 (balanced)
    - MoE layers 19-26:      0.40 (generatively leaning)
    """
    dim, n_layers, vocab_size = 2048, 27, 102400
    n_heads = 16
    rope_dim = 64

    fractions = [
        0.70,                                   # dense layer 0
        *[0.60] * 9,                            # early MoE layers 1-9
        *[0.50] * 9,                            # mid MoE layers 10-18
        *[0.40] * 8,                            # late MoE layers 19-26
    ]
    assert len(fractions) == n_layers

    layers = _build_eleos_layers(
        n_layers=n_layers,
        n_dense_layers=1,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=10944,
        moe_hidden_dim=1408,
        num_experts=64,
        num_shared_experts=2,
        router_top_k=6,
        router_score_func="softmax",
        score_before_experts=False,
        inner_attention=FlexAttention.Config(),
        mask_type="block_causal",
        logical_head_fractions=fractions,
        # Corpus callosum — 27 layers (dim=2048), topographic mapping:
        #   genu=0-1       (thin)  → prefrontal planning
        #   midbody=7-16   (thick) → motor/sensory integration
        #   isthmus=17-19  (thin)  → auditory/temporal
        #   splenium=25-26 (thin)  → visual/perceptual synthesis
        # fiber_rank_thin = 128 (dim//16, ~6% of dim — associative fibers)
        # fiber_rank_thick = 256 (dim//8,  ~12% of dim — fast sensorimotor)
        # gate_hidden = 64       (capped — gate is a scalar switch, not complex reasoning)
        # Callosal params add ~29M params total — ~0.2% of 16B active params
        callosal_layers=[0, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 25, 26],
        callosal_thick_layers=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        callosal_fiber_rank_thin=128,
        callosal_fiber_rank_thick=256,
        callosal_gate_hidden=64,
    )
    return EleosModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        moral_loss_weight=1e-3,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=_rope_config(rope_dim),
        layers=layers,
    )


def _236b() -> EleosModel.Config:
    """236B parameter Eleos model (LoRA Q projections)."""
    dim, n_layers, vocab_size = 5120, 60, 102400
    n_heads = 128
    q_lora_rank = 1536
    rope_dim = 64

    # Graduated: starts 65% logical, descends to 35% by final layer
    step = (0.65 - 0.35) / max(n_layers - 1, 1)
    fractions = [round(0.65 - i * step, 4) for i in range(n_layers)]

    layers = _build_eleos_layers(
        n_layers=n_layers,
        n_dense_layers=1,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=1.0,
        dense_hidden_dim=12288,
        moe_hidden_dim=1536,
        num_experts=160,
        num_shared_experts=2,
        router_top_k=6,
        router_score_func="softmax",
        router_num_expert_groups=8,
        router_num_limited_groups=3,
        router_route_scale=16.0,
        score_before_experts=False,
        inner_attention=FlexAttention.Config(),
        mask_type="block_causal",
        logical_head_fractions=fractions,
        # Corpus callosum — 60 layers (dim=5120), topographic mapping:
        #   genu=0-3       (thin)  → prefrontal, highest-order cognition
        #   midbody=15-37  (thick) → large sensorimotor integration band
        #   isthmus=38-44  (thin)  → auditory/temporal
        #   splenium=57-59 (thin)  → visual, perceptual synthesis
        # fiber_rank_thin = 256 (dim//20, ~5% of dim — capped for memory)
        # fiber_rank_thick = 512 (dim//10, ~10% of dim — fast fibers)
        # gate_hidden = 64       (fixed cap — gate complexity doesn't scale with dim)
        # Callosal overhead ≈ 0.5% of total params at this scale
        callosal_layers=[*range(0, 4), *range(15, 45), *range(57, 60)],
        callosal_thick_layers=list(range(15, 38)),
        callosal_fiber_rank_thin=256,
        callosal_fiber_rank_thick=512,
        callosal_gate_hidden=64,
    )
    return EleosModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        moral_loss_weight=1e-3,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=_rope_config(rope_dim),
        layers=layers,
    )


def _671b() -> EleosModel.Config:
    """671B parameter Eleos model (LoRA Q projections, sigmoid router)."""
    dim, n_layers, vocab_size = 7168, 61, 129280
    n_heads = 128
    q_lora_rank = 1536
    rope_dim = 64

    step = (0.65 - 0.35) / max(n_layers - 1, 1)
    fractions = [round(0.65 - i * step, 4) for i in range(n_layers)]

    layers = _build_eleos_layers(
        n_layers=n_layers,
        n_dense_layers=3,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=1.0,
        dense_hidden_dim=18432,
        moe_hidden_dim=2048,
        num_experts=256,
        num_shared_experts=1,
        router_top_k=8,
        router_score_func="sigmoid",
        router_num_expert_groups=8,
        router_num_limited_groups=4,
        router_route_scale=2.5,
        router_route_norm=True,
        score_before_experts=False,
        inner_attention=FlexAttention.Config(),
        mask_type="block_causal",
        logical_head_fractions=fractions,
        # Corpus callosum — 61 layers (dim=7168), topographic mapping:
        #   genu=0-3       (thin)  → prefrontal, executive function
        #   midbody=15-38  (thick) → primary motor + somatosensory
        #   isthmus=39-46  (thin)  → auditory cortex connections
        #   splenium=58-60 (thin)  → visual cortex, perceptual synthesis
        # fiber_rank_thin = 256 (dim//28, ~3.5% of dim — thinnest callosal budget)
        # fiber_rank_thick = 512 (dim//14, ~7% of dim — fast sensorimotor fibers)
        # gate_hidden = 64       (fixed cap — sigmoid scalar, no benefit from larger)
        # Callosal overhead ≈ 0.4% of total params at this scale
        callosal_layers=[*range(0, 4), *range(15, 47), *range(58, 61)],
        callosal_thick_layers=list(range(15, 39)),
        callosal_fiber_rank_thin=256,
        callosal_fiber_rank_thick=512,
        callosal_gate_hidden=64,
    )
    return EleosModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        moral_loss_weight=1e-3,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=_rope_config(rope_dim),
        layers=layers,
    )


def _small() -> EleosModel.Config:
    """~440M parameter model designed specifically to train comfortably on a MacBook/locally."""
    dim, n_layers, vocab_size = 768, 12, 102400
    n_heads = 12
    rope_dim = 64

    # Simple 50/50 split for all layers
    fractions = [0.50] * n_layers

    layers = _build_eleos_layers(
        n_layers=n_layers,
        n_dense_layers=1,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=256,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=2048,
        moe_hidden_dim=512,
        num_experts=16,
        num_shared_experts=2,
        router_top_k=4,
        router_score_func="softmax",
        score_before_experts=False,
        inner_attention=FlexAttention.Config(),
        mask_type="block_causal",
        logical_head_fractions=fractions,
        # Corpus callosum — 12 layers (dim=768), topographic mapping:
        #   genu=0-1      (thin)  → prefrontal, planning
        #   midbody=4-7   (thick) → motor/somatosensory integration
        #   isthmus=8-9   (thin)  → auditory/temporal
        #   splenium=10-11(thin)  → perceptual synthesis
        # fiber_rank_thin = 48  (dim//16, ~6% of dim — associative fibers)
        # fiber_rank_thick = 96  (dim//8,  ~12% of dim — fast sensorimotor)
        # gate_hidden = 24       (dim//32 — minimal gate for scalar switch)
        # Callosal overhead ≈ 0.8% of total params at this scale
        callosal_layers=[0, 1, 4, 5, 6, 7, 8, 9, 10, 11],
        callosal_thick_layers=[4, 5, 6, 7],
        callosal_fiber_rank_thin=48,
        callosal_fiber_rank_thick=96,
        callosal_gate_hidden=24,
    )
    return EleosModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        moral_loss_weight=1e-3,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=_rope_config(rope_dim),
        layers=layers,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

eleos_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_flex_attn": _debugmodel_flex_attn,
    "small": _small,
    "16B": _16b,
    "236B": _236b,
    "671B": _671b,
}


def eleos_loss_fn(pred: tuple[torch.Tensor, torch.Tensor] | torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if isinstance(pred, tuple):
        logits, moral_aux_loss = pred
        ce_loss_sum = cross_entropy_loss(logits, labels)
        local_valid_tokens = (labels != IGNORE_INDEX).sum()
        return ce_loss_sum + moral_aux_loss * local_valid_tokens
    else:
        return cross_entropy_loss(pred, labels)


def build_eleos_loss_fn(compile_config: CompileConfig, **kwargs):
    del kwargs
    loss_fn = eleos_loss_fn
    if compile_config.enable and "loss" in compile_config.components:
        logger.info("Compiling the Eleos loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=compile_config.backend)
    return loss_fn


def model_registry(flavor: str) -> ModelSpec:
    config = eleos_configs[flavor]()
    return ModelSpec(
        name="eleos",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_eleos,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_eleos_loss_fn,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=EleosStateDictAdapter,
    )
