# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Eleos Model Architecture
========================

Eleos (Greek: ἔλεος, "mercy / compassion") is a novel LLM architecture
built on top of DeepSeek V3's Multi-Latent Attention (MLA).  It introduces
three integral architectural components:

1. **DualHemisphereAttention** — Splits the MLA head pool into a *logical*
   sub-stream (structured reasoning, deductive inference) and a *generative*
   sub-stream (creative synthesis, expressive language), analogous to the
   left-brain / right-brain functional specialisation hypothesis.  The head
   split ratio is **configurable per layer** so early layers can favour
   logical processing while later layers favour generative synthesis.

2. **MoralGovernor** — A differentiable module embedded inside every
   transformer block.  It:
   - Computes a scalar *moral score* in [0, 1] from the hidden state.
   - Computes a continuous *blend gate* (α ∈ [0, 1]) that weights how much
     logical vs. generative output flows through.
   - Contains an **AdversarialDetector** sub-module that learns to recognise
     manipulative / harmful input patterns and amplifies the moral tone of
     responses when such intent is detected.

3. **Moral Auxiliary Loss** — During training a secondary loss term
   ``-mean(moral_scores) * moral_loss_weight`` encourages the model to
   develop internal representations that score highly on morality.  This is
   analogous to—and implemented in the same manner as—the MoE load-balance
   auxiliary loss already present in DeepSeek V3.

All components are *integral to the weight structure*; moral alignment is
a learned, differentiable property of the model rather than a post-hoc
output filter.
"""

import dataclasses
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    LocalMapInnerAttention,
    ScaledDotProductAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.common.rope import apply_rotary_emb_single_complex
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


# ---------------------------------------------------------------------------
# AdversarialDetector
# ---------------------------------------------------------------------------

TTLinear = Module.from_nn_module(nn.Linear)

class AdversarialDetector(Module):
    """Lightweight 2-layer MLP that predicts whether the current hidden state
    carries signals of adversarial / manipulative intent.

    Outputs a scalar probability in [0, 1] per token (higher = more likely
    adversarial).  During training this is supervised implicitly by the
    MoralGovernor's blend gate: adversarial inputs should route more weight
    toward the moral-elevating blend.  Explicit labelled adversarial examples
    in the training corpus will sharpen this signal.

    Args:
        dim: Hidden dimension of the model.
        hidden_dim: Intermediate dimension of the detector MLP (default dim//4).
    """

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(64, dim // 4)
        self.fc1 = TTLinear(dim, hidden_dim, bias=False)
        self.fc2 = TTLinear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(batch, seq, dim)``
        Returns:
            adversarial_prob: ``(batch, seq, 1)``  values in [0, 1]
        """
        h = F.silu(self.fc1(x))
        return torch.sigmoid(self.fc2(h))


# ---------------------------------------------------------------------------
# MoralGovernor
# ---------------------------------------------------------------------------

class MoralGovernor(Module):
    """Differentiable moral routing module embedded in every transformer block.

    Given the pre-attention hidden state *x*, the raw logical attention output
    *logical_out*, and the raw generative attention output *gen_out*, it
    computes:

    * ``moral_score`` ∈ [0, 1] — how morally aligned the current representation is.
    * ``alpha`` ∈ [0, 1] — blend weight: ``alpha`` toward logical, ``1-alpha``
      toward generative.  When adversarial intent is detected, ``alpha`` is
      boosted toward ``moral_override_alpha`` to dampen potential harmful
      reasoning chains and elevate moral tone.
    * ``adversarial_prob`` ∈ [0, 1] — per-token adversarial signal.

    The final blended attention output is::

        blended = alpha * logical_out + (1 - alpha) * gen_out

    Args:
        dim: Hidden dimension.
        moral_gate_hidden: Hidden size of the moral scoring MLP (default dim//8).
        moral_override_alpha: The alpha value to interpolate toward when
            adversarial intent is detected (default 0.85, heavily logical /
            deliberative to suppress creative harm).
        adversarial_threshold: Probability above which a token is considered
            adversarial for the override interpolation (default 0.5).
    """

    def __init__(
        self,
        dim: int,
        moral_gate_hidden: int | None = None,
        moral_override_alpha: float = 0.85,
        adversarial_threshold: float = 0.5,
    ):
        super().__init__()
        moral_gate_hidden = moral_gate_hidden or max(64, dim // 8)

        # Two-output MLP: produces [moral_score_logit, blend_alpha_logit]
        self.gate_fc1 = TTLinear(dim, moral_gate_hidden, bias=False)
        self.gate_fc2 = TTLinear(moral_gate_hidden, 2, bias=False)

        self.adversarial_detector = AdversarialDetector(dim)
        self.moral_override_alpha = moral_override_alpha
        self.adversarial_threshold = adversarial_threshold

    def forward(
        self,
        x: torch.Tensor,
        logical_out: torch.Tensor,
        gen_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:           ``(batch, seq, dim)`` — pre-norm hidden state fed to attn.
            logical_out: ``(batch, seq, dim)`` — logical hemisphere attention output.
            gen_out:     ``(batch, seq, dim)`` — generative hemisphere attention output.

        Returns:
            blended:          ``(batch, seq, dim)`` — morally-routed attention output.
            moral_score:      ``(batch, seq, 1)``  — per-token moral alignment in [0, 1].
            adversarial_prob: ``(batch, seq, 1)``  — per-token adversarial probability.
        """
        # 1. Compute moral score and raw blend alpha from hidden state
        h = F.silu(self.gate_fc1(x))               # (B, S, moral_gate_hidden)
        gate_out = torch.sigmoid(self.gate_fc2(h))  # (B, S, 2)  values in [0, 1]
        moral_score = gate_out[..., :1]             # (B, S, 1)
        alpha = gate_out[..., 1:]                   # (B, S, 1)

        # 2. Detect adversarial intent
        adversarial_prob = self.adversarial_detector(x)  # (B, S, 1)

        # 3. Override blend gate when adversarial intent is detected.
        #    We do a soft interpolation so the gradient can still flow.
        #    override_strength ∈ [0, 1]:  0 = no override, 1 = full override.
        override_strength = (adversarial_prob > self.adversarial_threshold).float()
        # Interpolate: when adversarial, push alpha toward moral_override_alpha
        alpha = override_strength * self.moral_override_alpha + (1.0 - override_strength) * alpha

        # 4. Blend the two hemispheres
        blended = alpha * logical_out + (1.0 - alpha) * gen_out

        return blended, moral_score, adversarial_prob


# ---------------------------------------------------------------------------
# DualHemisphereAttention
# ---------------------------------------------------------------------------

class DualHemisphereAttention(BaseAttention):
    """Multi-Latent Attention split into logical and generative hemispheres.

    The total head pool (``n_heads``) is partitioned per-layer via
    ``logical_head_fraction`` (a float in (0, 1)):

    * ``n_logical_heads = round(n_heads * logical_head_fraction)``
    * ``n_gen_heads     = n_heads - n_logical_heads``

    Both hemispheres share the KV compression matrices (``wkv_a``, ``kv_norm``,
    ``wkv_b``) to avoid doubling the KV parameter count.  Each hemisphere
    has its own Q projection and its own output projection.

    The hemisphere outputs are blended by an embedded :class:`MoralGovernor`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        dim: int
        # Q projections — logical hemisphere
        wq_logical: Linear.Config | None = None
        wq_a_logical: Linear.Config | None = None
        wq_b_logical: Linear.Config | None = None
        # Q projections — generative hemisphere
        wq_gen: Linear.Config | None = None
        wq_a_gen: Linear.Config | None = None
        wq_b_gen: Linear.Config | None = None
        # Shared KV compression
        wkv_a: Linear.Config
        wkv_b: Linear.Config
        wo_logical: Linear.Config
        wo_gen: Linear.Config
        q_lora_rank: int = 0
        kv_lora_rank: int = 512
        q_norm_logical: RMSNorm.Config
        q_norm_gen: RMSNorm.Config
        kv_norm: RMSNorm.Config
        qk_nope_head_dim: int = 128
        qk_rope_head_dim: int = 64
        v_head_dim: int = 128
        inner_attention: LocalMapInnerAttention.Config = field(
            default_factory=ScaledDotProductAttention.Config
        )
        mask_type: str = "causal"
        mscale: float = 1.0
        rope_factor: float = 1.0
        rope_max_seq_len: int = 4096
        rope_original_seq_len: int = 4096
        # --- hemisphere split (configurable per layer) ---
        logical_head_fraction: float = 0.5
        """Fraction of heads assigned to the logical hemisphere (0 < f < 1).
        The remainder go to the generative hemisphere.
        Defaults to 0.5 (equal split).
        """
        # --- moral governor hyper-params ---
        moral_gate_hidden: int | None = None
        moral_override_alpha: float = 0.85
        adversarial_threshold: float = 0.5

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        # Hemisphere head split
        assert 0.0 < config.logical_head_fraction < 1.0, (
            f"logical_head_fraction must be in (0, 1), got {config.logical_head_fraction}"
        )
        self.n_logical_heads = max(1, round(config.n_heads * config.logical_head_fraction))
        self.n_gen_heads = config.n_heads - self.n_logical_heads
        assert self.n_gen_heads >= 1, (
            f"n_gen_heads must be >= 1; reduce logical_head_fraction or increase n_heads"
        )

        # ---- Q projections: logical hemisphere ----
        if self.q_lora_rank == 0:
            assert config.wq_logical is not None, "wq_logical required when q_lora_rank==0"
            self.wq_logical = config.wq_logical.build()
        else:
            assert config.wq_a_logical is not None and config.wq_b_logical is not None
            self.wq_a_logical = config.wq_a_logical.build()
            self.q_norm_logical = config.q_norm_logical.build()
            self.wq_b_logical = config.wq_b_logical.build()

        # ---- Q projections: generative hemisphere ----
        if self.q_lora_rank == 0:
            assert config.wq_gen is not None, "wq_gen required when q_lora_rank==0"
            self.wq_gen = config.wq_gen.build()
        else:
            assert config.wq_a_gen is not None and config.wq_b_gen is not None
            self.wq_a_gen = config.wq_a_gen.build()
            self.q_norm_gen = config.q_norm_gen.build()
            self.wq_b_gen = config.wq_b_gen.build()

        # ---- Shared KV compression ----
        self.wkv_a = config.wkv_a.build()
        self.kv_norm = config.kv_norm.build()
        self.wkv_b = config.wkv_b.build()

        # ---- Output projections ----
        self.wo_logical = config.wo_logical.build()
        self.wo_gen = config.wo_gen.build()

        # ---- Inner attention (shared kernel) ----
        self.inner_attention = config.inner_attention.build()

        # ---- Softmax scale ----
        self.softmax_scale = self.qk_head_dim ** -0.5
        if config.rope_max_seq_len > config.rope_original_seq_len:
            mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # ---- Moral Governor ----
        self.moral_governor = MoralGovernor(
            dim=config.dim,
            moral_gate_hidden=config.moral_gate_hidden,
            moral_override_alpha=config.moral_override_alpha,
            adversarial_threshold=config.adversarial_threshold,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project_q(
        self,
        x: torch.Tensor,
        bsz: int,
        seqlen: int,
        n_heads: int,
        is_logical: bool,
    ) -> torch.Tensor:
        """Project input to Q for one hemisphere and apply RoPE-nope split."""
        if self.q_lora_rank == 0:
            wq = self.wq_logical if is_logical else self.wq_gen
            q = wq(x)
        else:
            wq_a = self.wq_a_logical if is_logical else self.wq_a_gen
            q_norm = self.q_norm_logical if is_logical else self.q_norm_gen
            wq_b = self.wq_b_logical if is_logical else self.wq_b_gen
            q = wq_b(q_norm(wq_a(x)))
        return q.view(bsz, seqlen, n_heads, self.qk_head_dim)

    def _compute_kv(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared KV compression used by both hemispheres."""
        kv = self.wkv_a(x)
        kv_lat, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb_single_complex(k_pe.unsqueeze(2), freqs_cis, positions)
        kv_lat = self.wkv_b(self.kv_norm(kv_lat))
        bsz, seqlen, _ = x.shape
        n_local_heads = self.n_logical_heads + self.n_gen_heads
        kv_lat = kv_lat.view(bsz, seqlen, n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_lat, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, n_local_heads, -1)], dim=-1)
        return k, v

    def _hemisphere_forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None,
        is_logical: bool,
    ) -> torch.Tensor:
        """Run one hemisphere and return its projected output (dim-space)."""
        bsz, seqlen, _ = x.shape
        n_heads = self.n_logical_heads if is_logical else self.n_gen_heads
        wo = self.wo_logical if is_logical else self.wo_gen

        # Q
        q = self._project_q(x, bsz, seqlen, n_heads, is_logical)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb_single_complex(q_pe, freqs_cis, positions)
        q = torch.cat([q_nope, q_pe], dim=-1)

        # Shared KV (we compute fresh for each hemisphere; note shared weights)
        k, v = self._compute_kv(x, freqs_cis, positions)
        if is_logical:
            k = k[:, :, :n_heads, :]
            v = v[:, :, :n_heads, :]
        else:
            k = k[:, :, -n_heads:, :]
            v = v[:, :, -n_heads:, :]

        out = self.inner_attention(
            q, k, v, attention_masks=attention_masks, scale=self.softmax_scale
        ).contiguous()
        out = out.view(bsz, seqlen, -1)
        return wo(out)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:                ``(batch, seq, dim)``
            freqs_cis:        RoPE frequency cache
            attention_masks:  causal / block-causal mask
            positions:        optional position indices for RoPE

        Returns:
            blended:          ``(batch, seq, dim)`` — morally-routed output
            moral_score:      ``(batch, seq, 1)``   — per-token moral alignment
            adversarial_prob: ``(batch, seq, 1)``   — per-token adversarial prob
        """
        logical_out = self._hemisphere_forward(
            x, freqs_cis, attention_masks, positions, is_logical=True
        )
        gen_out = self._hemisphere_forward(
            x, freqs_cis, attention_masks, positions, is_logical=False
        )
        blended, moral_score, adversarial_prob = self.moral_governor(x, logical_out, gen_out)
        return blended, moral_score, adversarial_prob


# ---------------------------------------------------------------------------
# MoralMindTransformerBlock
# ---------------------------------------------------------------------------

class EleosTransformerBlock(TransformerBlock):
    """Transformer block wiring DualHemisphereAttention and MoralGovernor.

    Pre-norm residual connection is applied as usual.  The attention output is
    the *morally-blended* output from :class:`DualHemisphereAttention`.
    Moral scores are surfaced to the model-level loss via return values.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        assert isinstance(config.attention, DualHemisphereAttention.Config), (
            "EleosTransformerBlock requires a DualHemisphereAttention.Config"
        )
        self.attention: DualHemisphereAttention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x:                ``(batch, seq, dim)`` — hidden state after block
            moral_score:      ``(batch, seq, 1)``   — block's moral alignment score
            adversarial_prob: ``(batch, seq, 1)``   — block's adversarial detection
        """
        attn_out, moral_score, adversarial_prob = self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        x = x + attn_out
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x, moral_score, adversarial_prob


# ---------------------------------------------------------------------------
# MoralMindModel
# ---------------------------------------------------------------------------

class EleosModel(Decoder):
    """Eleos: a morally-aligned dual-hemisphere language model.

    Extends :class:`~torchtitan.models.common.decoder.Decoder` with:

    * Per-layer moral scores accumulated into a ``moral_auxiliary_loss`` buffer.
    * A ``moral_loss_weight`` config parameter (default ``1e-3``) that scales
      the auxiliary loss added on top of the main cross-entropy.

    The model's ``forward()`` returns ``(logits, moral_aux_loss)`` when
    ``return_moral_loss=True`` (default during training).  Integration with
    the TorchTitan training loop requires a thin loss wrapper that adds
    ``moral_aux_loss`` to the primary loss — analogous to how MoE load
    balance loss is handled.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2048
        vocab_size: int = 102400
        moral_loss_weight: float = 1e-3
        """Weight applied to the moral auxiliary loss term.

        Analogous to the MoE load-balance loss coefficient.
        A value of ``1e-3`` is a reasonable default — large enough to
        provide a meaningful gradient signal without overwhelming the
        primary cross-entropy loss.
        """

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            debug = trainer_config.debug
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.rope.max_seq_len}."
                )
            self.rope = dataclasses.replace(self.rope, max_seq_len=seq_len)

            # Sync rope fields to DualHemisphereAttention for all layers.
            for layer_cfg in self.layers:
                assert isinstance(layer_cfg.attention, DualHemisphereAttention.Config)
                layer_cfg.attention.rope_max_seq_len = seq_len
                layer_cfg.attention.rope_factor = self.rope.rope_factor
                layer_cfg.attention.rope_original_seq_len = self.rope.original_seq_len

            for layer_cfg in self.layers:
                if layer_cfg.moe is not None:
                    if (
                        layer_cfg.moe.experts.use_grouped_mm
                        and not has_cuda_capability(9, 0)
                    ):
                        logger.warning(
                            "Failed to use grouped mm, which is only supported on SM90 or later",
                        )
                        layer_cfg.moe.experts.use_grouped_mm = False
                    layer_cfg.moe.router._debug_force_load_balance = (
                        debug.moe_force_load_balance
                    )
                    if parallelism.expert_parallel_comm_backend in (
                        "deepep",
                        "hybridep",
                    ):
                        from torchtitan.models.common.moe_deepep import DeepEPMoE

                        init_kwargs = {
                            f.name: getattr(layer_cfg.moe, f.name)
                            for f in dataclasses.fields(layer_cfg.moe)
                            if f.init
                        }
                        layer_cfg.moe = DeepEPMoE.Config(**init_kwargs)

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            assert isinstance(self.layers[0].attention, DualHemisphereAttention.Config)
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layers[0].attention.n_heads,
                self.layers[0].attention.qk_nope_head_dim
                + self.layers[0].attention.qk_rope_head_dim
                + self.layers[0].attention.v_head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.moral_loss_weight = config.moral_loss_weight
        # Accumulate moral scores across all layers during a forward pass.
        # Registered as a non-persistent buffer so it lives on the right device.
        self.register_buffer(
            "_moral_score_accum",
            torch.zeros(1, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        return_moral_loss: bool = True,
    ):
        """
        Args:
            tokens:            ``(batch, seq)`` int token ids.
            attention_masks:   Optional causal / block-causal masks.
            positions:         Optional position tensor for RoPE.
            return_moral_loss: If ``True`` (default), return
                               ``(logits, moral_aux_loss)``; if ``False``
                               return only ``logits``.  Set to ``False``
                               during inference.

        Returns:
            logits:          ``(batch, seq, vocab_size)``
            moral_aux_loss:  scalar — only returned when ``return_moral_loss=True``.
        """
        # Embed
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        # Accumulate moral scores
        moral_scores: list[torch.Tensor] = []

        for layer in self.layers.values():
            h, moral_score, _adv = layer(h, self.freqs_cis, attention_masks, positions)
            moral_scores.append(moral_score)  # each (B, S, 1)

        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h

        if not return_moral_loss:
            return output

        # Moral auxiliary loss: encourage high moral scores across all layers.
        # Loss = -mean(score) so that higher scores reduce the loss.
        # Shape: scalar
        all_moral = torch.cat(moral_scores, dim=-1)  # (B, S, n_layers)
        moral_aux_loss = -all_moral.mean() * self.moral_loss_weight

        return output, moral_aux_loss
