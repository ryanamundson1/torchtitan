# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Eleos parallelization strategy.

Extends the DeepSeek V3 parallelization with awareness of:
* The dual Q projection matrices (logical + generative hemispheres).
* The MoralGovernor sub-modules (replicated — they are small).
* The AdversarialDetector (replicated).

TP sharding follows the same Colwise/Rowwise convention as DeepSeek V3:
* wq_logical, wq_gen, wkv_b  → Colwise (split output/head dim)
* wo_logical, wo_gen          → Rowwise (reduce across head shards)
* wkv_a, kv_norm              → NoParallel (shared compression)
* MoralGovernor weights       → Replicate (tiny; not worth sharding)
"""

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.components.quantization.float8 import find_float8_linear_config
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile_sparse
from torchtitan.distributed.context_parallel import apply_cp_to_attention_module
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp, NoParallel
from torchtitan.models.llama3.parallelize import apply_replicate
from torchtitan.models.llama4.parallelize import apply_fsdp, apply_moe_ep_tp
from torchtitan.protocols import ModelConvertersContainer
from torchtitan.tools.logging import logger

from .model import DualHemisphereAttention, EleosModel


def parallelize_eleos(
    model: EleosModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Parallelise EleosModel across TP / EP / FSDP / CP dimensions."""

    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), (
        f"Sequence length {training.seq_len} must be divisible by the product of "
        f"TP degree ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp})."
    )

    if parallel_dims.tp_enabled:
        float8_config = find_float8_linear_config(model_converters.converters)
        enable_sp = parallelism.enable_sequence_parallel
        tp_mesh = parallel_dims.get_mesh("tp")
        _apply_dha_tp(
            model,
            tp_mesh,
            enable_loss_parallel=not parallelism.disable_loss_parallel,
            enable_cp=parallel_dims.cp_enabled,
            enable_sp=enable_sp,
        )
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    comm_backend = parallelism.expert_parallel_comm_backend
    if comm_backend in ("deepep", "hybridep"):
        if not parallel_dims.ep_enabled:
            raise ValueError(
                f"{comm_backend.upper()} requires expert parallelism (ep_degree > 1)."
            )
        if parallel_dims.etp_enabled:
            raise NotImplementedError(
                f"{comm_backend.upper()} with Expert Tensor Parallelism (ETP) is not supported."
            )

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        from torchtitan.components.quantization import find_pad_multiple
        pad_multiple = find_pad_multiple(model_converters.converters)
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
            comm_backend=comm_backend,
            hybridep_non_blocking_expert_capacity_factor=parallelism.hybridep_non_blocking_expert_capacity_factor,
            pad_multiple=pad_multiple,
        )

    if parallel_dims.cp_enabled:
        apply_cp_to_attention_module(
            # Each block's DualHemisphereAttention shares one inner_attention kernel
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    if model_compile_enabled:
        apply_compile_sparse(model, compile_config, parallel_dims.ep_enabled)

    dp_mesh: DeviceMesh | None = None
    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
        edp_mesh_names = (
            ["dp_replicate", "efsdp"] if parallel_dims.dp_replicate_enabled else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )
        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")
        if training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        apply_replicate(
            model,
            parallel_dims.get_mesh("dp_replicate"),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )

    return model


def _apply_dha_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    enable_loss_parallel: bool,
    enable_cp: bool,
    enable_sp: bool = True,
):
    """Apply tensor parallelism to MoralMindModel layers."""
    sp_layout = Shard(1) if enable_sp else Replicate()

    # Parallelise embedding, root norm, and output linear
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=sp_layout,
                use_local_output=enable_sp,
            ),
            "norm": SequenceParallel() if enable_sp else NoParallel(),
            "output": ColwiseParallel(
                input_layouts=sp_layout,
                output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
                use_local_output=not enable_loss_parallel,
            ),
        },
    )

    positions_sharding = Replicate() if enable_cp else None
    norm_plan = SequenceParallel() if enable_sp else NoParallel()
    rowwise_output_plan = RowwiseParallel(output_layouts=sp_layout, use_local_output=enable_sp)
    attention_kernel_plan = PrepareModuleInput(
        input_layouts=(Shard(1), Shard(1), Shard(1)),
        desired_input_layouts=(Shard(1), Shard(1), Shard(1)),
        use_local_output=True,
    )

    for transformer_block in model.layers.values():
        attn = transformer_block.attention
        assert isinstance(attn, DualHemisphereAttention)

        layer_plan = {
            "attention_norm": norm_plan,
            "attention": PrepareModuleInput(
                input_layouts=(sp_layout, Replicate(), None, positions_sharding),
                desired_input_layouts=(Replicate(), Replicate(), None, positions_sharding),
            ),
            # Shared KV compression — no sharding (small, shared across hemispheres)
            "attention.wkv_a": NoParallel(),
            "attention.kv_norm": NoParallel(),
            # KV B expands to full head space — shard colwise
            "attention.wkv_b": ColwiseParallel(use_local_output=False),
            # Inner attention kernel
            "attention.inner_attention": attention_kernel_plan,
            # Output projections — rowwise (reduce head shards back to dim)
            "attention.wo_logical": rowwise_output_plan,
            "attention.wo_gen": rowwise_output_plan,
            # MoralGovernor is small — replicate across TP ranks
            "attention.moral_governor": NoParallel(),
            "ffn_norm": norm_plan,
        }

        # Q projections: shard colwise (output = head dim)
        if attn.q_lora_rank == 0:
            layer_plan["attention.wq_logical"] = ColwiseParallel(use_local_output=False)
            layer_plan["attention.wq_gen"] = ColwiseParallel(use_local_output=False)
        else:
            layer_plan.update({
                "attention.wq_a_logical": NoParallel(),
                "attention.wq_b_logical": ColwiseParallel(use_local_output=False),
                "attention.q_norm_logical": NoParallel(),
                "attention.wq_a_gen": NoParallel(),
                "attention.wq_b_gen": ColwiseParallel(use_local_output=False),
                "attention.q_norm_gen": NoParallel(),
            })

        if not transformer_block.moe_enabled:
            layer_plan.update({
                "feed_forward": PrepareModuleInput(
                    input_layouts=(sp_layout,),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": ColwiseParallel(),
                "feed_forward.w2": rowwise_output_plan,
                "feed_forward.w3": ColwiseParallel(),
            })

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logger.info("Applied Tensor Parallelism to Eleos model")
