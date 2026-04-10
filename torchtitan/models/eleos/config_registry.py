# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Eleos Trainer configs — wires Trainer.Config for each named model size.

These are consumed by the torchtitan training script via the --config flag,
e.g.:

    MODEL=eleos CONFIG=eleos_debugmodel ./run_train.sh
"""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.quantization.float8 import (
    Float8GroupedMMConverter,
    Float8LinearConverter,
)
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.trainer import Trainer

from . import model_registry


def eleos_debugmodel() -> Trainer.Config:
    """Small smoke-test config — runs in minutes on a single GPU."""
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=1024,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def eleos_debugmodel_flex_attn() -> Trainer.Config:
    """Debug config using FlexAttention (block-causal masking)."""
    config = eleos_debugmodel()
    config.model_spec = model_registry("debugmodel_flex_attn")
    return config


def eleos_16b() -> Trainer.Config:
    """16B parameter Eleos training config."""
    return Trainer.Config(
        # Reuse the DeepSeek-compatible 16B tokenizer; swap for a custom one
        # once you have Eleos-specific training data prepared.
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
        model_spec=model_registry("16B"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=8,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=100),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def eleos_236b() -> Trainer.Config:
    """236B parameter Eleos training config (multi-node)."""
    return Trainer.Config(
        hf_assets_path="./assets/hf/eleos-tokenizer",
        model_spec=model_registry("236B"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(lr=1.5e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=8,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        compile=CompileConfig(enable=True, components=["loss"]),
        model_converters=ModelConvertersContainer.Config(
            converters=[
                Float8LinearConverter.Config(filter_fqns=["output", "router.gate"]),
                Float8GroupedMMConverter.Config(fqns=["experts"]),
            ],
        ),
    )


def eleos_671b() -> Trainer.Config:
    """671B parameter Eleos training config (large cluster)."""
    return Trainer.Config(
        hf_assets_path="./assets/hf/eleos-tokenizer",
        model_spec=model_registry("671B"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(lr=1.0e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=50000,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=8,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        compile=CompileConfig(enable=True, components=["loss"]),
        model_converters=ModelConvertersContainer.Config(
            converters=[
                Float8LinearConverter.Config(filter_fqns=["output", "router.gate"]),
                Float8GroupedMMConverter.Config(fqns=["experts"]),
            ],
        ),
    )
