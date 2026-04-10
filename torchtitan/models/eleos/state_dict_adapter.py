# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter
from .model import EleosModel

class EleosStateDictAdapter(StateDictAdapter):
    """
    Adapter for Eleos checkpointing and weight loading.
    
    Since Eleos is an experimental model internally authored in torchtitan,
    there is no pre-existing HuggingFace representation. Therefore, we provide
    identity transformations for now to satisfy the ModelSpec interface.
    """

    def __init__(self, model_config: EleosModel.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format."""
        return state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Obtain native model state dict from HuggingFace format."""
        return hf_state_dict
