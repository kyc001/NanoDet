# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .backbone import build_backbone
from .fpn import build_fpn
from .head import build_head
from .arch import NanoDetPlus

def build_model(cfg):
    """Build model from config.

    Args:
        cfg (dict): Model configuration.

    Returns:
        nn.Module: Built model.
    """
    model_cfg = cfg.copy()
    arch_name = model_cfg.pop("name")

    if arch_name == "NanoDetPlus":
        return NanoDetPlus(**model_cfg)
    else:
        raise NotImplementedError(f"Model {arch_name} not implemented")

__all__ = [
    "build_backbone",
    "build_fpn",
    "build_head",
    "build_model",
    "NanoDetPlus",
]
