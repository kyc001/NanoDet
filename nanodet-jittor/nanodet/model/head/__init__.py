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

import copy

from .nanodet_plus_head import NanoDetPlusHead
from .simple_conv_head import SimpleConvHead


def build_head(cfg):
    """Build detection head.

    Args:
        cfg (dict): Configuration dict for head.

    Returns:
        nn.Module: Built detection head.
    """
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop("name")

    if name == "NanoDetPlusHead":
        return NanoDetPlusHead(**head_cfg)
    elif name == "SimpleConvHead":
        return SimpleConvHead(**head_cfg)
    else:
        raise NotImplementedError(f"Head {name} not implemented")
