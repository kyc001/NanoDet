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

import jittor as jt
from jittor import nn


class Swish(nn.Module):
    """Swish activation function.
    
    Swish(x) = x * sigmoid(x)
    """
    
    def __init__(self):
        super(Swish, self).__init__()
        
    def execute(self, x):
        return x * jt.sigmoid(x)


class HardSwish(nn.Module):
    """Hard Swish activation function.
    
    HardSwish(x) = x * hard_sigmoid(x)
    where hard_sigmoid(x) = max(0, min(1, (x + 3) / 6))
    """
    
    def __init__(self):
        super(HardSwish, self).__init__()
        
    def execute(self, x):
        return x * jt.clamp((x + 3.0) / 6.0, 0.0, 1.0)


class Mish(nn.Module):
    """Mish activation function.
    
    Mish(x) = x * tanh(softplus(x))
    """
    
    def __init__(self):
        super(Mish, self).__init__()
        
    def execute(self, x):
        return x * jt.tanh(jt.nn.softplus(x))


# Dictionary of available activation layers
act_layers = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Swish": Swish,
    "HardSwish": HardSwish,
    "Mish": Mish,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
}


def build_activation_layer(cfg):
    """Build activation layer.
    
    Args:
        cfg (dict or str): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.
            
    Returns:
        nn.Module: Created activation layer.
    """
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()
    
    layer_type = cfg_.pop("type")
    if layer_type not in act_layers:
        raise KeyError(f"Unrecognized activation type {layer_type}")
    else:
        activation = act_layers[layer_type]
        
    layer = activation(**cfg_)
    return layer
