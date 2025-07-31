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

import math
import jittor as jt
from jittor import nn


def constant_init(module, val, bias=0):
    """Initialize module parameters with constant values.
    
    Args:
        module (nn.Module): Module to be initialized.
        val (float): The value to fill the weights.
        bias (float): The value to fill the bias. Defaults to 0.
    """
    if hasattr(module, 'weight') and module.weight is not None:
        jt.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        jt.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    """Initialize module parameters with Xavier initialization.
    
    Args:
        module (nn.Module): Module to be initialized.
        gain (float): An optional scaling factor. Defaults to 1.
        bias (float): The value to fill the bias. Defaults to 0.
        distribution (str): Either 'normal' or 'uniform'. Defaults to 'normal'.
    """
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            jt.init.xavier_uniform_(module.weight, gain=gain)
        else:
            jt.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        jt.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    """Initialize module parameters with normal distribution.
    
    Args:
        module (nn.Module): Module to be initialized.
        mean (float): The mean of the normal distribution. Defaults to 0.
        std (float): The standard deviation of the normal distribution. 
            Defaults to 1.
        bias (float): The value to fill the bias. Defaults to 0.
    """
    if hasattr(module, 'weight') and module.weight is not None:
        jt.init.gauss_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        jt.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    """Initialize module parameters with uniform distribution.
    
    Args:
        module (nn.Module): Module to be initialized.
        a (float): The lower bound of the uniform distribution. Defaults to 0.
        b (float): The upper bound of the uniform distribution. Defaults to 1.
        bias (float): The value to fill the bias. Defaults to 0.
    """
    if hasattr(module, 'weight') and module.weight is not None:
        jt.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        jt.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    """Initialize module parameters with Kaiming initialization.
    
    Args:
        module (nn.Module): Module to be initialized.
        a (int): The negative slope of the rectifier used after this layer
            (only used with 'leaky_relu'). Defaults to 0.
        mode (str): Either 'fan_in' or 'fan_out'. Defaults to 'fan_out'.
        nonlinearity (str): The non-linear function. Defaults to 'relu'.
        bias (float): The value to fill the bias. Defaults to 0.
        distribution (str): Either 'normal' or 'uniform'. Defaults to 'normal'.
    """
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            jt.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            jt.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        jt.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """Initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init
