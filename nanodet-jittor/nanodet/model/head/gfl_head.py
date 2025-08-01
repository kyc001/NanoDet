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


def reduce_mean(tensor):
    """Reduce mean for distributed training."""
    # In Jittor, we don't need distributed reduction for single GPU training
    # This is a placeholder for future distributed training support
    return tensor


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        # Create project tensor as a constant (not a parameter)
        # This mimics PyTorch's register_buffer behavior
        self._project_data = jt.linspace(0, self.reg_max, self.reg_max + 1).data

    @property
    def project(self):
        """Get project tensor as a non-parameter tensor"""
        return jt.array(self._project_data)

    def execute(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        shape = x.size()
        x = jt.nn.softmax(x.reshape(*shape[:-1], 4, self.reg_max + 1), dim=-1)
        # Equivalent to F.linear in PyTorch
        x = jt.matmul(x, self.project.type_as(x)).reshape(*shape[:-1], 4)
        return x
