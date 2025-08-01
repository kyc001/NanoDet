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

from .logger import get_logger, setup_logger
from .config import Config, load_config, save_config, merge_configs, DEFAULT_CONFIG
from .checkpoint import save_checkpoint, load_checkpoint, load_pytorch_checkpoint, get_latest_checkpoint
from .metrics import calculate_map, COCOEvaluator, SimpleEvaluator, calculate_iou, calculate_ap

__all__ = [
    'get_logger', 'setup_logger',
    'Config', 'load_config', 'save_config', 'merge_configs', 'DEFAULT_CONFIG',
    'save_checkpoint', 'load_checkpoint', 'load_pytorch_checkpoint', 'get_latest_checkpoint',
    'calculate_map', 'COCOEvaluator', 'SimpleEvaluator', 'calculate_iou', 'calculate_ap'
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'NanoDet Jittor Team'
__description__ = 'NanoDet implementation in Jittor framework'

__all__ = [
    "get_logger",
]

import jittor as jt


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def bbox2distance(points, bbox, max_dis=None, eps=1e-8):
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return jt.stack([left, top, right, bottom], -1)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return jt.stack([x1, y1, x2, y2], -1)


def overlay_bbox_cv(img, dets, class_names, score_thresh):
    """Overlay bounding boxes on image using OpenCV.
    
    Args:
        img: Input image
        dets: Detection results
        class_names: List of class names
        score_thresh: Score threshold
        
    Returns:
        Image with overlaid bounding boxes
    """
    # This is a placeholder for visualization function
    # Full implementation would require OpenCV
    return img


# Import partial for multi_apply
from functools import partial

__all__ = [
    "multi_apply",
    "bbox2distance", 
    "distance2bbox",
    "overlay_bbox_cv",
]
