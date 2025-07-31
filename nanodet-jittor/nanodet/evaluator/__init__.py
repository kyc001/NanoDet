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

from .coco_detection import CocoDetectionEvaluator


def build_evaluator(cfg, dataset):
    """Build evaluator from config.
    
    Args:
        cfg (dict): Evaluator configuration.
        dataset: Dataset instance.
        
    Returns:
        Evaluator: Built evaluator.
    """
    evaluator_cfg = cfg.copy()
    name = evaluator_cfg.pop("name")
    
    if name == "CocoDetectionEvaluator":
        return CocoDetectionEvaluator(dataset, **evaluator_cfg)
    else:
        raise NotImplementedError(f"Evaluator {name} not implemented")


__all__ = [
    "CocoDetectionEvaluator",
    "build_evaluator",
]
