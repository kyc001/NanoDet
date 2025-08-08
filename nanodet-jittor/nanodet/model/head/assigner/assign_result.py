# Copyright (c) OpenMMLab. All rights reserved.
import jittor as jt


class AssignResult:

    def __init__(self, num_gts: int, gt_inds: jt.Var, max_overlaps: jt.Var,
                 labels: jt.Var) -> None:
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        # Interface for possible user-defined properties
        self._extra_properties = {}

    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_inds)

    def set_extra_property(self, key, value):
        """Set user-defined new property."""
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        """Get user-defined property."""
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {
            'num_gts': self.num_gts,
            'num_preds': self.num_preds,
            'gt_inds': self.gt_inds,
            'max_overlaps': self.max_overlaps,
            'labels': self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

    def add_gt_(self, gt_labels):
        """Add ground truth as assigned results.

        Args:
            gt_labels (jt.Tensor): Labels of gt boxes
        """
        self_inds = jt.arange(1, len(gt_labels) + 1, dtype=jt.int64)
        self.gt_inds = jt.concat([self_inds, self.gt_inds])

        self.max_overlaps = jt.concat([
            jt.ones(len(gt_labels), dtype=self.max_overlaps.dtype),
            self.max_overlaps
        ])

        self.labels = jt.concat([gt_labels, self.labels])
