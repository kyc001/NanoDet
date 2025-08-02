## [Jittor 迁移] ##
# 导入 jittor 相关的模块
import jittor as jt
# NumPy 依赖保持不变
import numpy as np

# 注意：nanodet.util.util_mixins.NiceRepr 是一个外部依赖
# 这里假定您项目中已有其 Jittor 兼容版本，或者它是一个用于打印的简单 mixin
from nanodet.util import util_mixins


class AssignResult(util_mixins.NiceRepr):
    """
    Stores assignments between predicted and truth boxes. (Jittor Version)
    """

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        self._extra_properties = {}

    @property
    def num_preds(self):
        return len(self.gt_inds)

    def set_extra_property(self, key, value):
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        basic_info = {
            "num_gts": self.num_gts,
            "num_preds": self.num_preds,
            "gt_inds": self.gt_inds,
            "max_overlaps": self.max_overlaps,
            "labels": self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

    def __nice__(self):
        # 此方法基本与框架无关，可以直接使用
        parts = []
        parts.append(f"num_gts={self.num_gts!r}")
        if self.gt_inds is None:
            parts.append(f"gt_inds={self.gt_inds!r}")
        else:
            parts.append(f"gt_inds.shape={tuple(self.gt_inds.shape)!r}")
        if self.max_overlaps is None:
            parts.append(f"max_overlaps={self.max_overlaps!r}")
        else:
            parts.append("max_overlaps.shape=" f"{tuple(self.max_overlaps.shape)!r}")
        if self.labels is None:
            parts.append(f"labels={self.labels!r}")
        else:
            parts.append(f"labels.shape={tuple(self.labels.shape)!r}")
        return ", ".join(parts)

    @classmethod
    def random(cls, **kwargs):
        # NumPy 相关逻辑保持不变
        rng = kwargs.get("rng", None)
        num_gts = kwargs.get("num_gts", None)
        num_preds = kwargs.get("num_preds", None)
        p_ignore = kwargs.get("p_ignore", 0.3)
        p_assigned = kwargs.get("p_assigned", 0.7)
        p_use_label = kwargs.get("p_use_label", 0.5)
        num_classes = kwargs.get("p_use_label", 3)

        if rng is None:
            rng = np.random.mtrand._rand
        elif isinstance(rng, int):
            rng = np.random.RandomState(rng)
        
        if num_gts is None:
            num_gts = rng.randint(0, 8)
        if num_preds is None:
            num_preds = rng.randint(0, 16)

        if num_gts == 0:
            ## [Jittor 迁移] ##
            # torch.zeros -> jt.zeros
            # torch.float32 -> 'float32', torch.int64 -> 'int64'
            max_overlaps = jt.zeros(num_preds, dtype='float32')
            gt_inds = jt.zeros(num_preds, dtype='int64')
            if p_use_label is True or p_use_label < rng.rand():
                labels = jt.zeros(num_preds, dtype='int64')
            else:
                labels = None
        else:
            # Create an overlap for each predicted box
            ## [Jittor 迁移] ##
            # torch.from_numpy -> jt.array
            max_overlaps = jt.array(rng.rand(num_preds))

            # Construct gt_inds for each predicted box
            is_assigned = jt.array(rng.rand(num_preds) < p_assigned)
            n_assigned = min(num_preds, min(num_gts, is_assigned.sum().item()))

            assigned_idxs = np.where(is_assigned.numpy())[0] # .numpy() to use with np.where
            rng.shuffle(assigned_idxs)
            assigned_idxs = assigned_idxs[0:n_assigned]
            assigned_idxs.sort()

            # is_assigned 是 Jittor Var，不能直接用 is_assigned[:] 修改，需要重新创建
            is_assigned_np = np.zeros_like(is_assigned.numpy())
            is_assigned_np[assigned_idxs] = True
            is_assigned = jt.array(is_assigned_np)

            is_ignore = (jt.array(rng.rand(num_preds) < p_ignore)) & is_assigned

            gt_inds = jt.zeros(num_preds, dtype='int64')

            true_idxs = np.arange(num_gts)
            rng.shuffle(true_idxs)
            true_idxs = jt.array(true_idxs)
            gt_inds[is_assigned] = true_idxs[:n_assigned]
            
            # 这行代码似乎有误，它会覆盖上面的分配，但我们按原逻辑迁移
            gt_inds_random = jt.array(rng.randint(1, num_gts + 1, size=num_preds))
            gt_inds[is_assigned] = gt_inds_random[is_assigned]

            gt_inds[is_ignore] = -1
            gt_inds[~is_assigned] = 0
            max_overlaps[~is_assigned] = 0

            if p_use_label is True or p_use_label < rng.rand():
                if num_classes == 0:
                    labels = jt.zeros(num_preds, dtype='int64')
                else:
                    labels = jt.array(rng.randint(0, num_classes, size=num_preds))
                    labels[~is_assigned] = 0
            else:
                labels = None
        
        self = cls(num_gts, gt_inds, max_overlaps, labels)
        return self

    def add_gt_(self, gt_labels):
        """Add ground truth as assigned results."""
        ## [Jittor 迁移] ##
        # torch.arange -> jt.arange. 去掉 device 参数
        self_inds = jt.arange(1, len(gt_labels) + 1, dtype='int64')
        # torch.cat -> jt.concat
        self.gt_inds = jt.concat([self_inds, self.gt_inds])
        
        ## [Jittor 迁移] ##
        # .new_ones() -> jt.ones()
        ones_tensor = jt.ones(len(gt_labels), dtype=self.max_overlaps.dtype)
        self.max_overlaps = jt.concat([ones_tensor, self.max_overslaps])

        if self.labels is not None:
            self.labels = jt.concat([gt_labels, self.labels])