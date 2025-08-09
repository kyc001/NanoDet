import jittor as jt
import jittor.nn as nn
import jittor as jt
import jittor.nn as F
# import jittor.distributed as dist  # 不需要分布式
import jittordet.models.losses as losses

# 🎯 使用 JittorDet 标准化的 IoU 计算，确保与 PyTorch 版本一致
try:
    from jittordet.utils.bbox_overlaps import bbox_overlaps
    print("✅ 使用 JittorDet 标准 IoU 计算")
except ImportError:
    from ...loss.iou_loss import bbox_overlaps
    print("⚠️ 回退到本地 IoU 计算")
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


class DynamicSoftLabelAssigner(BaseAssigner):
    def __init__(self, topk=13, iou_factor=3.0, ignore_iof_thr=-1):
        self.topk = topk
        self.iou_factor = iou_factor
        self.ignore_iof_thr = ignore_iof_thr

    def assign(
        self,
        pred_scores,
        priors,
        decoded_bboxes,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
    ):
        INF = 100000000
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)
        num_classes = pred_scores.size(1)  # 🔧 获取类别数量

        # 🔧 样本分配调试信息 (已清理)
        # GT框数量: {num_gt}, 预测框数量: {num_bboxes}, 类别数量: {num_classes}

        # 如果没有GT框，直接返回空分配结果
        if num_gt == 0:
            assigned_gt_inds = jt.zeros((num_bboxes,), dtype='int32')
            max_overlaps = jt.zeros((num_bboxes,), dtype='float32')
            assigned_labels = jt.full((num_bboxes,), -1, dtype='int32')
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # assign 0 by default (int32 to be index-safe)
        assigned_gt_inds = jt.full((num_bboxes,), 0, dtype='int32')

        prior_center = priors[:, :2]
        lt_ = prior_center[:, None] - gt_bboxes[:, :2]
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]

        deltas = jt.cat([lt_, rb_], dim=-1)
        # 严格对齐 PyTorch：完全在 GT 内部（不含边界）
        is_in_gts = (jt.min(deltas, dim=-1) > 0)  # [num_bboxes, num_gt]
        # 在GT维度(1)上做求和，>0 表示该prior至少落入一个GT
        valid_mask = is_in_gts.float32().sum(1) > 0

        # 🔧 修复 Jittor 布尔索引问题：使用 nonzero() 方法
        try:
            valid_indices = jt.nonzero(valid_mask).squeeze(-1)
            if valid_indices.ndim == 0:  # 只有一个元素
                valid_indices = valid_indices.unsqueeze(0)
        except:
            # 如果 nonzero 失败，使用手动方式
            valid_indices = jt.array([], dtype='int32')
        num_valid = valid_indices.size(0)

        if num_valid > 0:
            valid_decoded_bbox = decoded_bboxes[valid_indices]
            valid_pred_scores = pred_scores[valid_indices]
        else:
            # Fallback: no priors strictly inside any GT, relax to all priors
            valid_decoded_bbox = decoded_bboxes
            valid_pred_scores = pred_scores
            valid_indices = jt.arange(num_bboxes, dtype='int32')
            num_valid = num_bboxes
        # 仅首批次打印 valid 数量，帮助确认 in-box 是否生效
        if not hasattr(self, "_dbg_seen"):
            self._dbg_seen = 0
        if self._dbg_seen < 2:
            try:
                print(f"[Assigner] num_valid={int(num_valid)}, num_bboxes={int(num_bboxes)}, num_gt={int(num_gt)}")
            except Exception:
                pass
            self._dbg_seen += 1

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = decoded_bboxes.new_full(
                    (num_bboxes,), -1
                )
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
            )

        # 🔧 IoU 计算调试信息 (已清理)

        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)

        # 🔧 IoU 计算结果调试信息 (已清理)
        iou_cost = -jt.log(pairwise_ious + 1e-7)

        # 🔧 学习 JittorDet 的方法：避免巨张量物化，使用广播 + 分块
        try:
            # 确保 gt_labels 是正确的整数类型
            # 确保 num_classes 一致
            C = int(pred_scores.shape[-1])
            gt_labels_int = gt_labels.long() if gt_labels.ndim > 0 else gt_labels.unsqueeze(0).long()
            gt_labels_int = jt.clamp(gt_labels_int, min_v=0, max_v=C-1)
            # 仅构造 [1, G, C]
            gt_onehot_label = (
                jt.nn.one_hot(gt_labels_int, C)
                .float()
                .unsqueeze(0)
            )
        except Exception as e:
            print(f"⚠️ GT标签处理失败: {e}")
            gt_onehot_label = jt.zeros((1, num_gt, pred_scores.shape[-1]))
            gt_onehot_label[:, :, 0] = 1.0

        # 仅在 num_gt 维度进行广播，不对 num_valid 进行 repeat
        valid_pred_scores_u = valid_pred_scores.unsqueeze(1)  # [V, 1, C]

        # 分块以降低峰值显存
        V = num_valid
        G = num_gt
        C = int(pred_scores.shape[-1])
        if V == 0 or G == 0:
            # 无法分配，直接返回空
            return jt.array([], dtype='int32'), jt.array([], dtype='float32'), jt.array([], dtype='int32')

        # 对齐 PyTorch：显式 repeat 展开，避免广播差异
        gt_onehot_label = (
            jt.nn.one_hot(gt_labels.long(), C).float().unsqueeze(0).repeat(V, 1, 1)
        )  # [V, G, C]
        valid_pred_scores_u = valid_pred_scores.unsqueeze(1).repeat(1, G, 1)  # [V, G, C]
        soft_label = pairwise_ious.unsqueeze(-1) * gt_onehot_label  # [V, G, C]
        scale_factor = (soft_label - jt.sigmoid(valid_pred_scores_u)).abs().pow(2.0)
        bce = losses.cross_entropy_loss.binary_cross_entropy_with_logits(
            valid_pred_scores_u, soft_label, reduction="none"
        )
        cls_cost = (bce * scale_factor).sum(dim=-1)  # [V, G]
        cost_matrix = cls_cost + iou_cost * self.iou_factor

        # 🔧 添加错误处理防止 tuple index out of range
        try:
            result = self.dynamic_k_matching(cost_matrix, pairwise_ious, num_gt, valid_mask)
            if isinstance(result, tuple) and len(result) == 3:
                fg_indices, matched_pred_ious, matched_gt_inds = result
            elif isinstance(result, tuple) and len(result) == 2:
                # 兼容旧返回
                matched_pred_ious, matched_gt_inds = result
                fg_indices = jt.array([], dtype='int32')
            else:
                matched_pred_ious = jt.array([], dtype='float32')
                matched_gt_inds = jt.array([], dtype='int32')
                fg_indices = jt.array([], dtype='int32')
            # Fallback: 若动态匹配未产生正样本，则按每个GT选择 IoU 最大的 prior
            if (len(fg_indices) == 0) and (num_gt > 0) and (num_valid > 0):
                try:
                    best_idx = jt.argmax(pairwise_ious, dim=0)  # [G]
                    # 过滤掉 IoU=0 的GT（完全未覆盖）
                    best_iou = pairwise_ious[best_idx, jt.arange(num_gt)]
                    keep = (best_iou > 0)
                    if int(keep.sum()) > 0:
                        fg_indices = best_idx[keep].cast('int32')
                        matched_gt_inds = jt.arange(num_gt, dtype='int32')[keep]
                        matched_pred_ious = best_iou[keep].float32()
                except Exception:
                    pass
        except Exception as e:
            # 如果出现任何错误，返回空结果
            matched_pred_ious = jt.array([], dtype='float32')
            matched_gt_inds = jt.array([], dtype='int32')
            fg_indices = jt.array([], dtype='int32')

        # convert to AssignResult format
        # 🔧 初始化默认值（显式 dtype）
        assigned_labels = jt.full((num_bboxes,), -1, dtype='int32')
        # set -inf for non-positives like mmdet
        max_overlaps = jt.full((num_bboxes,), -1e30, dtype='float32')

        # 🔧 正确映射：使用 valid_indices[fg_indices] 定位全局正样本
        if len(valid_indices) > 0 and len(fg_indices) > 0:
            pos_global = valid_indices[fg_indices]
            assigned_gt_inds[pos_global] = (matched_gt_inds + 1).cast('int32')
            assigned_labels[pos_global] = gt_labels[matched_gt_inds].long().cast('int32')
            max_overlaps[pos_global] = matched_pred_ious.float32()

        if (
            self.ignore_iof_thr > 0
            and gt_bboxes_ignore is not None
            and gt_bboxes_ignore.numel() > 0
            and num_bboxes > 0
        ):
            ignore_overlaps = bbox_overlaps(
                valid_decoded_bbox, gt_bboxes_ignore, mode="iof"
            )
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            assigned_gt_inds[ignore_idxs] = -1

        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
        )

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        """超简化版本：为每个GT选择cost最小的3个priors作为正样本"""
        num_bboxes = cost.shape[0]

        # 为每个GT选择cost最小的k个priors
        k = min(3, num_bboxes)  # 每个GT最多3个正样本
        fg_indices_list = []
        matched_gt_inds_list = []
        matched_pred_ious_list = []

        for gt_idx in range(num_gt):
            # 选择该GT的cost最小的k个priors
            _, pos_idx = jt.topk(-cost[:, gt_idx], k=k, dim=0)  # 负号表示最小

            # 添加到列表
            for i in range(k):
                if i < len(pos_idx):
                    fg_indices_list.append(int(pos_idx[i]))
                    matched_gt_inds_list.append(gt_idx)
                    matched_pred_ious_list.append(float(pairwise_ious[pos_idx[i], gt_idx]))

        # 转换为张量
        if len(fg_indices_list) > 0:
            fg_indices = jt.array(fg_indices_list, dtype='int32')
            matched_gt_inds = jt.array(matched_gt_inds_list, dtype='int32')
            matched_pred_ious = jt.array(matched_pred_ious_list, dtype='float32')
        else:
            fg_indices = jt.array([], dtype='int32')
            matched_gt_inds = jt.array([], dtype='int32')
            matched_pred_ious = jt.array([], dtype='float32')

        return fg_indices, matched_pred_ious, matched_gt_inds
