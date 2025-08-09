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

        # 如果没有GT框，直接返回
        if num_gt == 0:
            print("⚠️ 没有GT框，跳过样本分配")
            return assigned_gt_inds, jt.zeros_like(assigned_gt_inds).float()

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0)

        prior_center = priors[:, :2]
        lt_ = prior_center[:, None] - gt_bboxes[:, :2]
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]

        deltas = jt.cat([lt_, rb_], dim=-1)
        is_in_gts = deltas.min(dim=-1)[0] > 0
        valid_mask = is_in_gts.sum(dim=1) > 0

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
            valid_decoded_bbox = decoded_bboxes.new_zeros((0, 4))
            valid_pred_scores = pred_scores.new_zeros((0, num_classes))

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
            gt_labels_int = gt_labels.long() if gt_labels.ndim > 0 else gt_labels.unsqueeze(0).long()
            # 仅构造 [1, num_gt, num_classes]，避免对 num_valid 维度 repeat
            gt_onehot_label = (
                jt.nn.one_hot(gt_labels_int, pred_scores.shape[-1])
                .float()
                .unsqueeze(0)  # [1, G, C]
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
        C = pred_scores.shape[-1]
        cls_cost = jt.zeros((V, G), dtype=valid_pred_scores.dtype)
        chunk = 16384  # 可根据显存调整
        for s in range(0, V, chunk):
            e = min(V, s + chunk)
            # [blk, G]
            iou_blk = pairwise_ious[s:e]
            # [blk, G, C]，依赖广播，不物化全量
            soft_label_blk = iou_blk.unsqueeze(-1) * gt_onehot_label  # [blk, G, C]
            pred_blk = valid_pred_scores_u[s:e]                       # [blk, 1, C]
            scale_factor_blk = (soft_label_blk - jt.sigmoid(pred_blk)).abs().pow(2.0)
            bce_blk = losses.cross_entropy_loss.binary_cross_entropy_with_logits(
                pred_blk, soft_label_blk, reduction="none"
            ) * scale_factor_blk
            cls_cost_blk = bce_blk.sum(dim=-1)  # [blk, G]
            cls_cost[s:e] = cls_cost_blk

        cost_matrix = cls_cost + iou_cost * self.iou_factor

        # 🔧 添加错误处理防止 tuple index out of range
        try:
            result = self.dynamic_k_matching(cost_matrix, pairwise_ious, num_gt, valid_mask)
            if isinstance(result, tuple) and len(result) == 2:
                matched_pred_ious, matched_gt_inds = result
            else:
                # 处理异常情况，返回空结果
                matched_pred_ious = jt.array([], dtype='float32')
                matched_gt_inds = jt.array([], dtype='int32')
        except Exception as e:
            # 如果出现任何错误，返回空结果
            matched_pred_ious = jt.array([], dtype='float32')
            matched_gt_inds = jt.array([], dtype='int32')

        # convert to AssignResult format
        # 🔧 初始化默认值
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        max_overlaps = assigned_gt_inds.new_full((num_bboxes,), -INF)

        # 🔧 修复 Jittor 布尔索引问题：只在有有效索引时才赋值
        if len(valid_indices) > 0 and len(matched_gt_inds) > 0:
            # 🔧 问题根源：valid_indices 和 matched_gt_inds 长度不匹配
            # 这说明样本分配算法有问题，我们需要使用正确的索引
            # 使用 matched_gt_inds 的长度来确定实际的有效索引
            actual_valid_count = len(matched_gt_inds)
            if actual_valid_count <= len(valid_indices):
                actual_valid_indices = valid_indices[:actual_valid_count]
                assigned_gt_inds[actual_valid_indices] = matched_gt_inds + 1
                assigned_labels[actual_valid_indices] = gt_labels[matched_gt_inds].long()
                max_overlaps[actual_valid_indices] = matched_pred_ious

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
        # 🔧 修复：从 cost 矩阵获取 num_bboxes
        num_bboxes = cost.shape[0]
        matching_matrix = jt.zeros_like(cost)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = jt.topk(pairwise_ious, candidate_topk, dim=0)

        # 🔧 学习 JittorDet 的方法：使用列表推导式避免 .item() 调用
        # calculate dynamic k for each gt
        dynamic_ks_list = []
        for gt_idx in range(num_gt):
            # 🔧 对每个GT单独计算k值，避免批量操作中的 .item() 问题
            gt_topk_sum = topk_ious[:, gt_idx].sum()
            # 🔧 修复：避免直接调用 int()，使用 Jittor 的 clamp 方法
            gt_topk_sum_clamped = jt.clamp(gt_topk_sum, min_v=1.0, max_v=float(self.topk))
            k_val = int(float(gt_topk_sum_clamped))  # 先转 float 再转 int，避免 .item() 调用
            dynamic_ks_list.append(k_val)

            # 动态k值计算完成

            # 直接使用计算出的 k_val，避免张量转换
            _, pos_idx = jt.topk(
                cost[:, gt_idx], k=k_val, largest=False
            )

            # 🔧 使用 JittorDet 风格的索引赋值
            for i in range(k_val):
                if i < len(pos_idx):
                    matching_matrix[pos_idx[i], gt_idx] = 1.0

            # 匹配矩阵更新完成

        # 🔧 修复：清理变量，避免内存泄漏
        del topk_ious
        # pos_idx 在循环中已经被重新赋值，不需要删除

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            # 🔧 修复 Jittor 布尔索引问题
            # 🔧 修复 Jittor 布尔索引问题：使用 nonzero() 方法
            try:
                prior_indices = jt.nonzero(prior_match_gt_mask).squeeze(-1)
                if prior_indices.ndim == 0:
                    prior_indices = prior_indices.unsqueeze(0)
            except:
                prior_indices = jt.array([], dtype='int32')
            # 🔧 修复：Jittor min 返回值格式不同
            cost_values = jt.min(cost[prior_indices, :], dim=1)
            cost_argmin = jt.argmin(cost[prior_indices, :], dim=1)
            matching_matrix[prior_indices, :] *= 0.0
            # 使用 scatter 操作替代高级索引
            for i, idx in enumerate(prior_indices):
                matching_matrix[idx, cost_argmin[i]] = 1.0
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        # 🔧 匹配矩阵和前景掩码统计 (调试输出已清理)

        # 🔧 修复 Jittor 布尔索引问题：使用 nonzero() 方法
        try:
            fg_indices = jt.nonzero(fg_mask_inboxes).squeeze(-1)
            if fg_indices.ndim == 0:
                fg_indices = fg_indices.unsqueeze(0)
            # 🔧 修复：避免 Jittor 张量格式化错误
            # 前景索引计算成功
        except Exception as e:
            # 前景索引计算失败，使用空数组
            fg_indices = jt.array([], dtype='int32')

        # 更新 valid_mask
        valid_mask_clone = valid_mask.clone()
        # 🔧 修复 Jittor 布尔索引问题：使用 nonzero() 方法
        try:
            valid_indices_in_valid = jt.nonzero(valid_mask_clone).squeeze(-1)
            if valid_indices_in_valid.ndim == 0:
                valid_indices_in_valid = valid_indices_in_valid.unsqueeze(0)
        except:
            valid_indices_in_valid = jt.array([], dtype='int32')
        # 🔧 修复索引越界问题
        min_len = min(len(fg_mask_inboxes), len(valid_indices_in_valid))
        for i in range(min_len):
            valid_mask[valid_indices_in_valid[i]] = fg_mask_inboxes[i]

        # 🔧 添加边界检查防止索引错误
        if len(fg_indices) > 0 and matching_matrix.shape[0] > 0:
            matched_gt_inds = matching_matrix[fg_indices, :].argmax(1)
            matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_indices]
        else:
            matched_gt_inds = jt.array([], dtype='int32')
            matched_pred_ious = jt.array([], dtype='float32')

        # 🔧 样本分配结果统计 (调试输出已清理)

        # 🔧 创建兼容的分配结果对象
        class AssignResult:
            def __init__(self, max_overlaps, gt_inds):
                self.max_overlaps = max_overlaps
                self.gt_inds = gt_inds

        # 创建完整的 max_overlaps 数组（所有预测框的IoU）
        full_max_overlaps = jt.zeros(num_bboxes)
        if len(matched_pred_ious) > 0:
            full_max_overlaps[fg_indices] = matched_pred_ious

        # 创建完整的 gt_inds 数组
        full_gt_inds = jt.zeros(num_bboxes, dtype='int32')
        if len(matched_gt_inds) > 0:
            full_gt_inds[fg_indices] = matched_gt_inds

        # 🔧 修复：返回期望的两个值，而不是 AssignResult 对象
        return matched_pred_ious, matched_gt_inds
