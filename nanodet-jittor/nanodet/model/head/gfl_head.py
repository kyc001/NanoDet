# GFLHead and its dependencies migrated from PyTorch to Jittor
# Author: Gemini

import math
import cv2
import numpy as np
import jittor as jt
from jittor import nn
from nanodet.util import (
    bbox2distance,
    distance2bbox,
    images_to_levels,
    multi_apply,
    overlay_bbox_cv,
)

from ...data.transform.warp import warp_boxes
from ..loss.gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from ..loss.iou_loss import GIoULoss, bbox_overlaps
from ..module.conv import ConvModule
from ..module.init_weights import normal_init
from ..module.nms import multiclass_nms
from ..module.scale import Scale
from .assigner.atss_assigner import ATSSAssigner

def reduce_mean(tensor):
    """Jittor version of reduce_mean for distributed training."""
    if not jt.mpi or not jt.mpi.is_initialized():
        return tensor
    tensor = tensor.clone()
    jt.mpi.all_reduce(tensor / jt.mpi.world_size(), op='sum')
    return tensor

class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution. (Jittor Version)"""
    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.project = jt.linspace(0, self.reg_max, self.reg_max + 1)

    def execute(self, x):
        """Forward feature from the regression head to get integral result."""
        shape = x.shape
        reshaped_x = x.reshape(shape[:-1] + (4, self.reg_max + 1))
        softmax_x = jt.nn.softmax(reshaped_x, dim=-1)
        projected_x = jt.nn.linear(softmax_x, self.project.cast(x.dtype))
        return projected_x.reshape(shape[:-1] + (4,))


class GFLHead(nn.Module):
    """
    Generalized Focal Loss Head (Jittor 版本)
    """
    def __init__(
        self, num_classes, loss, input_channel, feat_channels=256,
        stacked_convs=4, octave_base_scale=4, strides=[8, 16, 32],
        conv_cfg=None, norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        reg_max=16, ignore_iof_thr=-1, **kwargs
    ):
        super(GFLHead, self).__init__()
        # --- 成员变量定义 (与 PyTorch 相同) ---
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.grid_cell_scale = octave_base_scale
        self.strides = strides
        self.reg_max = reg_max
        self.loss_cfg = loss
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_sigmoid = self.loss_cfg.loss_qfl.use_sigmoid
        self.ignore_iof_thr = ignore_iof_thr
        self.cls_out_channels = num_classes if self.use_sigmoid else num_classes + 1

        # --- 依赖模块实例化 (假设已有 Jittor 版本) ---
        self.assigner = ATSSAssigner(topk=9, ignore_iof_thr=ignore_iof_thr)
        self.distribution_project = Integral(self.reg_max)
        self.loss_qfl = QualityFocalLoss(
            use_sigmoid=self.use_sigmoid, beta=self.loss_cfg.loss_qfl.beta,
            loss_weight=self.loss_cfg.loss_qfl.loss_weight)
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=self.loss_cfg.loss_dfl.loss_weight)
        self.loss_bbox = GIoULoss(loss_weight=self.loss_cfg.loss_bbox.loss_weight)
        
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        # [遷移] inplace=True 在 Jittor 中由編譯器自動優化，無需設置
        self.relu = nn.ReLU()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            # 假设 ConvModule 已被迁移
            self.cls_convs.append(
                ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                           conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                           conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        # [遷移] nn.Conv2d -> nn.Conv
        self.gfl_cls = nn.Conv(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.gfl_reg = nn.Conv(self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
        # 假设 Scale 已被迁移
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        # 假设 normal_init 已被迁移
        for m in self.cls_convs: normal_init(m.conv, std=0.01)
        for m in self.reg_convs: normal_init(m.conv, std=0.01)
        bias_cls = -4.595
        normal_init(self.gfl_cls, std=0.01, bias=bias_cls)
        normal_init(self.gfl_reg, std=0.01)

    # [遷移] forward -> execute
    def execute(self, feats):
        # [遷移] 移除 PyTorch 特有的 ONNX 导出逻辑
        outputs = []
        for x, scale in zip(feats, self.scales):
            cls_feat, reg_feat = x, x
            for cls_conv in self.cls_convs: cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs: reg_feat = reg_conv(reg_feat)
            cls_score = self.gfl_cls(cls_feat)
            bbox_pred = scale(self.gfl_reg(reg_feat)).float32()
            # [遷移] torch.cat -> jt.concat
            output = jt.concat([cls_score, bbox_pred], dim=1)
            # [遷移] .flatten(start_dim=2) -> .flatten(2)
            outputs.append(output.flatten(start_dim=2))
        return jt.concat(outputs, dim=2).permute(0, 2, 1)

    def loss(self, preds, gt_meta):
        # [遷移] .split 在 Jittor 中用法相同
        cls_scores, bbox_preds = preds.split([self.num_classes, 4 * (self.reg_max + 1)], dim=-1)
        # [遷移] .device 在 Jittor 中是全局的，無需從張量獲取
        gt_bboxes = gt_meta["gt_bboxes"]
        gt_labels = gt_meta["gt_labels"]
        input_height, input_width = gt_meta["img"].shape[2:]
        featmap_sizes = [(math.ceil(input_height / s), math.ceil(input_width / s)) for s in self.strides]

        cls_reg_targets = self.target_assign(
            cls_scores, bbox_preds, featmap_sizes, gt_bboxes,
            gt_meta["gt_bboxes_ignore"], gt_labels)
        if cls_reg_targets is None: return None

        (cls_preds_list, reg_preds_list, grid_cells_list, labels_list,
         label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        # [遷移] torch.tensor(...).to(device) -> jt.array(...)
        num_total_samples = reduce_mean(jt.array(num_total_pos)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_qfl, losses_bbox, losses_dfl, avg_factor = multi_apply(
            self.loss_single, grid_cells_list, cls_preds_list, reg_preds_list,
            labels_list, label_weights_list, bbox_targets_list, self.strides,
            num_total_samples=num_total_samples)

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).item()
        if avg_factor <= 0:
            # [遷移] torch.tensor(...) -> jt.float32(0.0)
            loss_qfl = jt.float32(0.0)
            loss_bbox = jt.float32(0.0)
            loss_dfl = jt.float32(0.0)
        else:
            losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
            losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
            loss_qfl = sum(losses_qfl)
            loss_bbox = sum(losses_bbox)
            loss_dfl = sum(losses_dfl)

        loss = loss_qfl + loss_bbox + loss_dfl
        return loss, dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

    def loss_single(self, grid_cells, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, stride, num_total_samples):
        grid_cells = grid_cells.reshape(-1, 4)
        cls_score = cls_score.reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        bg_class_ind = self.num_classes
        # [遷移] torch.nonzero(...) -> jt.nonzero(...)[0]
        pos_inds = jt.nonzero((labels >= 0) & (labels < bg_class_ind))[0]
        # [遷移] .new_zeros -> jt.zeros
        score = jt.zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_grid_cells = grid_cells[pos_inds]
            pos_grid_cell_centers = self.grid_cells_to_center(pos_grid_cells) / stride
            weight_targets = cls_score.detach().sigmoid().max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.distribution_project(pos_bbox_pred)
            pos_decode_bbox_pred = distance2bbox(pos_grid_cell_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = bbox2distance(
                pos_grid_cell_centers, pos_decode_bbox_targets, self.reg_max).reshape(-1)
            loss_bbox = self.loss_bbox(pos_decode_bbox_pred, pos_decode_bbox_targets,
                                       weight=weight_targets, avg_factor=1.0)
            loss_dfl = self.loss_dfl(pred_corners, target_corners,
                                     weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                                     avg_factor=4.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            # [遷移] torch.tensor(0).to(...) -> jt.array(0)
            weight_targets = jt.array(0)

        loss_qfl = self.loss_qfl(cls_score, (labels, score),
                                 weight=label_weights, avg_factor=num_total_samples)
        return loss_qfl, loss_bbox, loss_dfl, weight_targets.sum()

    def target_assign(self, cls_preds, reg_preds, featmap_sizes, gt_bboxes_list,
                      gt_bboxes_ignore_list, gt_labels_list):
        batch_size = cls_preds.shape[0]
        # [遷移] 移除 device 參數
        multi_level_grid_cells = [
            self.get_grid_cells(
                featmap_sizes[i], self.grid_cell_scale, stride, dtype='float32')
            for i, stride in enumerate(self.strides)
        ]
        mlvl_grid_cells_list = [multi_level_grid_cells for i in range(batch_size)]
        # [遷移] .size(0) -> .shape[0]
        num_level_cells = [grid_cells.shape[0] for grid_cells in mlvl_grid_cells_list[0]]
        num_level_cells_list = [num_level_cells] * batch_size
        for i in range(batch_size):
            mlvl_grid_cells_list[i] = jt.concat(mlvl_grid_cells_list[i])
        
        if gt_bboxes_ignore_list is None: gt_bboxes_ignore_list = [None] * batch_size
        if gt_labels_list is None: gt_labels_list = [None] * batch_size
        
        (all_grid_cells, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
            self.target_assign_single_img, mlvl_grid_cells_list,
            num_level_cells_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list)
        
        if any([labels is None for labels in all_labels]): return None
        
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        
        mlvl_cls_preds = images_to_levels([c for c in cls_preds], num_level_cells)
        mlvl_reg_preds = images_to_levels([r for r in reg_preds], num_level_cells)
        mlvl_grid_cells = images_to_levels(all_grid_cells, num_level_cells)
        mlvl_labels = images_to_levels(all_labels, num_level_cells)
        mlvl_label_weights = images_to_levels(all_label_weights, num_level_cells)
        mlvl_bbox_targets = images_to_levels(all_bbox_targets, num_level_cells)
        mlvl_bbox_weights = images_to_levels(all_bbox_weights, num_level_cells)
        
        return (mlvl_cls_preds, mlvl_reg_preds, mlvl_grid_cells, mlvl_labels,
                mlvl_label_weights, mlvl_bbox_targets, mlvl_bbox_weights,
                num_total_pos, num_total_neg)

    def target_assign_single_img(self, grid_cells, num_level_cells, gt_bboxes,
                                 gt_bboxes_ignore, gt_labels):
        # [遷移] .to(device) 移除, torch.from_numpy -> jt.array
        gt_bboxes = jt.array(gt_bboxes)
        gt_labels = jt.array(gt_labels)
        if gt_bboxes_ignore is not None:
            gt_bboxes_ignore = jt.array(gt_bboxes_ignore)

        assign_result = self.assigner.assign(
            grid_cells, num_level_cells, gt_bboxes, gt_bboxes_ignore, gt_labels)
        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes)

        num_cells = grid_cells.shape[0]
        bbox_targets = jt.zeros_like(grid_cells)
        bbox_weights = jt.zeros_like(grid_cells)
        # [遷移] .new_full -> jt.full
        labels = jt.full((num_cells,), self.num_classes, dtype='int64')
        # [遷移] .new_zeros -> jt.zeros
        label_weights = jt.zeros(num_cells, dtype='float32')

        if len(pos_inds) > 0:
            bbox_targets[pos_inds, :] = pos_gt_bboxes
            bbox_weights[pos_inds, :] = 1.0
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (grid_cells, labels, label_weights, bbox_targets,
                bbox_weights, pos_inds, neg_inds)

    def sample(self, assign_result, gt_bboxes):
        # [遷移] torch.nonzero(...).squeeze().unique() -> jt.nonzero(...)[0].unique()
        pos_inds = jt.nonzero(assign_result.gt_inds > 0)[0].unique()
        neg_inds = jt.nonzero(assign_result.gt_inds == 0)[0].unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            assert pos_assigned_gt_inds.numel() == 0
            # [遷移] torch.empty_like -> jt.empty_like
            pos_gt_bboxes = jt.empty_like(gt_bboxes).reshape(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.reshape(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def post_process(self, preds, meta):
        cls_scores, bbox_preds = preds.split([self.num_classes, 4 * (self.reg_max + 1)], dim=-1)
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
        det_results = {}
        warp_matrixes = meta["warp_matrix"]
        # [遷移] .cpu().numpy() -> .numpy()
        img_heights = meta["img_info"]["height"].numpy() if isinstance(meta["img_info"]["height"], jt.Var) else meta["img_info"]["height"]
        img_widths = meta["img_info"]["width"].numpy() if isinstance(meta["img_info"]["width"], jt.Var) else meta["img_info"]["width"]
        img_ids = meta["img_info"]["id"].numpy() if isinstance(meta["img_info"]["id"], jt.Var) else meta["img_info"]["id"]

        for result, img_width, img_height, img_id, warp_matrix in zip(
            result_list, img_widths, img_heights, img_ids, warp_matrixes
        ):
            det_bboxes, det_labels = result
            # [遷移] .detach().cpu().numpy() -> .numpy()
            det_bboxes = det_bboxes.numpy()
            det_bboxes[:, :4] = warp_boxes(
                det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height)
            classes = det_labels.numpy()
            det_result = {}
            for i in range(self.num_classes):
                inds = classes == i
                det_result[i] = np.concatenate([
                    det_bboxes[inds, :4].astype(np.float32),
                    det_bboxes[inds, 4:5].astype(np.float32),
                ], axis=1).tolist()
            det_results[img_id] = det_result
        return det_results

    def show_result(self, img, dets, class_names, score_thres=0.3, show=True, save_path=None):
        # 此方法與框架無關
        result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
        if show:
            cv2.imshow("det", result)
        return result

    def get_bboxes(self, cls_preds, reg_preds, img_metas):
        b = cls_preds.shape[0]
        input_height, input_width = img_metas["img"].shape[2:]
        input_shape = (input_height, input_width)
        featmap_sizes = [(math.ceil(input_height / s), math.ceil(input_width / s)) for s in self.strides]
        
        mlvl_center_priors = []
        for i, stride in enumerate(self.strides):
            # [遷移] 移除 device 參數
            y, x = self.get_single_level_center_point(featmap_sizes[i], stride, 'float32')
            # [遷移] .new_full -> jt.full
            strides_tensor = jt.full((x.shape[0],), stride)
            # [遷移] torch.stack -> jt.stack
            priors = jt.stack([x, y, strides_tensor, strides_tensor], dim=-1)
            mlvl_center_priors.append(priors.unsqueeze(0).repeat(b, 1, 1))

        center_priors = jt.concat(mlvl_center_priors, dim=1)
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_preds.sigmoid()
        
        result_list = []
        for i in range(b):
            score, bbox = scores[i], bboxes[i]
            # [遷移] .new_zeros -> jt.zeros
            padding = jt.zeros((score.shape[0], 1))
            score = jt.concat([score, padding], dim=1)
            results = multiclass_nms(
                bbox, score, score_thr=0.05,
                nms_cfg=dict(type="nms", iou_threshold=0.6), max_num=100)
            result_list.append(results)
        return result_list

    def get_single_level_center_point(self, featmap_size, stride, dtype, flatten=True):
        h, w = featmap_size
        # [遷移] torch.arange, 移除 device
        x_range = (jt.arange(w, dtype=dtype) + 0.5) * stride
        y_range = (jt.arange(h, dtype=dtype) + 0.5) * stride
        # [遷移] torch.meshgrid -> jt.meshgrid
        y, x = jt.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        return y, x

    def get_grid_cells(self, featmap_size, scale, stride, dtype):
        cell_size = stride * scale
        # [遷移] 移除 device 參數
        y, x = self.get_single_level_center_point(featmap_size, stride, dtype, flatten=True)
        grid_cells = jt.stack([
            x - 0.5 * cell_size, y - 0.5 * cell_size,
            x + 0.5 * cell_size, y + 0.5 * cell_size,
        ], dim=-1)
        return grid_cells

    def grid_cells_to_center(self, grid_cells):
        cells_cx = (grid_cells[:, 2] + grid_cells[:, 0]) / 2
        cells_cy = (grid_cells[:, 3] + grid_cells[:, 1]) / 2
        return jt.stack([cells_cx, cells_cy], dim=-1)

    # _forward_onnx 是 PyTorch 特有的方法，用於 ONNX 導出。
    # Jittor 有自己的模型導出機制，因此這部分邏輯無法直接遷移。
    # 在此處將其註釋掉。
    # def _forward_onnx(self, feats):
    #     ...
