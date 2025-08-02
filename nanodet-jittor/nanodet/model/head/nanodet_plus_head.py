import math

import cv2
import numpy as np
import jittor as jt
import jittor.nn as nn

from nanodet.util import bbox2distance, distance2bbox, multi_apply, overlay_bbox_cv

from ...data.transform.warp import warp_boxes
from ..loss.gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from ..loss.iou_loss import GIoULoss
from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from ..module.nms import multiclass_nms
from .assigner.dsl_assigner import DynamicSoftLabelAssigner
from .gfl_head import Integral, reduce_mean


class NanoDetPlusHead(nn.Module):
    """NanoDet-Plus Head (Jittor Version)"""
    def __init__(
        self, num_classes, loss, input_channel, feat_channels=96,
        stacked_convs=2, kernel_size=5, strides=[8, 16, 32],
        conv_type="DWConv", norm_cfg=dict(type="BN"), reg_max=7,
        activation="LeakyReLU", assigner_cfg=dict(topk=13), **kwargs
    ):
        super(NanoDetPlusHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_size = kernel_size
        self.strides = strides
        self.reg_max = reg_max
        self.activation = activation
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule
        self.loss_cfg = loss
        self.norm_cfg = norm_cfg

        self.assigner = DynamicSoftLabelAssigner(**assigner_cfg)
        self.distribution_project = Integral(self.reg_max)

        self.loss_qfl = QualityFocalLoss(
            beta=self.loss_cfg.loss_qfl.beta,
            loss_weight=self.loss_cfg.loss_qfl.loss_weight)
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=self.loss_cfg.loss_dfl.loss_weight)
        self.loss_bbox = GIoULoss(loss_weight=self.loss_cfg.loss_bbox.loss_weight)
        
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        for _ in self.strides:
            cls_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)

        self.gfl_cls = nn.ModuleList([
            # [遷移] nn.Conv2d -> nn.Conv
            nn.Conv(
                self.feat_channels,
                self.num_classes + 4 * (self.reg_max + 1),
                1, padding=0)
            for _ in self.strides
        ])

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn, self.feat_channels, self.kernel_size,
                    stride=1, padding=self.kernel_size // 2,
                    norm_cfg=self.norm_cfg, bias=self.norm_cfg is None,
                    activation=self.activation))
        return cls_convs

    def init_weights(self):
        for m in self.cls_convs.modules():
            # [遷移] nn.Conv2d -> nn.Conv
            if isinstance(m, nn.Conv):
                normal_init(m, std=0.01)
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
        print("Finish initialize NanoDet-Plus Head.")

    # [遷移] forward -> execute
    def execute(self, feats):
        # [遷移] 移除 ONNX 相關邏輯
        outputs = []
        for feat, cls_convs, gfl_cls in zip(feats, self.cls_convs, self.gfl_cls):
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            outputs.append(output.flatten(start_dim=2))
        # [遷移] torch.cat -> jt.concat
        return jt.concat(outputs, dim=2).permute(0, 2, 1)

    def loss(self, preds, gt_meta, aux_preds=None):
        # [遷移] 移除 device
        batch_size = preds.shape[0]
        gt_bboxes = gt_meta["gt_bboxes"]
        gt_labels = gt_meta["gt_labels"]
        gt_bboxes_ignore = gt_meta.get("gt_bboxes_ignore", None)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None] * batch_size

        input_height, input_width = gt_meta["img"].shape[2:]
        featmap_sizes = [(math.ceil(input_height/s), math.ceil(input_width/s)) for s in self.strides]
        
        mlvl_center_priors = [
            self.get_single_level_center_priors(batch_size, featmap_sizes[i], s, dtype='float32')
            for i, s in enumerate(self.strides)
        ]
        center_priors = jt.concat(mlvl_center_priors, dim=1)

        cls_preds, reg_preds = preds.split([self.num_classes, 4 * (self.reg_max + 1)], dim=-1)
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)

        if aux_preds is not None:
            aux_cls_preds, aux_reg_preds = aux_preds.split([self.num_classes, 4 * (self.reg_max + 1)], dim=-1)
            aux_dis_preds = self.distribution_project(aux_reg_preds) * center_priors[..., 2, None]
            aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds)
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                aux_cls_preds.detach(), center_priors, aux_decoded_bboxes.detach(),
                gt_bboxes, gt_labels, gt_bboxes_ignore)
        else:
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                cls_preds.detach(), center_priors, decoded_bboxes.detach(),
                gt_bboxes, gt_labels, gt_bboxes_ignore)

        loss, loss_states = self._get_loss_from_assign(cls_preds, reg_preds, decoded_bboxes, batch_assign_res)

        if aux_preds is not None:
            aux_loss, aux_loss_states = self._get_loss_from_assign(
                aux_cls_preds, aux_reg_preds, aux_decoded_bboxes, batch_assign_res)
            loss += aux_loss
            for k, v in aux_loss_states.items():
                loss_states["aux_" + k] = v
        return loss, loss_states

    def _get_loss_from_assign(self, cls_preds, reg_preds, decoded_bboxes, assign):
        (labels, label_scores, label_weights, bbox_targets, dist_targets, num_pos) = assign
        
        # [遷移] torch.tensor(...).to(device) -> jt.array(...)
        num_total_samples = max(reduce_mean(jt.array(sum(num_pos))).item(), 1.0)

        labels = jt.concat(labels, dim=0)
        label_scores = jt.concat(label_scores, dim=0)
        label_weights = jt.concat(label_weights, dim=0)
        bbox_targets = jt.concat(bbox_targets, dim=0)
        cls_preds = cls_preds.reshape(-1, self.num_classes)
        reg_preds = reg_preds.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        
        loss_qfl = self.loss_qfl(
            cls_preds, (labels, label_scores),
            weight=label_weights, avg_factor=num_total_samples)

        # [遷移] torch.nonzero(...) -> jt.nonzero(...)[0]
        pos_inds = jt.nonzero((labels >= 0) & (labels < self.num_classes))[0]

        if len(pos_inds) > 0:
            weight_targets = cls_preds[pos_inds].detach().sigmoid().max(dim=1)[0]
            bbox_avg_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)
            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds], bbox_targets[pos_inds],
                weight=weight_targets, avg_factor=bbox_avg_factor)
            dist_targets = jt.concat(dist_targets, dim=0)
            loss_dfl = self.loss_dfl(
                reg_preds[pos_inds].reshape(-1, self.reg_max + 1),
                dist_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * bbox_avg_factor)
        else:
            loss_bbox = reg_preds.sum() * 0
            loss_dfl = reg_preds.sum() * 0

        loss = loss_qfl + loss_bbox + loss_dfl
        return loss, dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

    # [遷移] @torch.no_grad() -> @jt.no_grad()
    @jt.no_grad()
    def target_assign_single_img(self, cls_preds, center_priors, decoded_bboxes,
                                 gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        # [遷移] torch.from_numpy(...).to(device) -> jt.array(...)
        gt_bboxes = jt.array(gt_bboxes).cast(decoded_bboxes.dtype)
        gt_labels = jt.array(gt_labels)
        if gt_bboxes_ignore is not None:
            gt_bboxes_ignore = jt.array(gt_bboxes_ignore).cast(decoded_bboxes.dtype)

        assign_result = self.assigner.assign(
            cls_preds, center_priors, decoded_bboxes,
            gt_bboxes, gt_labels, gt_bboxes_ignore)
        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes)

        num_priors = center_priors.shape[0]
        bbox_targets = jt.zeros_like(center_priors)
        dist_targets = jt.zeros_like(center_priors)
        # [遷移] .new_full -> jt.full, .new_zeros -> jt.zeros
        labels = jt.full((num_priors,), self.num_classes, dtype='int64')
        label_weights = jt.zeros(num_priors, dtype='float32')
        label_scores = jt.zeros(labels.shape, dtype='float32')

        num_pos_per_img = pos_inds.shape[0]
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            bbox_targets[pos_inds, :] = pos_gt_bboxes
            dist_targets[pos_inds, :] = (
                bbox2distance(center_priors[pos_inds, :2], pos_gt_bboxes) /
                center_priors[pos_inds, None, 2])
            dist_targets = dist_targets.clamp(min_v=0, max_v=self.reg_max - 0.1)
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        return (labels, label_scores, label_weights, bbox_targets,
                dist_targets, num_pos_per_img)

    def sample(self, assign_result, gt_bboxes):
        pos_inds = jt.nonzero(assign_result.gt_inds > 0)[0].unique()
        neg_inds = jt.nonzero(assign_result.gt_inds == 0)[0].unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            assert pos_assigned_gt_inds.numel() == 0
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
        img_heights = meta["img_info"]["height"].numpy() if isinstance(meta["img_info"]["height"], jt.Var) else meta["img_info"]["height"]
        img_widths = meta["img_info"]["width"].numpy() if isinstance(meta["img_info"]["width"], jt.Var) else meta["img_info"]["width"]
        img_ids = meta["img_info"]["id"].numpy() if isinstance(meta["img_info"]["id"], jt.Var) else meta["img_info"]["id"]

        for result, img_width, img_height, img_id, warp_matrix in zip(
            result_list, img_widths, img_heights, img_ids, warp_matrixes
        ):
            det_bboxes, det_labels = result
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
        result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
        if show:
            cv2.imshow("det", result)
        return result

    def get_bboxes(self, cls_preds, reg_preds, img_metas):
        b = cls_preds.shape[0]
        input_height, input_width = img_metas["img"].shape[2:]
        input_shape = (input_height, input_width)
        featmap_sizes = [(math.ceil(input_height / s), math.ceil(input_width / s)) for s in self.strides]
        
        mlvl_center_priors = [
            self.get_single_level_center_priors(b, featmap_sizes[i], s, 'float32')
            for i, s in enumerate(self.strides)
        ]
        center_priors = jt.concat(mlvl_center_priors, dim=1)
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_preds.sigmoid()
        
        result_list = []
        for i in range(b):
            score, bbox = scores[i], bboxes[i]
            padding = jt.zeros((score.shape[0], 1))
            score = jt.concat([score, padding], dim=1)
            results = multiclass_nms(
                bbox, score, score_thr=0.05,
                nms_cfg=dict(type="nms", iou_threshold=0.6), max_num=100)
            result_list.append(results)
        return result_list

    def get_single_level_center_priors(self, batch_size, featmap_size, stride, dtype):
        h, w = featmap_size
        x_range = jt.arange(w, dtype=dtype) * stride
        y_range = jt.arange(h, dtype=dtype) * stride
        y, x = jt.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        strides_tensor = jt.full((x.shape[0],), stride)
        priors = jt.stack([x, y, strides_tensor, strides_tensor], dim=-1)
        return priors.unsqueeze(0).repeat(batch_size, 1, 1)

    # _forward_onnx is PyTorch-specific and removed.