import jittor as jt
from jittor import init
from jittor import nn
import contextlib
import copy
import io
import itertools
import json
import logging
import os
import warnings
import numpy as np
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
logger = logging.getLogger('NanoDet')

def xyxy2xywh(bbox):
    return [bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

class CocoDetectionEvaluator():

    def __init__(self, dataset):
        assert hasattr(dataset, 'coco_api')
        self.class_names = dataset.class_names
        self.coco_api = dataset.coco_api
        self.cat_ids = dataset.cat_ids
        self.metric_names = ['mAP', 'AP_50', 'AP_75', 'AP_small', 'AP_m', 'AP_l']

    def results2json(self, results):
        json_results = []
        for (image_id, dets) in results.items():
            for (label, bboxes) in dets.items():
                category_id = self.cat_ids[label]
                for bbox in bboxes:
                    score = float(bbox[4])
                    detection = dict(image_id=int(image_id), category_id=int(category_id), bbox=xyxy2xywh(bbox), score=score)
                    json_results.append(detection)
        return json_results

    def evaluate(self, results, save_dir, rank=(- 1)):
        results_json = self.results2json(results)
        if (len(results_json) == 0):
            warnings.warn('Detection result is empty! Please check whether training set is too small (need to increase val_interval in config and train more epochs). Or check annotation correctness.')
            empty_eval_results = {}
            for key in self.metric_names:
                empty_eval_results[key] = 0
            return empty_eval_results
        json_path = os.path.join(save_dir, 'results{}.json'.format(rank))
        json.dump(results_json, open(json_path, 'w'))
        coco_dets = self.coco_api.loadRes(json_path)
        coco_eval = COCOeval(copy.deepcopy(self.coco_api), copy.deepcopy(coco_dets), 'bbox')
        # Restrict evaluation to the subset of images we actually predicted on (mini-eval case)
        try:
            subset_img_ids = sorted([int(k) for k in results.keys()])
            if len(subset_img_ids) > 0:
                coco_eval.params.imgIds = subset_img_ids
                logger.info(f"Evaluating on subset of {len(subset_img_ids)} images (mini-eval mode)")
        except Exception:
            pass
        coco_eval.evaluate()
        coco_eval.accumulate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()
        logger.info("\n" + redirect_string.getvalue())
        headers = ['class', 'AP50', 'mAP']
        colums = 6
        per_class_ap50s = []
        per_class_maps = []
        precisions = coco_eval.eval['precision']
        assert (len(self.class_names) == precisions.shape[2])
        for (idx, name) in enumerate(self.class_names):
            precision_50 = precisions[0, :, idx, 0, (- 1)]
            precision_50 = precision_50[precision_50 > (- 1)]
            ap50 = (np.mean(precision_50) if precision_50.size else float('nan'))
            per_class_ap50s.append(float((ap50 * 100)))
            precision = precisions[:, :, idx, 0, (- 1)]
            precision = precision[(precision > (- 1))]
            ap = (np.mean(precision) if precision.size else float('nan'))
            per_class_maps.append(float((ap * 100)))
        num_cols = min(colums, (len(self.class_names) * len(headers)))
        flatten_results = []
        for (name, ap50, mAP) in zip(self.class_names, per_class_ap50s, per_class_maps):
            flatten_results += [name, ap50, mAP]
        row_pair = itertools.zip_longest(*[flatten_results[i::num_cols] for i in range(num_cols)])
        table_headers = (headers * (num_cols // len(headers)))
        table = tabulate(row_pair, tablefmt='pipe', floatfmt='.1f', headers=table_headers, numalign='left')
        logger.info("\n" + table)
        aps = coco_eval.stats[:6]
        eval_results = {}
        for (k, v) in zip(self.metric_names, aps):
            eval_results[k] = v
        return eval_results