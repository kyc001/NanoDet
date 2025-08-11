# -*- coding: utf-8 -*-
import os, argparse, json, sys
# Force import jittor version of 'nanodet' by putting nanodet-jittor on sys.path first
THIS_DIR = os.path.dirname(__file__)
JT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
sys.path.insert(0, JT_ROOT)

import cv2
import jittor as jt
from nanodet.util import cfg, load_config, mkdir
from nanodet.data.dataset import build_dataset
from nanodet.trainer.task import TrainingTask
from nanodet.evaluator import build_evaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', default='workspace/vis')
    ap.add_argument('--num', type=int, default=20)
    ap.add_argument('--score_thr', type=float, default=0.3)
    args = ap.parse_args()

    load_config(cfg, args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    # dataset & evaluator
    val_dataset = build_dataset(cfg.data.val, 'val')
    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    # model task
    task = TrainingTask(cfg, evaluator)
    ckpt = jt.load(args.ckpt)
    # 兼容 .ckpt 或 .pth
    state_dict = ckpt.get('state_dict', ckpt)
    try:
        task.model.load_state_dict(state_dict)
    except Exception:
        # 兼容某些权重前缀，例如 'model.'
        fixed = {}
        for k,v in state_dict.items():
            nk = k
            if nk.startswith('model.'):
                nk = nk[len('model.') :]
            fixed[nk] = v
        task.model.load_state_dict(fixed)
    task.model.eval()

    # 遍历样本，做推理与保存可视化
    saved = 0
    for idx in range(len(val_dataset)):
        meta = val_dataset[idx]
        # 保留原始文件名
        orig_file_name = meta['img_info'].get('file_name', f"img_{meta['img_info'].get('id', idx)}.jpg")
        # 单图推理时补齐 batch 维度
        if isinstance(meta['img'], jt.Var) and len(meta['img'].shape) == 3:
            meta['img'] = meta['img'].unsqueeze(0)
        # 适配 post_process 期望的批量 meta 结构
        h = meta['img_info']['height']
        w = meta['img_info']['width']
        img_id = meta['img_info']['id']
        meta['img_info'] = {
            'height': [h],
            'width': [w],
            'id': [img_id],
        }
        with jt.no_grad():
            dets = task.model.inference(meta)
        # 获取原图（优先磁盘读取）
        img_path = os.path.join(cfg.data.val.img_path, orig_file_name)
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            # 退化到使用处理后的图像（B,C,H,W -> H,W,C）
            im = meta['img'][0]
            if isinstance(im, jt.Var): im = im.numpy()
            if im.shape[0] in (1,3):
                im = im.transpose(1,2,0)
            raw_img = (im * 255).astype('uint8') if im.max()<=1.0 else im.astype('uint8')
        # 保存路径
        save_path = os.path.join(args.out_dir, f"{os.path.splitext(os.path.basename(orig_file_name))[0]}_det.jpg")
        # 仅取当前图像的检测结果
        dets_for_img = dets[int(img_id)] if isinstance(dets, dict) and int(img_id) in dets else dets
        task.model.head.show_result(raw_img, dets_for_img, cfg.class_names, score_thres=args.score_thr, show=False, save_path=save_path)
        saved += 1
        if saved >= args.num:
            break
    print(f"Saved {saved} visualizations to {args.out_dir}")


if __name__ == '__main__':
    main()

