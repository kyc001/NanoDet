import argparse
import datetime
import os
import sys

import jittor as jt

# 确保优先加载 nanodet-jittor 版本
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
from nanodet.util.logger import NanoDetLightningLogger
from nanodet.util import cfg, load_config, load_model_weight


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="val", help="task to run, test or val"
    )
    parser.add_argument("--config", type=str, help="model config file(.yml) path")
    parser.add_argument("--model", type=str, help="ckeckpoint file(.ckpt) path")
    args = parser.parse_args()
    return args


def main(args):
    load_config(cfg, args.config)
    local_rank = -1
    if jt.has_cuda:
        jt.flags.use_cuda = 1
    cfg.defrost()
    timestr = datetime.datetime.now().__format__("%Y%m%d%H%M%S")
    cfg.save_dir = os.path.join(cfg.save_dir, timestr)
    os.makedirs(cfg.save_dir, exist_ok=True)
    logger = NanoDetLightningLogger(cfg.save_dir)

    assert args.task in ["val", "test"]
    cfg.update({"test_mode": args.task})

    logger.info("Setting up data...")
    val_dataset = build_dataset(cfg.data.val, args.task)
    val_bs = getattr(cfg.device, "val_batchsize_per_gpu", 1)
    val_dataloader = val_dataset.set_attrs(
        batch_size=val_bs,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        collate_batch=naive_collate,
        drop_last=False,
    )
    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    logger.info("Creating model...")
    task = TrainingTask(cfg, evaluator)

    ckpt = jt.load(args.model)
    load_model_weight(task.model, ckpt, logger)
    task.model.eval()

    logger.info("Starting testing...")
    results = {}
    for i, batch in enumerate(val_dataloader):
        batch = task._preprocess_batch_input(batch)
        with jt.no_grad():
            dets = task.model.inference(batch)
        if isinstance(dets, dict):
            results.update(dets)
    eval_results = evaluator.evaluate(results, cfg.save_dir)
    logger.info(f"Eval Results: {eval_results}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
