import argparse

import jittor as jt

from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config, get_model_complexity_info


def main(config, input_shape=(320, 320)):
    model = build_model(config.model)
    # 使用 Jittor 版本的 FLOPs 计算
    try:
        flops, params = get_model_complexity_info(
            model, input_shape, print_per_layer_stat=True
        )
        print(f"FLOPs: {flops}")
        print(f"Params: {params}")
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
        return


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert .pth model to onnx.",
    )
    parser.add_argument("cfg", type=str, help="Path to .yml config file.")
    parser.add_argument(
        "--input_shape", type=str, default=None, help="Model intput shape."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg_path = args.cfg
    load_config(cfg, cfg_path)

    input_shape = args.input_shape
    if input_shape is None:
        input_shape = cfg.data.train.input_size
    else:
        input_shape = tuple(map(int, input_shape.split(",")))
        assert len(input_shape) == 2
    main(config=cfg, input_shape=input_shape)
