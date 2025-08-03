import argparse

import torch

from nanodet.util import convert_old_model


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert .pth model to onnx.",
    )
    parser.add_argument("--file_path", type=str, help="Path to .pth checkpoint.")
    parser.add_argument("--out_path", type=str, help="Path to .ckpt checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    file_path = args.file_path
    out_path = args.out_path
    old_check_point = torch.load(file_path)
    new_check_point = convert_old_model(old_check_point)
    torch.save(new_check_point, out_path)
    print("Checkpoint saved to:", out_path)
