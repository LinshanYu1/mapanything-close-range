"""GraspNet 单来源兼容入口，共用多来源划分实现。"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from close_range.data_split import make_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="configs/graspnet_split.json")
    parser.add_argument("--seed", type=int, default=443)
    args = parser.parse_args()
    make_split(["data/manifests/graspnet"], args.output, seed=args.seed)


if __name__ == "__main__":
    main()
