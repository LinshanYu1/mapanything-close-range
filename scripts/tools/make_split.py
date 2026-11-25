"""多个 manifest 目录生成同一划分；可以修改数组后直接运行。"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from close_range.data_split import make_split

MANIFESTS_PATH = ["data/manifests/graspnet", "data/manifests/phone"]
OUTPUT_PATH = "configs/mixed_split.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", default=MANIFESTS_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=443)
    args = parser.parse_args()
    make_split(args.manifests, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
