"""一次性生成 GraspNet 场景划分；不覆盖已有文件、不在每次训练时重分。"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from close_range.common import ROOT, portable_path
from close_range.training.main import load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--output", default="configs/graspnet_split.json")
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(f"划分已存在，不覆盖：{output}")
    groups, skipped = {}, []
    for path in sorted((ROOT / "data/manifests/graspnet").glob("*.json")):
        manifest = json.loads(path.read_text())
        original = manifest.get("original_scene", "")
        if manifest.get("source") != "graspnet" or not re.fullmatch(r"scene_\d{4}", original):
            skipped.append(portable_path(path))
            continue
        groups.setdefault(int(original[6:]), []).append(portable_path(path))
    if set(groups) != set(range(190)):
        raise ValueError("本划分要求完整的 scene_0000 至 scene_0189 预处理清单")
    # 已用于方法开发的场景固定留在验证集，不声称它们是独立测试数据。
    validation = {0, 17}
    candidates = sorted(set(range(100)) - validation)
    random.Random(args.seed).shuffle(candidates)
    validation.update(candidates[:18])
    ids = {"train": sorted(set(range(100)) - validation), "val": sorted(validation),
           "test": list(range(100, 190))}
    split = {name: [p for scene in scenes for p in groups[scene]] for name, scenes in ids.items()}
    split["metadata"] = {"seed": args.seed, "policy": "official train 0-99 split 80/20; official test 100-189 held out",
                         "development_scenes": [0, 17], "excluded_manifests": skipped}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate before publishing the final file.
    temp = output.with_suffix(".json.tmp")
    try:
        temp.write_text(json.dumps(split, indent=2) + "\n")
        load_split(temp)
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    print({name: len(paths) for name, paths in split.items() if name != "metadata"})
    print(portable_path(output))


if __name__ == "__main__":
    main()
