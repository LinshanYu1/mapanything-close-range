"""训练入口：固定数据划分，支持新实验、模块热启动和完整断点恢复。"""

import argparse
import json
import math
import sys
from pathlib import Path

# 支持 PyCharm 直接运行本文件，无需先设置 PYTHONPATH。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from close_range.common import ROOT, REVISION, sha, portable_path
from close_range.data_split import load_split
# 直接点击 Run 时使用下面的数组；按注释顺序填写。
# TRAIN_ARGS = [
#     "A",                              # 0: A / B / AB
#     "configs/graspnet_split.json",      # 1: 固定的数据划分文件
#     "checkpoints/graspnet_A",           # 2: 本次训练输出目录（新阶段换目录）
#     "mps",                            # 3: mps / cuda / cpu
#     50,                               # 4: 训练轮数
#     5e-4,                             # 5: 学习率
#     None,                             # 6: 一致性权重；None 表示 A=0、B/AB=0.1
#     443,                                # 7: 随机种子
#     None,                             # 8: 历史模型目录；None 表示初始化新的 A/B
#     "model_best.pt",                  # 9: 历史权重文件，也可用 model.pt
# ]
# TRAIN_ARGS = [
#     "B",                          # 改为 B
#     "configs/graspnet_split.json", # 使用相同的数据划分
#     "checkpoints/graspnet_B_run2", # 新训练必须使用空目录，不覆盖共享权重
#     "mps",
#     50,                            # 先跑 2 轮检查，再做正式训练
#     5e-4,
#     0.1,                          # 跨视角一致性损失权重
#     443,
#     None,                         # 首次训练 B，不加载 A 参数
#     "model_best.pt",
# ]

# TRAIN_ARGS = [
#     "A",                          # 改为 B
#     "configs/phone_split.json", # 使用相同的数据划分
#     "checkpoints/phone_A", # 新训练必须使用空目录，不覆盖共享权重
#     "mps",
#     30,                            # 先跑 2 轮检查，再做正式训练
#     5e-4,
#     0.1,                          # 跨视角一致性损失权重
#     443,
#     "checkpoints/graspnet_A",       # 把之前graspnet_A 训练的参数当成其实参数
#     "model_best.pt",
# ]

TRAIN_ARGS = [
    "AB",                             # 同时训练 A 和 B
    "configs/mixed_split.json",        # GraspNet＋Phone 混合划分
    "checkpoints/mixed_AB_run1",       # 新的输出目录，不能已有训练结果
    "mps",                            # Mac GPU
    30,
    5e-4,                             # 学习率
    0.1,                              # 跨视角一致性损失权重
    443,                              # 随机种子
    None,                             # 初始化新的 A、B
    "model_best.pt",                   # None 时这一项不用于加载
]

def ordered_args(values):
    """按固定顺序读取数组，复用命令行参数校验。"""
    if len(values) != 10:
        raise ValueError("TRAIN_ARGS 必须按顺序包含 10 项，见数组注释")
    names = ("variant", "split", "output-dir", "device", "epochs",
             "learning-rate", "consistency-weight", "seed", "init-from", "weights-name")
    argv = []
    for name, value in zip(names, values):
        if value is not None:
            argv.extend([f"--{name}", str(value)])
    return argv


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("A", "B", "AB"))
    parser.add_argument("--split", required=True, help="JSON，包含 train/val，可选 test；路径相对于项目根目录")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--consistency-weight", type=float, default=None,
                        help="默认 A=0，B/AB=0.1；不改变视角采样方式")
    parser.add_argument("--seed", type=int, default=0)
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--init-from", help="已有模型目录；只加载模块参数，新建优化器")
    start.add_argument("--resume", help="完整 checkpoint 文件，恢复优化器与进度")
    parser.add_argument("--weights-name", choices=("model.pt", "model_best.pt"),
                        default="model_best.pt", help="init-from 使用的权重文件")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("epochs 必须大于 0")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate 必须为有限正数")
    if args.consistency_weight is None:
        args.consistency_weight = 0.0 if args.variant == "A" else 0.1
    if not math.isfinite(args.consistency_weight) or args.consistency_weight < 0:
        parser.error("consistency-weight 必须为有限非负数")
    return args



def main(argv=None):
    args = parse_args(ordered_args(TRAIN_ARGS) if argv is None else argv)
    split = load_split(args.split)
    output = project_path(args.output_dir)
    resume = project_path(args.resume) if args.resume else None
    if output.exists() and (not output.is_dir() or (not resume and any(output.iterdir()))):
        raise ValueError("新实验必须使用空的输出目录；不会覆盖已有实验")
    if resume and not resume.is_file():
        raise FileNotFoundError(resume)

    revision = REVISION
    initialization = None
    if args.init_from:
        source = project_path(args.init_from)
        config = json.loads((source / "model_config.json").read_text())
        if config["variant"] != args.variant:
            raise ValueError("init-from 的模块类型必须与 variant 相同")
        revision = config["revision"]
        weights = source / args.weights_name
        initialization = {"directory": portable_path(source), "weights_name": args.weights_name,
                          "sha256": sha(weights)}
    elif resume:
        # 恢复时沿用断点记录的基础模型版本与初始化来源。
        import torch
        saved_config = torch.load(resume, map_location="cpu", weights_only=True)["config"]
        revision = saved_config["pretrained_revision"]
        initialization = saved_config.get("initialization")


    from close_range.training.dataset import SceneDistanceDataset
    from close_range.training.trainer import Trainer
    from close_range.models.model_io import load_module_parameters

    train = SceneDistanceDataset(ROOT, split["train"], training=True,
                                 seed=args.seed, use_consistency=True)
    val = SceneDistanceDataset(ROOT, split["val"], training=False,
                               seed=args.seed, use_consistency=True)
    print(f"variant={args.variant}; train={len(train)}; val={len(val)}; "
          f"test={len(split['test'])}（不参与训练）; output={output}", flush=True)
    trainer = Trainer.from_pretrained(
        output_dir=output, variant=args.variant, device=args.device,
        learning_rate=args.learning_rate, consistency_weight=args.consistency_weight,
        seed=args.seed, pretrained_revision=revision,
    )
    if args.init_from:
        load_module_parameters(trainer.model, source, args.weights_name)
    if initialization is not None:
        trainer.config["initialization"] = initialization
    trainer.fit(train, val, epochs=args.epochs, resume=resume)
    print(f"训练完成：{output}", flush=True)
    for name in ("model.pt", "model_best.pt"):
        if (output / name).is_file():
            print(f"模型参数：{output / name}")


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else None)
