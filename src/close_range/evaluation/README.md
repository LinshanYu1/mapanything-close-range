# 评估

所有命令都在项目根目录运行。模型只接收 RGB；标注点只用来读取预测的三维点，真实距离只用来评分，不做真实尺度对齐。

| 模块 | 作用 |
|---|---|
| `run` | 一批场景 × 多个模型 × 单图/多图，逐点对打分 → `outputs/eval/<name>/results.csv` |
| `summarize` | 总体和按距离分段（<20 / 20–60 / ≥60 cm）的中位误差、10%/20% 以内比例、无效数 |
| `plot_phone` | 预测距离 vs 真实距离散点图（对数坐标） |
| `stability` | 跨视角子集稳定性：固定参考图，换辅助图组合，看预测距离波动多大 |

**数据来源**：默认从 `configs/mixed_split.json` 的 `test` 集取场景（目前是 GraspNet 100–189），
可用 `--split 文件 --set test` 换划分文件（例如 `configs/mixed_split_1.json`）。
train/val 在训练中已经影响过模型，**不能**用来报告结果。

若某个划分把一部分手机场景放进了 train/val，评估全部手机照片时加 `--exclude-seen 划分文件`，
自动去掉训练/验证用过的场景，剩下的都是模型没见过的：

```bash
bash scripts/local.sh -m close_range.evaluation.run --name phone_heldout --manifests 'data/manifests/phone/scene-*.json' \
    --exclude-seen configs/mixed_split_1.json --models baseline A B AB
```

## 1. 测距评估

```bash
bash scripts/local.sh -m close_range.evaluation.run --name test --models baseline A B AB
bash scripts/local.sh -m close_range.evaluation.run --name phone --manifests 'data/manifests/phone/scene-*.json'
bash scripts/local.sh -m close_range.evaluation.summarize outputs/eval/phone/results.csv --latex
bash scripts/local.sh -m close_range.evaluation.plot_phone
```

- 每个模型只加载一次；中断后重跑同一条命令，已完成的组合直接复用。
- **换一套权重时一定换 `--name`**，否则会复用旧结果。例如混合训练的权重：
  `--name phone_mixed --models baseline AB --dir-AB checkpoints/mixed_AB_run1`
- 无效预测（端点落在 mask 区域）计为失败，中位误差只统计有效预测。

## 2. 跨视角子集稳定性

与训练时的一致性采样相同：同一场景、同一对端点，参考图固定（标注在参考图上），从其余视图中取不同组合作为辅助图。
子集只由场景和随机种子决定，原模型、A、B、AB 用完全相同的组合，可以直接比较。

```bash
bash scripts/local.sh -m close_range.evaluation.stability --name test_stability --models baseline A B AB
bash scripts/local.sh -m close_range.evaluation.stability --name phone --manifests 'data/manifests/phone/scene-*.json'
bash scripts/local.sh -m close_range.evaluation.stability --name phone --summarize-only
```

- 默认子集为“参考图 + 1 张”“参考图 + 2 张”（`--sizes 2 3`，最多 6 个，固定种子抽样），再加上全部视图（`--no-full` 可去掉）。
  GraspNet 每个场景 3 张图，即 2 个双图子集 + 1 个全部视图。
- 指标（每个场景、模型、点对，至少 2 个有效子集才统计）：
  - `cv_pct`：预测距离标准差 / 平均值，越小越稳定
  - `range_gt_pct`：(最大 − 最小) / 真实距离
  - `mean_rel_err_pct`：各子集相对误差的平均
- 汇总：`stability_summary.csv`（按模型和距离段）；逐场景：`stability_scenes.csv`。
- B 的作用是给视角分配权重，所以这项评估主要比较 B/AB 和 M0：B 越稳定，说明融合越不受个别差视角影响。
