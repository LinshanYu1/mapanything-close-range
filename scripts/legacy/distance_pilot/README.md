# 多视角测距试验

## 已完成与未完成
数据准备已在本机运行：scene_0017、RealSense，帧 0,24,...,216。选点 (610,198)、(800,330)；参考距离约 10.655 cm。阈值为 10 mm，实际通过数量见 manifest。之前软管点不稳定，换点仅依据标签质量，尚未查看任何模型输出。此试验存在选点筛选，不能代表全数据集成功率。

2026-10-16 已在本机 Apple GPU（MPS）完成 20 帧联合推理。使用项目 Python 3.12 环境 `.venv-baseline`。系统 Python 3.9 不满足依赖要求；沙箱内 MPS 检测可能为假阴性。

## 1. 数据准备
在项目根目录运行：

```sh
python3 scripts/distance_pilot/prepare.py --points 610 198 800 330
```

依赖 numpy、scipy、Pillow。脚本会从本机现有 /private/tmp/graspnet-scene-prefix.tar 补齐十帧，已提取的文件不需重复下载。如果临时归档被清理且缺文件，可使用 --archive 指定归档路径。

查看 output/distance_pilot/annotated.jpg 与 manifest.json。RGB、depth、label、meta 必须同帧对齐。

数学：P_cam = depth * inverse(K) * [u,v,1]；P_world = T_camera_to_world * P_cam；distance = norm(P1-P2)。本数据 factor_depth=1000。第一帧深度给出参考坐标；其他视角独立深度用于一致性检查。相同物体 ID 不能单独证明同一表面点可见。

逐视角排除：点在相机后方/画面外、深度为零、物体 ID 不同、深度与投影参考相差超过 10 mm。阈值只是初步质检规则，不是厘米级精度保证。尤其 10.7 cm 点对上的 1 cm 标签差异已接近 10% 成功阈值，精细结论需要更可靠标签。

## 2. RGB-only baseline
在有 GPU 的 Python 环境，按 https://github.com/facebookresearch/map-anything 的官方安装步骤安装包和 PyTorch，再运行：

```sh
python scripts/distance_pilot/baseline.py --device cuda --manifest output/distance_pilot_20/manifest.json
```

会下载 facebook/map-anything 权重，默认二十帧联合推理。只输入 RGB，不输入真实深度/相机内外参/参考距离，不做真实尺度拟合。使用标注辅助对应点评估，暂不评估自主点跟踪。

为避免缩放和裁剪导致测错像素，先明确把 RGB 缩放为 518x294，并同步变换点坐标，再要求官方 loader 保持这个尺寸；有轻微宽高比变化。输出坐标使用最近像素采样。后续正式实验可改为保存官方预处理的精确映射。

结果：baseline.csv、baseline_summary.json。先看有效标签数量与有效预测数量，再看绝对/相对误差。相对误差 <=10% 是试验阈值；预测失败算失败，不从分母删除；标签质检失败另列，不评分。十帧仍然只是一个独立场景。

## 3. 验证顺序
1. 人工查看红点是否落在同一物体的对应表面，特别检查轮廓和自遮挡。
2. 检查参考标签质量，不能把传感器数值称为尺子真值。
3. 再运行 baseline，完整保留所有失败及原因；不能根据模型误差换点或放宽 QC。
4. 本次仅验证流程；正式训练需另设按场景划分的数据集，并固定模型权重 revision。


## 本机已完成的 20 帧 baseline

- 输出目录：`output/distance_pilot_20/`。
- 参考距离：10.654866 cm，来自首帧传感器；按固定物理点对评分。
- 标签 QC 通过 10 帧，其中模型掩码允许的预测 7 帧，3 帧预测无效。
- 7 帧有效距离为 49.58–60.90 cm；有效预测 MAE 43.43 cm；10% 相对误差阈值成功 0/10。
- 不合格标签不参与评分；未使用参考距离校准预测尺度。
- 已验证官方 loader 输出与输入图片一致（反归一化最大像素差约 0.000021/255），没有额外裁剪。官方 pts3d 已包含 metric scaling，不再乘尺度。
- 仅一个场景、一个点对；误差原因尚未确定，不可据此判断模型总体水平或直接宣称改进空间。
- `baseline.csv` 为逐帧结果，`baseline_summary.json` 为汇总，`baseline_comparison_1.jpg`/`baseline_comparison_2.jpg` 为图片对照，`baseline_run.log` 为日志。
- 权重 revision 与代码 commit 见 `baseline_provenance.json`，依赖见 `baseline_environment.txt`。

复现本机运行（首次可能联网下载；终端可直接使用 MPS）：

```sh
HF_HOME="$PWD/.cache/huggingface" TORCH_HOME="$PWD/.cache/torch" MPLCONFIGDIR="$PWD/.cache/matplotlib" PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-baseline/bin/python -u scripts/distance_pilot/baseline.py --device mps --manifest output/distance_pilot_20/manifest.json
.venv-baseline/bin/python scripts/distance_pilot/report.py
```
