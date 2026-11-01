# 手机照片测距试跑

推荐批量入口：`bash scripts/local.sh scripts/predict_directory.py --input data/processed/phone`。自动按目录分组并按图片数量选择单图/多图，已有标签自动评分。结果位于 `outputs/batch_<时间戳>/summary.csv`。以下单场景命令保留作诊断。

当前仅实现 M0 试跑，没有 A/B 或训练代码。原始 main.py 保留不动。

距离统一在 `data/raw/phone/distances.conf` 的 `[distances]` 段配置，例如 `scene-001 = 17`，单位固定为 inch。场景名必须与目录名一致。缺失、零、负值或非有限数会报错。

## 本机环境

在新仓库根目录运行下面命令。`scripts/local.sh` 优先使用 PYTHON_BIN 指定的环境，其次项目 .venv，否则复用旧项目 `/Users/linshanyu/Desktop/dl-project/.venv-baseline` 和已有模型缓存。
不要用 IDE 默认的 Python 3.14 直接运行。此启动器的默认缓存路径是本机便捷配置；同学需安装上游依赖并设置 PYTHON_BIN、HF_HOME、TORCH_HOME，不能假设旧目录存在。

`third_party/map-anything` 是旧项目上游代码快照，保留许可证；commit 和来源见 `third_party/upstream.json`。模型固定使用原实验 revision。没有复制大型权重。

## 1. 数据准备

三张输入图直接位于 `data/raw/phone/scene-001/`，该目录只放输入照片。原尺子照片已移到 `outputs/measurement_archive/scene-001/` 保留，不作为模型输入。

```bash
bash scripts/local.sh scripts/phone/prepare.py
```

每个场景仅运行一次。EXIF 方向转正，再按上游默认比例映射和 crop/resize 生成无标记 PNG；不会强制拉伸到旧实验的 518×294。处理后选点，因此不需要猜测原图到模型的坐标映射。后续不可修改这些 PNG；程序用哈希检查。

## 2. 标注

```bash
bash scripts/local.sh scripts/phone/annotate.py
```

距离会自动从配置读取并显示，命令不再接受 --distance 或 --unit。窗口出现后，先点 Mac mini 上实际测量的端点 A，再点耳机上的实际端点 B。可先用工具栏放大，再退出缩放模式后选点。两个点选齐即保存。图内圆点不会写进模型输入图，只另外输出 `data/annotations/phone/scene-001.preview.png`。请先检查它。

43.18 cm 是用户报告的手工测量标签，不代表已验证到 0.01 cm 的精度。纸中心只有在它确实是测量端点时才能选。

重选需显式加 `--overwrite`。修改 distances.conf 后也需重新确认标注；评估会拒绝配置与标注不一致的结果。无图形界面可以加 `--points xA yA xB yB`，坐标必须是处理后图片上的像素坐标，不能猜测原图坐标。

## 3. 单图与三图推理

```bash
bash scripts/local.sh scripts/run_baseline.py --mode single --output outputs/phone_single_001
bash scripts/local.sh scripts/evaluate.py --run outputs/phone_single_001
bash scripts/local.sh scripts/run_baseline.py --mode multi --output outputs/phone_multi_001
bash scripts/local.sh scripts/evaluate.py --run outputs/phone_multi_001
```

默认使用 MPS。单图和多图使用同一张参考图和相同的点对。重复实验换一个 output 名称，已有结果不覆盖。
推理脚本不读取点标注和真实距离；只预测三维点图。评估脚本读取保存的预测和真实标签，计算距离，不进行真实尺度对齐。模型掩码无效、坐标越界或非有限数均作为无效预测，不用附近像素偷偷替换。

输出目录包含 `run.json`（版本、输入、时间）、`reference_prediction.npz`（参考图三维点与掩码）、`metrics.csv`、`metrics.json`。

## 4. 复用 GraspNet 旧试验

```bash
bash scripts/local.sh scripts/graspnet/import_legacy.py --legacy-manifest /Users/linshanyu/Desktop/dl-project/output/distance_pilot_20/manifest.json --legacy-root /Users/linshanyu/Desktop/dl-project
bash scripts/local.sh scripts/run_baseline.py --manifest data/manifests/graspnet/graspnet_pilot.json --mode single --output outputs/graspnet_single_001
bash scripts/local.sh scripts/evaluate.py --manifest data/manifests/graspnet/graspnet_pilot.json --run outputs/graspnet_single_001
```

这是已有传感器标签的兼容导入器，不是完整 GraspNet 下载/标注工具。保持旧预处理与 QC；只评估第一个合格参考视图上的原点对。`--mode multi` 将使用清单全部视图，20 张可能耗时较长。
旧 `prepare.py`、`baseline.py`、诊断和报告脚本存于 `scripts/legacy/distance_pilot/`，仅供迁移参考，包含旧项目路径假设，不作为新项目直接运行入口。

手机与 GraspNet 统一为 schema_version=1 的场景清单、model-input 像素坐标标签和米单位距离；两者共用模型和评分脚本。当前每场景人工标注入口只生成一对点，评估器支持多对。

## 结果解释

有效完成推理与评分表示流程跑通，不表示模型测距准确。只有一个场景，不能据此得出总体性能结论。正式训练前再固定按场景划分的数据集。
