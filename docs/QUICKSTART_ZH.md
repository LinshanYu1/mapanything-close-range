# 手机照片测距试跑

## 统一安装依赖（Mac）

项目使用 pip 管理依赖，版本记录在 `requirements-macos-py312.txt`。
该文件来自已通过测试的 Python 3.12 环境快照，不是所有平台通用的锁文件。
本地修改后的模型代码从 `third_party` 加载，不使用旧项目路径。

```bash
# 在项目根目录运行；需要已有 Python 3.12
bash scripts/tools/setup_env.sh
source .venv/bin/activate
python scripts/tools/check_environment.py
```

没有 `python3.12` 命令时，可通过 `PYTHON312_BIN` 指定已有 Python 3.12 解释器。
脚本只创建/更新项目 `.venv`，不改系统 Python。已有非 3.12 的 `.venv` 会报错，不会删除它。
PyCharm 的项目解释器选择本项目 `.venv/bin/python`；安装与运行必须使用同一个环境。
依赖安装包含 OpenCV（cv2）、jaxtyping、PyTorch 和上游依赖，不需要逐个补包。
不要同时安装 `opencv-python` 和 `opencv-python-headless`；本项目使用后者。
`check_environment.py` 只检查依赖和本地代码来源，不证明数据划分或训练已经就绪。
Colab/CUDA 环境需单独确认 PyTorch 与 CUDA 版本，不直接套用这个 Mac 快照。


推荐批量入口：`bash scripts/local.sh scripts/predict_directory.py --input data/processed/phone`。自动按目录分组并按图片数量选择单图/多图，已有标签自动评分。结果位于 `outputs/batch_<时间戳>/summary.csv`。以下单场景命令保留作诊断。

当前仅实现 M0 试跑，没有 A/B 或训练代码。原始 main.py 保留不动。

距离统一在 `data/raw/phone/distances.conf` 的 `[distances]` 段配置，例如 `scene-001 = 17`，单位固定为 inch。场景名必须与目录名一致。缺失、零、负值或非有限数会报错。

## 本机环境

在新仓库根目录运行下面命令。`scripts/local.sh` 优先使用 PYTHON_BIN 指定的环境，其次项目 .venv，再其次当前已激活的虚拟环境，否则使用项目内 `.venv-baseline` ，缓存默认放在项目 .cache 目录。
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

## 4. 处理 GraspNet 数据

当前流程是下载原始场景、生成模型输入和距离标签，再批量检查与预测：

```bash
# 已下载并解压完成时跳过这一条
bash scripts/local.sh scripts/graspnet/download_mirror.py

# 默认扫描全部场景，选择三帧、生成十对距离标签
bash scripts/local.sh scripts/graspnet/prepare.py

# 检查输入，不加载模型权重
bash scripts/local.sh scripts/predict_directory.py --input data/processed/graspnet --check-only

# 实际预测，有标签时自动评估距离
bash scripts/local.sh scripts/predict_directory.py --input data/processed/graspnet
```

真实深度与相机内参用于生成距离标签；RGB-only 推理不将它们提供给模型。检查通过后还需抽查点位与标签，具体步骤见 [GraspNet 操作说明](../scripts/graspnet/README.md)。

手机与 GraspNet 统一为 schema_version=1 的场景清单、model-input 像素坐标标签和米单位距离；两者共用模型和评分脚本。当前每场景人工标注入口只生成一对点，评估器支持多对。

## 结果解释

有效完成推理与评分表示流程跑通，不表示模型测距准确。只有一个场景，不能据此得出总体性能结论。正式训练前再固定按场景划分的数据集。
