# 训练代码说明

本目录负责逐场景的 A / B / A+B 训练。M1 暂未实现：需要先明确原模型解冻范围和验证内存预算。

- `dataset.py`：读取预处理图片、参考图坐标和以米为单位的标签。单张/双张返回一组；启用一致性且至少三张时返回两组双视角输入。关闭一致性时输入该场景全部照片，请注意视角数量和内存。
- `trainer.py`：加载并冻结预训练模型、安装模块、更新参数、逐场景验证。
- `losses.py`：单组对数距离平方监督；双组监督取平均，再加一致性损失。
- `checkpoint.py`：保存模块参数、优化器、数据/配置签名和训练进度，在完整 epoch 边界恢复。

## 调用示例

在项目环境运行，入口须能导入 `src`（现有 `scripts/local.sh` 已设置）。先固定互不重叠的训练、验证 manifest 清单。

```python
from close_range.training.trainer import Trainer
from close_range.training.dataset import SceneDistanceDataset

from close_range.common import ROOT
root = ROOT
train = SceneDistanceDataset(root, train_manifest_paths, training=True, seed=0)
val = SceneDistanceDataset(root, val_manifest_paths, training=False, seed=0)
trainer = Trainer.from_pretrained(
    output_dir="checkpoints/graspnet_ab",
    variant="AB",             # A / B / AB
    device="mps",             # Mac: mps；Colab: cuda
    learning_rate=5e-4,
    consistency_weight=0.1,
    seed=0,
)
trainer.fit(train, val, epochs=10)
```

输出路径相对于启动程序的工作目录；建议在项目根目录启动或传绝对路径。每次新训练使用空目录。原模型保持 eval 模式以禁用 dropout，但训练 forward 开启梯度；RGB-only 模式不提供深度或相机真值。模型输入不包含距离标签。

B-only 会跳过单张场景并记录数量；A+B 的单张场景仍能训练 A。若只用单张照片，则 B 学不到视角偏置。数据错误和训练非有限值直接报错。

## 输出与恢复

- `metrics/epoch_0001.json`：训练和验证明细，场景平均误差，以及按 Phone / GraspNet 分组的验证结果。
- `training_config.json`：独立的可读训练配置，包含学习率、目标轮数、模块类型、数据清单、采样设置及预训练版本。
- `model.pt`：最近完成一轮的 A/B 模块参数，不含优化器。训练结束后就是最终参数。
- `model_best.pt`：验证误差最低一轮的模块参数。
- 上述模型文件不是完整 MapAnything 权重；使用时须加载记录的预训练版本、接入对应模块，再恢复 parameters 中的同名参数。模块权重文件不能传给 resume。
- `checkpoints/latest.pt`：最近完成的 epoch，用于恢复。
- `checkpoints/best.pt`：验证场景平均相对误差最低的参数。只有全部验证场景距离有效才参与最佳选择；失败数单独记录。

```python
# 新进程中重新创建相同配置的 Trainer 和 Dataset 后：
trainer.fit(train, val, epochs=20, resume="checkpoints/graspnet_ab/checkpoints/latest.pt")
```

`epochs` 是目标总轮数，不是额外训练轮数。恢复要求模型版本、学习率、模块类型和数据/采样配置一致，重新加载相同的冻结预训练权重，再恢复训练过的模块及优化器。更换数据开始第二阶段属于新的训练阶段，不能直接通过严格 resume 接口换数据。中途打断时从最近完成的一轮恢复；不承诺跨设备逐位一致。旧实验的 `modules.pt` 不是这一格式的完整断点。

## 评估边界

验证在原始点图上按最近像素取点，计算所有标注点对的距离。先对场景内的点对和输入子集取平均，再对场景取平均，避免标签较多的场景支配结果。无效预测场景记录为失败，汇总均值仅涵盖有效场景，必须结合失败数阅读。此验证用于选模型，不替代使用 mask 规则的最终正式评估。

代码检查相同 `(source, scene_id)` 和图片哈希的训练/验证重叠；同一布置的不同距离设置或不同相机仍需在制作清单时按布置分组，代码无法从图片自动识别。

第一版不使用默认 DataLoader 拼接，不跨场景合并 batch；固定每次一个场景，兼容不同图片尺寸。每轮采样由 Dataset seed/epoch 决定；验证 Dataset 必须设置 training=False。


## 独立加载已训练的模型

训练配置在 `training_config.json`，模型结构和基础模型版本在 `model_config.json`。后者由保存时实际的 A/B 模块推导。`model.pt` / `model_best.pt` 只保存模块权重和匹配元信息，不包含优化器。新的模型文件格式为 close_range_model_v1，旧 close_range_modules_v1 导出文件不能直接加载；旧完整续训 checkpoint 仍可用于同配置续训，并在下一轮导出新格式。

```python
from close_range.models.model_io import load_model

model = load_model(
    "checkpoints/graspnet_ab",
    device="mps",
    weights_name="model_best.pt",  # 最后一轮则选择 model.pt
)
```

返回模型已关闭参数梯度并处于 eval 模式。照片仍需按项目原流程预处理、放到对应设备；可使用已有 `predict_scene` 进行 RGB-only 预测。加载不包含照片预处理功能。若验证全部失败没有生成 model_best.pt，加载会明确报错，不自动换成其他权重。

项目保存目录 checkpoints 默认不提交 Git，权重需要单独共享。安装代码与项目内的 UniCeption 依赖仍然必需；只复制权重文件无法运行模型。原始 Hugging Face 预训练权重从缓存加载，缓存缺失时需要联网下载。

## 命令行训练入口 main.py

在项目根目录运行。所有相对路径以项目根目录为基准，不会自动随机划分数据。
先创建 `configs/graspnet_split.json`，格式如下（仅示例，不是正式划分）：

```json
{
  "train": ["data/manifests/graspnet/scene_0000_realsense_pilot.json"],
  "val": ["data/manifests/graspnet/scene_0001_realsense_pilot.json"],
  "test": ["data/manifests/graspnet/scene_0002_realsense_pilot.json"]
}
```

train/val 必须非空，test 可省略。清单内路径也相对于项目根目录。
入口检查原始场景（original_scene）和清单图片哈希是否跨集合重复；test 只检查划分元信息，不读取其标签计算指标。
同一布置使用不同名字的手机场景仍需人工归入同一集合。

```bash
# M2；M3 改为 B，M4 改为 AB，并更换输出目录
bash scripts/local.sh -m close_range.training.main \
  --variant A --split configs/graspnet_split.json \
  --device mps --epochs 2 --output-dir checkpoints/graspnet_A

# 下一阶段：继承相同模块类型的参数，重新创建优化器
bash scripts/local.sh -m close_range.training.main \
  --variant A --split configs/graspnet_split.json \
  --init-from checkpoints/graspnet_A --weights-name model_best.pt \
  --device mps --epochs 10 --output-dir checkpoints/graspnet_A_stage2

# 可选：完整恢复；训练设置和数据须与原实验一致
bash scripts/local.sh -m close_range.training.main \
  --variant A --split configs/graspnet_split.json \
  --resume checkpoints/graspnet_A/checkpoints/latest.pt \
  --device mps --epochs 10 --output-dir checkpoints/graspnet_A
```

`--init-from` 和 `--resume` 互斥。指定文件缺失时直接报错，不自动从头训练。
新实验/新阶段使用空输出目录；热启动沿用来源记录的基础模型版本，模块结构必须匹配，并在训练配置中记录来源与权重指纹。
`--weights-name model.pt` 可选择最后一轮，而不是验证最优参数。

默认学习率 `5e-4`，一致性权重 A 为 0，B/AB 为 0.1，可用 `--learning-rate` 和 `--consistency-weight` 调整。这只是试跑起点，不代表最优配置。
三组统一启用双子集采样：至少三张图时采样两组共享参考图的双视角输入；A 不使用一致性惩罚，但两组都参与距离监督。
Mac 使用 `mps`，Colab GPU 使用 `cuda`。入口不自动下载 GraspNet、不做正式测试集评估。

## PyCharm 直接运行

打开 `main.py`，修改顶部 `TRAIN_ARGS` 数组，然后直接点击 Run。
数组依次为：模块类型、划分文件、输出目录、设备、轮数、学习率、一致性权重、随机种子、历史模型目录、历史权重文件。
历史模型目录填 `None` 时初始化新模块；填写目录时加载已有模块参数。所有相对路径基于项目根目录，与 IDE 工作目录无关。
也可以运行 `python src/close_range/training/main.py`，不传参数时使用数组；原命令行参数方式仍可使用。

直接运行需要在 PyCharm 中选择已安装项目依赖的 Python 解释器。若报 `No module named cv2`，说明当前解释器缺少 OpenCV，修改参数不能解决；应切换到已配置好的环境，或在当前环境安装 `opencv-python`。
划分文件必须事先准备，入口不会自动生成训练/验证/测试划分。

## 预处理完成后生成固定划分

```bash
bash scripts/local.sh scripts/graspnet/make_split.py
```

只执行一次，已有划分不会覆盖。要求 190 个原始场景的处理清单齐全，旧 pilot 排除。
官方 0–99 场景划为 80 个训练、20 个验证；100–189 的 90 个场景留作测试。
已用于开发的 0000、0017 固定在验证集，其余按种子 443 划分。同一原始场景的所有组进入同一集合。
生成 configs/graspnet_split.json，A/B/AB 共用。若其他测试场景也已用于调参，正式实验前应调整测试范围并记录。
PyCharm 必须选择本项目 .venv/bin/python（Python 3.12），而非此前外部的 Python 3.14 环境。

## 多来源共用划分（推荐）

```python
from close_range.data_split import make_split

make_split(
    ["data/manifests/graspnet", "data/manifests/phone"],
    "configs/mixed_split.json",
)
```

也可直接运行 `scripts/tools/make_split.py`，修改顶部 MANIFESTS_PATH 数组和 OUTPUT_PATH。
只传 phone 目录就生成 Phone 划分；只传 graspnet 就生成 GraspNet 划分。
每个目录必须存在且有清单，图片与标注须通过 Dataset 校验，缺失时明确报错，不生成半份混合配置。
GraspNet 保留官方测试范围，训练范围按 80/20 分训练与验证；不再要求必须下载齐 190 个场景。
Phone 按独立场景约 70/15/15 划分，至少 3 个场景。已开发的 scene-001 放验证集，其他已开发场景需人工复核，不能冒充独立测试。
Phone 的同一布置若有多个编号，应在 manifest 的 original_scene 中填写同一分组标识。
训练数组中使用 configs/mixed_split.json，即会读取两类训练数据；各来源验证结果由 Trainer 分别汇总。
划分生成一次后固定使用，已有文件拒绝覆盖。此功能不会自动恢复缺失数据，也不意味着两种来源在训练中数量平衡。
旧 scripts/graspnet/make_split.py 仅是单来源兼容入口，共用同一实现，不再跳过旧 pilot：缺少原始场景信息的清单需先整理。
