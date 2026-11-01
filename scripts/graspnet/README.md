# GraspNet 下载与预处理

在项目根目录运行。官方来源：https://graspnet.net/datasets.html

## 1. 下载

```bash
bash scripts/local.sh scripts/graspnet/download.py --pack train_1 --scenes scene_0017
```

需要当前 Python 环境安装 `gdown`。官方训练包约 6.3–20 GB；脚本下载完整 ZIP，然后只解压指定场景和相机的数据，并非按场景联网下载。默认 RealSense，可用 `--camera kinect`。Google Drive 可能限流；可以通过官方网站其他镜像手动下载，再执行：

```bash
bash scripts/local.sh scripts/graspnet/download.py --archive /absolute/path/train_1.zip --scenes scene_0017
```

输出 `data/raw/graspnet/scenes/scene_0017/realsense/`，保留 RGB、depth、label、meta 和相机参数。拒绝覆盖已有文件。下载可以续传，解压中断后的部分文件需自行检查并移走后重试。指定场景必须存在于所选包内，脚本会检查。

不传 `--scenes` 时，提取所选 ZIP 内所有场景（不是自动下载全部训练包）：

```bash
bash scripts/local.sh scripts/graspnet/download.py --pack train_1
```

## 2. 预处理

默认扫描 `data/raw/graspnet/scenes/` 下所有非隐藏场景目录：

```bash
bash scripts/local.sh scripts/graspnet/prepare.py
```

自定义根目录也可扫描；传入的根目录必须直接包含 scene_XXXX 文件夹：

```bash
bash scripts/local.sh scripts/graspnet/prepare.py --raw-root /absolute/path/scenes
```

传 `--scene` 则只处理该场景：

```bash
bash scripts/local.sh scripts/graspnet/prepare.py --scene scene_0017 --frames 0000 0012 0024 --pairs 10
```

首张图作为参考图。默认随机种子 0，自动选择不同物体内部的 10 对点；过滤背景、无效深度和分割边缘。按 meta 中 factor_depth 转换为米，使用 intrinsic_matrix 反投影后计算三维直线距离。图片采用上游缩放裁剪，另用最近邻源像素映射生成对应标签；存在像素采样精度限制。

输出：
- `data/processed/graspnet/<组名>/`：纯 RGB 模型输入。
- `data/annotations/graspnet/<组名>.json`：点坐标、距离和来源记录。
- `data/annotations/graspnet/previews/<组名>/`：每一对点的预览，请检查。
- `data/manifests/graspnet/<组名>.json`：统一输入清单。

默认组名 `scene_0017_realsense_pilot`；可用 `--name` 新建其他组，不覆盖已有结果。清单含绝对路径，移动数据后需重新生成。标签来自传感器，不是尺量真值；基本过滤不能保证反光表面等区域的深度准确。只有参考图有点对标签，不执行跨视图点跟踪。一个场景的不同组必须进入同一个训练/验证/测试划分。

复用本机已有旧数据，无须下载：

```bash
bash scripts/local.sh scripts/graspnet/prepare.py --raw-root /Users/linshanyu/Desktop/dl-project/data/graspnet_audit --scene scene_0017 --frames 0000 0012 0024 --pairs 10
```

## 3. 预测和评估

```bash
bash scripts/local.sh scripts/predict_directory.py --input data/processed/graspnet --check-only
bash scripts/local.sh scripts/predict_directory.py --input data/processed/graspnet
```

遍历此目录全部场景组（含以前导入的组），每组单图或多图自动判断，有标签则自动评估。模型只读取 RGB，不接收真实深度或相机参数。检查与预测分别生成新的输出目录。

`import_legacy.py` 仍用于导入旧的已处理 pilot，与新 `prepare.py` 是不同入口。

默认每个场景选择 0000、0012、0024 三个视角，并非全部 256 张；用 --frames 可修改，所有场景使用同一组帧编号。已有 processed 目录、标注和清单时跳过，不自动检查原数据变更；部分输出存在时报告失败。单场景失败不阻断其他场景，最终返回非零退出码。--name 只能与 --scene 一起使用，换参数重新生成时应换组名。

## 4. 简单验证方法与本次结果（2026-10-31）

本节验证的是“本地 GraspNet 数据能处理成模型可加载的输入”。不下载整个官方包、不加载模型权重，也不评价测距精度。

已有标准 raw 数据时，在项目根目录运行：

```bash
bash scripts/local.sh scripts/graspnet/prepare.py
bash scripts/local.sh scripts/predict_directory.py --input data/processed/graspnet --check-only
```

若第一步跳过已有组，只代表复用输出；重新验证预处理请使用新的输出数据目录。以下是本机利用旧项目数据进行隔离验证的命令（再次执行时更换临时目录名称）：

```bash
bash scripts/local.sh scripts/graspnet/prepare.py --raw-root /Users/linshanyu/Desktop/dl-project/data/graspnet_audit --data-root /private/tmp/graspnet-my-check --pairs 10

bash scripts/local.sh scripts/predict_directory.py --input /private/tmp/graspnet-my-check/processed/graspnet --manifests-dir /private/tmp/graspnet-my-check/manifests/graspnet --check-only --output /private/tmp/graspnet-my-check/input-check
```

检查方法：

1. 预处理终端应显示生成的视图数、点对数及清单路径。
2. 查看 `annotations/graspnet/previews/<组名>/` 的 A/B 点预览，确认点落在预期物体表面。自动选点与深度过滤不等于人工确认无误。
3. 批量检查应显示 `checked`；查看其输出目录的 `summary.csv`，失败时读取 error 列。

本次实测：默认扫描本地根目录，发现 scene_0017，使用 RealSense 的 0000、0012、0024 三帧重新生成数据；3 张 RGB PNG 均为 518×294。实际推理入口的 `prepare_inputs` 和官方 `load_images` 均成功读取。额外检查了 10 对标签距离均为正且有限、A/B 坐标位于图片范围内，并确认生成了 10 张预览。批量输入状态为 `checked`。

本次检查记录：`/private/tmp/data-branches-20261031-ey588qfi/graspnet_check/summary.csv`。临时目录可能被系统清理，复验时使用上述命令生成新记录。

范围说明：本次没有实际测试官方网络下载，没有执行模型推理，没有人工逐张审核点位，也没有独立验证传感器距离的准确度。
