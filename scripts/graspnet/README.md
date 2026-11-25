# GraspNet 下载与预处理

本文先介绍每个脚本的作用和用法，再说明完整处理顺序与完成标准。所有命令均在项目根目录执行，使用 `bash scripts/local.sh` 选择项目 Python 环境。

官方数据说明：[GraspNet 数据集](https://graspnet.net/datasets.html)。

## 1. 每个脚本做什么、怎么用

| 脚本 | 作用 |
| --- | --- |
| `download.py` | 下载一个官方训练包，提取指定场景和相机 |
| `download_all.py` | 下载并解压全部 7 个官方场景包 |
| `download_mirror.py` | 从第三方镜像下载完整场景数据，校验并解压 |
| `prepare.py` | 把原始数据转换成模型输入图片、点对距离标签和清单 |
| `make_split.py` | 按原始场景生成固定的训练、验证、测试清单，不复制图片 |
| `../predict_directory.py` | 通用脚本：批量检查输入，或执行模型预测和距离评估 |

三个 download 脚本是不同下载入口，选择一个即可，不需要全部运行。“全部下载”指场景数据，不包含网站上其他独立资源，例如物体模型和抓取标注包。

### 1.1 download.py：一个官方训练包

适合先试少量场景，或已经手动下载了 ZIP。运行环境需要 `gdown`。


```bash
# 从 train_1 提取一个场景，默认使用 RealSense
bash scripts/local.sh scripts/graspnet/download.py --pack train_1 --scenes scene_0017

# 提取 train_1 中全部场景
bash scripts/local.sh scripts/graspnet/download.py --pack train_1

# 已经有 ZIP：跳过网络下载
bash scripts/local.sh scripts/graspnet/download.py --archive data/raw/graspnet/archives/train_1.zip --scenes scene_0017
```

`--pack` 支持 `train_1` 至 `train_4`。`--camera kinect` 可改选 Kinect。指定场景只减少解压量，仍需下载整个 ZIP。此脚本拒绝覆盖已有文件，解压中断后重跑可能需要先检查已有输出。


### 1.2 download_all.py：全部官方场景包


```bash
bash scripts/local.sh scripts/graspnet/download_all.py
```

需要 `gdown`，默认同时下载两个包；减少并行数：

```bash
bash scripts/local.sh scripts/graspnet/download_all.py --workers 1
```

仅下载、检查 ZIP，不解压：

```bash
bash scripts/local.sh scripts/graspnet/download_all.py --download-only
```

仅下载完成后，再运行不带 `--download-only` 的命令才能解压。压缩包在 `data/raw/graspnet/archives/`，进度在 `data/raw/graspnet/download_status.json`。正常结束会显示 `All seven scene archives complete.`。使用 `--download-only` 时，这句话不代表场景已经解压。

Google Drive 可能因下载配额报错；此时可改用 mirror。解压保留官方包目录结构，预处理前确认场景实际位于 `data/raw/graspnet/scenes/`。此脚本不生成 mirror 的 `scene_inventory.json`。


### 1.3 download_mirror.py：完整场景镜像（当前方式）


```bash
bash scripts/local.sh scripts/graspnet/download_mirror.py
```

来源为 `DravenALG/GraspNet-1Billion`。脚本记录本次下载版本，下载约 142.5 GB 的 TAR；磁盘还需要容纳解压后的文件。运行环境需安装 `huggingface_hub`。

脚本依次执行下载、文件哈希校验、解压、场景文件完整性检查，不需要再手动解压。输出：

- `data/raw/graspnet/archives/mirror/scenes.tar`：压缩包。
- `data/raw/graspnet/scenes/`：解压后的场景。
- `data/raw/graspnet/mirror_status.json`：阶段与校验记录。
- `data/raw/graspnet/scene_inventory.json`：场景与相机文件数量清单。

成功时 `mirror_status.json` 中 `state` 为 `complete`、`sha256_verified` 为 `true`、`scenes` 为 190、`rgb_images` 为 97280。清单应有 380 条记录（190 个场景 × 2 种相机），每条的 RGB、depth、label、meta 各 256 个文件。

注意：哈希核对的是镜像提供的校验值，不是官方独立校验值。当前下载依赖 Hugging Face 缓存机制，中断后重跑不保证从断点继续，可能重新下载。状态文件记录的是上次写入状态，不能仅凭 `downloading` 判断进程仍在运行。已有解压文件只检查大小后跳过，不逐个重新校验内容。


### 1.4 prepare.py：生成模型输入与距离标签


原始数据应该类似：

```text
data/raw/graspnet/scenes/
└── scene_0000/
    ├── realsense/
    │   ├── rgb/0000.png
    │   ├── depth/0000.png
    │   ├── label/0000.png
    │   └── meta/0000.mat
    └── kinect/
        └── ...
```

先处理一个场景，确认成功后再处理全部：

```bash
bash scripts/local.sh scripts/graspnet/prepare.py --scene scene_0000 --frames 0000 0012 0024 0036 0048 --pairs 10

bash scripts/local.sh scripts/graspnet/prepare.py --frames 0000 0012 0024 0036 0048 --pairs 10
```

不传 `--scene` 就扫描原始目录下的所有场景。上面每个场景选 5 张图片、生成 10 对距离标签；默认相机为 RealSense。省略 `--frames` 时只有 `0000 0012 0024` 三张，不是全部 256 张。要处理 Kinect，增加 `--camera kinect`。

预处理具体做了这些事：

1. 读取选定帧的 RGB、深度、物体分割标签和相机参数。
2. 将 RGB 按模型要求缩放、裁剪，保存不带标记的模型输入图片。
3. 在第一张参考图中自动选择不同物体内部的 A/B 点，过滤背景、无效深度和部分边缘点。
4. 用深度和相机内参还原两个点的三维位置，计算直线距离，标签单位为米。
5. 保存点坐标、距离、预览图，以及描述模型输入的 manifest 清单。

例如 `scene_0000` 会生成：

```text
data/
├── processed/graspnet/scene_0000_realsense_pilot/  # 模型输入 RGB
├── annotations/graspnet/scene_0000_realsense_pilot.json  # 点对与距离
├── annotations/graspnet/previews/scene_0000_realsense_pilot/  # A/B 预览
└── manifests/graspnet/scene_0000_realsense_pilot.json  # 输入清单
```

真实深度只用于生成监督标签；这里的 RGB baseline 推理不把真实深度交给模型。点对标签属于参考图，脚本不会自动标注同一物理点在所有视角的位置。

若显示 `skipped existing group`，表示复用已有结果，不代表按新的帧数或参数重新处理。更换设置时，可指定单场景和新的组名，例如：

```bash
bash scripts/local.sh scripts/graspnet/prepare.py --scene scene_0000 --name scene_0000_realsense_5views --frames 0000 0012 0024 0036 0048 --pairs 10
```

不要把同一场景的多个组当成独立场景划入不同数据集。新生成的 manifest 使用项目相对路径；复制完整项目和数据后仍可读取。项目外数据的相对目录关系需要保持。


补充参数：`--raw-root` 指定直接包含 `scene_XXXX` 的原始目录；`--data-root` 指定输出数据根目录；`--seed` 控制选点随机种子，默认 0。`--name` 只能和 `--scene` 一起使用。部分输出已存在时会报错；单个场景失败后继续其他场景，最后以非零退出码提示失败。

### 1.5 make_split.py：单来源与混合数据划分

共用函数位于 `src/close_range/data_split.py`：

```python
make_split(
    ["data/manifests/graspnet", "data/manifests/phone"],
    "configs/mixed_split.json",
)
```

统一入口 `scripts/tools/make_split.py` 顶部提供目录数组，可直接运行，或执行：

```bash
bash scripts/local.sh scripts/tools/make_split.py --manifests data/manifests/graspnet data/manifests/phone --output configs/mixed_split.json
```

旧 `scripts/graspnet/make_split.py` 保留为只传 GraspNet 目录的兼容入口，支持 --seed 和 --output。
默认 seed=443，已有文件不覆盖。输入目录、图片、标注缺失会报错，不静默省略某一数据来源。
按原始场景分组：GraspNet 保留 0100–0189 为测试，0000–0099 按 80/20 分训练和验证；0000、0017 固定用于验证。
不再要求完整 190 场景，但训练和验证必须非空；旧 pilot 缺少标准原始场景信息时需先整理。
Phone 约 70/15/15 划分，至少三个独立场景，scene-001 留在验证集。同一布置应填写相同 original_scene。
输出 train/val/test 的 manifest 列表和来源统计；训练时读取这一份 JSON 即可混合训练。
仓库已有正式划分时可直接复用，不必重新生成。其他已用于开发的场景需要人工复核测试独立性。

### 1.6 predict_directory.py：检查或预测

这个通用脚本位于 `scripts/` 下，遍历 `--input` 中每个场景目录；一张图片采用单图模式，多张采用多图模式。

```bash
# 只检查模型输入，不加载模型权重
bash scripts/local.sh scripts/predict_directory.py --input data/processed/graspnet --check-only

# 实际执行 baseline，有标签时计算测距误差
bash scripts/local.sh scripts/predict_directory.py --input data/processed/graspnet
```

默认查找 `data/manifests/graspnet/` 下的清单。自定义数据位置时，可以通过 `--manifests-dir` 指定清单目录，通过 `--output` 指定新的结果目录。每次检查或预测均生成结果汇总 `summary.csv`，终端会打印路径。各状态的判断方法见第 2 节。

## 2. 整个处理过程：按这个顺序执行

```text
选择一个下载脚本
    ↓
data/raw/graspnet/scenes/（RGB、深度、分割、相机参数）
    ↓ prepare.py
data/processed + data/annotations + data/manifests
    ↓ predict_directory.py --check-only + 标注预览抽查
确认数据准备完成
    ↓
make_split.py 生成固定划分（已有则复用）
    ↓
training/main.py 开始训练
```

### 2.1 下载并确认原始文件完整

当前采用 mirror：

```bash
bash scripts/local.sh scripts/graspnet/download_mirror.py
```

如果已经下载、解压成功，不需要重跑。确认 `mirror_status.json` 为 `complete`，并且 `scene_inventory.json` 记录了 190 个场景、两种相机的数据。具体输出和限制见 1.3。

### 2.2 先处理一个场景，再处理全部场景

```bash
bash scripts/local.sh scripts/graspnet/prepare.py --scene scene_0000 --frames 0000 0012 0024 0036 0048 --pairs 10

bash scripts/local.sh scripts/graspnet/prepare.py --frames 0000 0012 0024 0036 0048 --pairs 10
```

确认生成了图片、标注、预览和清单。第二条命令扫描全部原始场景，已经成功生成的同名组会跳过。如果以前用三帧生成过同名组，它不会自动变成五帧；重新生成方法见 1.4。

### 2.3 检查输入、标注与预测结果


**第一步：批量检查模型输入。**

```bash
bash scripts/local.sh scripts/predict_directory.py --input data/processed/graspnet --check-only
```

它逐组读取图片，按图片数量选择单图或多图输入方式，执行模型输入准备，但不加载模型权重、不执行预测。已有 manifest 时会核对图片清单，并检查输入图片等信息。

成功的组会显示 `checked`。终端最后打印 `summary.csv` 路径，打开这个文件检查：

- 每个预期场景组都有一行，而不是只检查了少数已有组。
- `status` 全部为 `checked`，没有 `failed`。
- `image_count` 与预期一致，例如上面命令的新组应为 5。
- 出现失败时查看 `error` 列，修复对应数据后重新检查。

**第二步：检查标注与预览。**

`check-only` 即使没有标注也可以检查图片，因此它不能单独证明训练数据已经完整。还要确认每组都有对应的 annotation JSON 和 manifest；打开若干场景的预览图，确认 A/B 点落在物体表面，而不是背景或明显错误区域。检查标签中的点坐标在图片范围内、距离为有限正数，点对数量符合设置。

GraspNet 标签来自传感器深度，不是手工尺量。自动过滤只能降低错误风险，不能保证所有深度标签都准确。

**可选第三步：实际跑 baseline。**

```bash
bash scripts/local.sh scripts/predict_directory.py --input data/processed/graspnet
```

这会遍历该目录全部组，加载模型并预测，有标签则计算距离误差，耗时明显长于 check。成功预测并评估的状态为 `evaluated`；`predicted_without_labels` 表示没有完成距离评估，`evaluation_failed` 表示预测后评估失败。模型误差大不直接等于预处理失败，需要另外排查标签与模型预测。


### 2.4 怎样才算准备完成


满足以下条件，才算本轮计划使用的数据已准备完成：

- **原始数据完整**：下载与解压成功；完整 mirror 应有 190 个场景及对应清单，不能只看压缩包存在。
- **预处理齐全**：计划使用的每个场景组都有 RGB、annotation、manifest 和点对预览。只处理部分场景时，应明确本轮范围。
- **输入检查通过**：预期的全部组都出现在 `summary.csv` 中，状态为 `checked`，图片数量正确。
- **标签抽查通过**：点位合理、坐标在图内、距离有效、标签数量正确。check-only 不能替代这一步。

如果以上都通过，就可以认为“下载和前期处理完成”，下一步是划分训练集、验证集和测试集并开始训练。同一原始场景的不同相机、视角和处理组必须放入同一划分，避免泄漏；数据全部下载下来不等于全部用于训练。


### 2.5 衔接训练：生成或复用划分

完成输入检查和标签抽查后，检查 `configs/graspnet_split.json`：

- **文件已存在**：复用它，确认对应 manifest 均已准备好，不必运行 make_split。
- **文件不存在**：执行 `bash scripts/local.sh scripts/graspnet/make_split.py`。

随后在 `src/close_range/training/main.py` 的数组中选择 A/B/AB、这份划分文件和一个新的空输出目录，再启动训练。
如果报“缺少数据划分”，不是要求重新下载或预处理，而是需要完成本步骤。
PyCharm 使用项目 `.venv/bin/python`（Python 3.12）。

## 3. 历史验证记录（2026-10-31）

以下为历史单场景验证，不代表当前完整数据已经逐一通过检查。本次实测：默认扫描本地根目录，发现 scene_0017，使用 RealSense 的 0000、0012、0024 三帧重新生成数据；3 张 RGB PNG 均为 518×294。实际推理入口的 `prepare_inputs` 和官方 `load_images` 均成功读取。额外检查了 10 对标签距离均为正且有限、A/B 坐标位于图片范围内，并确认生成了 10 张预览。批量输入状态为 `checked`。


范围说明：本次没有实际测试官方网络下载，没有执行模型推理，没有人工逐张审核点位，也没有独立验证传感器距离的准确度。
