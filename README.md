# MapAnything 近距离测距项目

本项目使用预训练 MapAnything，根据同一静态场景的一张或多张照片预测三维点图，再计算人工指定的两个表面点 A、B 之间的直线距离，与尺子实测距离比较。

当前已实现 M0 baseline 的数据准备、参考图选点、单图/多图推理和距离评估。模型改动 A/B、微调训练尚未实现。

## 按场景目录批量测试（推荐入口）

预处理完成后只需指定图片父目录，不必逐个指定场景，也不必手动选择 single/multi：

```bash
bash scripts/local.sh scripts/predict_directory.py --input data/processed/phone
```

也可使用绝对路径，例如本机：

```bash
bash scripts/local.sh scripts/predict_directory.py --input /Users/linshanyu/PycharmProjects/github.com/LinshanYu1/mapanything-close-range/data/processed/phone
```

程序按名称排序遍历父目录的直接子目录，每个子目录是一组场景。一张图片自动单图预测，两张及以上自动联合预测；不同场景不会混在一起。支持图片直接放在场景目录，或放在该场景的 images 子目录（不能同时两处放图）。输入必须是已预处理的图片，不能直接传 raw。

默认自动查找 `data/manifests/<父目录名>/<场景名>.json`，例如 phone/scene-001.json；已有清单的图片必须与目录一致，否则该场景报错，不会悄悄漏掉新图片。清单保持其参考图设定；没有清单时按图片文件名选第一张参考图，只预测、不猜标签。当前评分仍针对参考图上的标注点对，不会自动标注其他视角。

整批只加载一次模型。存在有效标注时自动评分；没有标注则保存三维预测并标明未评分；标签异常会记录 evaluation_failed。失败场景写入汇总并继续下一组，若有失败最终退出码为 1。

输出自动创建 `outputs/batch_<时间戳>/`：

```text
batch_<时间戳>/
├── summary.csv             # 所有场景的状态和各点对距离/误差
├── summary.json
├── scene-001/              # 本场景预测、run.json、metrics 等
└── scene-002/
```

可选参数：`--output outputs/批次名` 指定新的输出目录；`--device cuda` 使用 NVIDIA GPU；`--manifests-dir 路径` 指定自定义清单目录；`--check-only` 仅检查输入并生成检查汇总，不加载模型。输出目录存在则拒绝覆盖。

例如测试 GraspNet 已处理目录：

```bash
bash scripts/local.sh scripts/predict_directory.py --input data/processed/graspnet
```

下方的单场景命令保留用于诊断或专门进行单图/多图对照，日常批量测试优先使用本入口。

## 一、先理解三个不同的东西

- **实测距离**：用尺子量出来的答案，填写在 `distances.conf`。可写 `cm`、`mm`、`m`、`in` 单位，不写单位按 inch。
- **点位标注**：在图片窗口中点击 A、B，告诉程序测哪两个位置。脚本保存的是像素坐标，不会从点击行为得知真实距离。
- **预测距离**：模型先预测整张图的三维点位置，评估脚本再取出 A、B 对应的三维坐标计算距离。

模型预测时不读取实测距离，也不使用真实距离校正尺度。

## 二、当前可运行的流程

```text
固定物体，选定两个物理端点并量尺
                ↓
拍摄同一场景的多视角原图，填写 distances.conf
                ↓
phone/prepare.py：统一方向、RGB、官方缩放裁剪，生成处理图和清单
                ↓
phone/annotate.py：在处理后的第一张参考图上点击 A、B
                ↓
run_baseline.py：仅用图片预测参考图的三维点图
                ↓
evaluate.py：读取 A/B 坐标，计算距离，与实测值比较
```

**当前是先预处理、再标注第一张参考图。** 讨论过的“自动遍历全部场景、逐张原图标注、再转换坐标”的流程尚未实现。不要把计划中的功能当成当前脚本行为；当前标注入口用 `--scene` 或 `--manifest` 指定场景，不带参数时依次标注所有尚未标注的场景。

## 三、怎么拍照和测量

1. 两个物体固定不动，选取清晰、可以再次找到的表面端点 A、B。可以在实物上贴纸并画小十字，以交点作为端点。
2. 用尺子测量两个端点之间的空间直线距离，避免尺子弯曲或沿物体表面绕行。两点高度不同就量连接两点的斜线，不是水平投影距离。
3. 记录数值并写明单位，例如 `43cm` 或 `17in`（不写单位按 inch）。单位转换不意味着手工测量精度达到 0.01 cm。
4. 移开尺子和手，不改变物体位置，从不同位置拍照；保留共同可见区域，确保参考图能清楚看到两个端点。
5. 保存未后期画点、未添加 A/B 文字的原照片。建议保持同一镜头、清晰对焦和稳定光照，传输时尽量保留原始文件。

不需要使用画图软件在照片上加点。鼠标点击标注即可。若已经画了点，请把那份作为位置说明副本，模型使用未加点版本。

### 本机 scene-001 的特殊情况

目前 `raw/phone/scene-001/` 的 JPG 后来被手工添加了黑点和 A/B；`processed/phone/scene-001/` 保留了修改前生成的无黑点 PNG。本次已完成的测试使用后者。**不要以为 raw 和 processed 目前是同一版本，也不要重新用带黑点 JPG 准备正式模型输入。** 未来重建该场景前应恢复未修改原图。现有处理图和清单可以继续用于此次试跑。

## 四、目录分别放什么

```text
data/
├── raw/phone/
│   ├── distances.conf
│   ├── scene-001/                 # 场景目录直接放输入 JPG/PNG
│   │   ├── view_001.jpg
│   │   ├── view_002.jpg
│   │   └── view_003.jpg
│   └── scene-002/
├── processed/phone/scene-001/
│   ├── view_001.png               # 处理后的模型输入，不带后加标记
│   ├── view_002.png
│   └── view_003.png
├── annotations/phone/
│   ├── scene-001.json             # 参考图 A/B 像素坐标和实测距离
│   └── scene-001.preview.png      # 带 A/B 的检查图，不输入模型
└── manifests/phone/
    └── scene-001.json             # 图片路径、顺序、尺寸和标注路径清单

outputs/某次实验/
├── run.json                      # 输入、模型版本、设备和推理时间
├── reference_prediction.npz       # 参考图的三维点图和有效性掩码
├── metrics.csv                   # 距离与误差表
└── metrics.json                  # 评分结果
```

`processed` 是模型实际读取的图片；`annotations` 是你指定的测量位置与标签；`manifest` 是自动生成的文件索引，告诉程序到哪里寻找图片和标签。无需手写 manifest。

当前没有单独的 `processed/.../previews/` 或 `points.json`；这些属于后续逐张原图标注方案。

## 五、运行环境

所有命令在仓库根目录执行。本机为：

```bash
cd /Users/linshanyu/PycharmProjects/github.com/LinshanYu1/mapanything-close-range
```

`bash scripts/local.sh ...` 是运行入口：先切换到项目根目录，再选择 Python，配置上游代码路径和缓存，最后运行指定脚本。

Python 选择顺序为：`PYTHON_BIN` 指定的解释器 → 本项目 `.venv/bin/python` → 本机旧项目 `/Users/linshanyu/Desktop/dl-project/.venv-baseline/bin/python`。

本机默认复用旧项目的 Hugging Face 和 Torch 缓存；不要删除旧环境后还期待启动器正常运行。不要直接使用未安装依赖的 IDE 默认解释器。

其他同学需根据 `third_party/map-anything/README.md` 安装匹配的依赖环境，设置自己的路径，例如：

```bash
export PYTHON_BIN=/你的环境/bin/python
export HF_HOME="$PWD/.cache/huggingface"
export TORCH_HOME="$PWD/.cache/torch"
```

这不是一个已完成一键安装的跨机器环境。依赖安装遵循固定上游快照说明，未缓存的模型文件首次需要下载。上游版本来源见 `third_party/upstream.json`；权重固定为 `facebook/map-anything` 的 revision `a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa`。

## 六、从新照片开始操作

### 1. 放照片，填写距离

以新场景 `scene-002` 为例，照片放在 `data/raw/phone/scene-002/`。目录内只放模型输入图片，不放尺子记录图或标注预览。当前准备脚本支持 JPG、JPEG、PNG；HEIC 请先转换为保留原始内容的 JPEG/PNG。

编辑 `data/raw/phone/distances.conf`：

```ini
# 可写 cm/mm/m/in，不写单位按 inch；以下 scene-002 数值仅作示例，应替换为实际测量值
[distances]
scene-001 = 17
scene-002 = 25cm
```

一个场景默认一个固定物理点对和一个距离。目录名必须与 key 完全一致，距离必须为正的有限数字。

### 2. 准备图片

```bash
bash scripts/local.sh scripts/phone/prepare.py --scene scene-002
```

脚本校正 EXIF 方向、转成 RGB，按这组图片的平均宽高比选择官方输入尺寸，再进行保持比例的缩放和必要裁剪。当前 scene-001 为 518×392，其他比例可能不同。清单记录的是处理图文件哈希，后续加载会检查它是否改变。

第一张参考图按文件名排序决定，建议使用 view_001、view_002 这样的编号。

已存在的 processed 场景或 manifest 不会被覆盖，以免破坏已有坐标。不要反复运行准备脚本；新一组数据使用新场景名。

### 3. 点击两个测量点

```bash
bash scripts/local.sh scripts/phone/annotate.py --scene scene-002
```

程序读取配置中的实测距离，在窗口显示处理后的参考图。用左键先点端点 A，再点端点 B，两点选齐即保存。可用工具栏缩放查看细节，退出缩放模式后再选点；第二点确认前可用右键撤销。必须点击尺子实际测量的端点，不能随意换成白纸中心。

输出位于 `data/annotations/phone/scene-002.json` 和 `scene-002.preview.png`。先查看预览确认圆点中心位置正确。模型输入图不会被绘制标记。

重新选择：

```bash
bash scripts/local.sh scripts/phone/annotate.py --scene scene-002 --overwrite
```

如果修改了 distances.conf，也需重新确认标注。评估会拒绝配置与已保存标签不一致的结果。坐标属于处理后图片，不能直接填原始大图坐标。

### 4. 单图推理与评分

```bash
bash scripts/local.sh scripts/run_baseline.py --manifest data/manifests/phone/scene-002.json --mode single --output outputs/scene002_single_001
bash scripts/local.sh scripts/evaluate.py --manifest data/manifests/phone/scene-002.json --run outputs/scene002_single_001
```

### 5. 多图推理与评分

```bash
bash scripts/local.sh scripts/run_baseline.py --manifest data/manifests/phone/scene-002.json --mode multi --output outputs/scene002_multi_001
bash scripts/local.sh scripts/evaluate.py --manifest data/manifests/phone/scene-002.json --run outputs/scene002_multi_001
```

`single` 只输入参考图；`multi` 输入清单中全部图片，但仍评估同一参考图上的同一对点。当前不对三张图片分别评分。

默认设备为 Mac MPS；有 NVIDIA GPU 时给推理命令加 `--device cuda`。CPU 可显式指定 `--device cpu`，可能很慢。输出目录已经存在时请换新的实验名。单纯修改点位可以复用同一份预测重新评分，但应保留旧标签/结果以便比较。

## 七、现有 scene-001 快速复跑

该场景已经预处理并保存参考图标注，无需重新准备。确认 preview 中的 A/B 后直接运行：

```bash
bash scripts/local.sh scripts/run_baseline.py --mode single --output outputs/phone_single_002
bash scripts/local.sh scripts/evaluate.py --run outputs/phone_single_002
bash scripts/local.sh scripts/run_baseline.py --mode multi --output outputs/phone_multi_002
bash scripts/local.sh scripts/evaluate.py --run outputs/phone_multi_002
```

若要重新选点，运行 `bash scripts/local.sh scripts/phone/annotate.py --scene scene-001 --overwrite`。

## 八、怎么看结果

打开输出目录中的 metrics.csv：

| 字段 | 含义 |
|---|---|
| reference_inch / reference_cm | 实测距离 |
| predicted_inch / predicted_cm | 模型预测的点对距离 |
| absolute_error_cm | 预测距离与实测距离之差的绝对值 |
| relative_error_pct | 绝对误差除以实测距离，再乘 100% |
| prediction_valid | 两点坐标、三维值和模型掩码是否有效 |
| success_10pct | 是否有效且相对误差不超过 10% |
| reason | 无效预测的原因 |

内部统一使用米；展示同时提供 inch/cm。原模型输出的 pts3d 已含实际尺度，不重复乘尺度。不使用真实距离对预测结果缩放。不以附近点替换无效点，预测无效算失败。

### 已完成的本机试验（2026-10-30）

同一场景、第一张参考图同一对点，用户实测 17 inch（43.18 cm）：

| 输入 | 预测 inch | 预测 cm | 绝对误差 cm | 相对误差 |
|---|---:|---:|---:|---:|
| 单图 | 28.18 | 71.57 | 28.39 | 65.74% |
| 三图联合 | 26.68 | 67.77 | 24.59 | 56.95% |

结果保存在 `outputs/phone_single_check_20261030/` 和 `outputs/phone_multi_check_20261030/`。两次预测有效，但均未达到 10% 误差标准。三图有所改善；这仅说明此场景的当前结果，不能代表模型整体表现，也不能确定误差原因。

“流程跑通”和“预测准确”是两个不同结论。先检查点位、单位和模型输入，再讨论模型误差。

## 九、GraspNet 如何共用这套流程

手机标签来自尺子；GraspNet 标签来自 RGB 对齐的深度和相机内参。两者转换成统一清单与点对标签后，共用 run_baseline.py 和 evaluate.py。

当前提供旧 GraspNet pilot 的兼容导入，不是完整下载和批量标签生成器：

```bash
bash scripts/local.sh scripts/graspnet/import_legacy.py --legacy-manifest /Users/linshanyu/Desktop/dl-project/output/distance_pilot_20/manifest.json --legacy-root /Users/linshanyu/Desktop/dl-project
bash scripts/local.sh scripts/run_baseline.py --manifest data/manifests/graspnet/graspnet_pilot.json --mode single --output outputs/graspnet_single_002
bash scripts/local.sh scripts/evaluate.py --manifest data/manifests/graspnet/graspnet_pilot.json --run outputs/graspnet_single_002
```

本机该 GraspNet 场景已导入，不必重跑导入命令。导入器选择首个标签质检合格的参考视图，保留原标签来源和预处理方式；传感器标签不等于独立尺量真值。RGB-only 推理不提供真实深度或相机参数。

## 十、脚本职责和后续工作

脚本按数据来源分组：`scripts/phone/` 处理手机照片，`scripts/graspnet/` 处理 GraspNet，`scripts/tools/` 预留共用检查工具。预测与评估入口留在 `scripts/`，两类数据共用。底层共用函数放在 `src/close_range/`。

本次仅调整目录，没有改变数据格式或处理顺序：手机照片仍然先 prepare，再 annotate，然后 predict/evaluate。GraspNet 已提供 download.py、prepare.py 及旧数据导入器，详见 [GraspNet 操作说明](scripts/graspnet/README.md)。

| 脚本 | 职责 |
|---|---|
| local.sh | 选择环境、设置缓存、启动 Python |
| src/close_range/common.py | 文件读写、路径、版本、距离配置校验 |
| phone/prepare.py | 默认扫描手机场景，也支持 --scene 指定场景 |
| phone/annotate.py | 参考图 A/B 点击标注；默认依次标注未标注场景，--scene 指定单个 |
| predict_directory.py | 遍历场景目录，自动选择单图/多图预测并评估 |
| tools/ | 共用检查工具预留目录 |
| run_baseline.py | RGB-only 模型预测，不读取标签 |
| evaluate.py | 读取预测和标签，计算测距误差 |
| graspnet/download.py | 下载官方包、选择场景解压 |
| graspnet/prepare.py | 原始 RGB-D 数据预处理与自动点对标签生成 |
| graspnet/import_legacy.py | 导入旧 GraspNet pilot 数据 |
| legacy/distance_pilot/ | 旧实验代码存档，含旧路径假设，不是新流程入口 |

后续计划：默认扫描各场景并逐张原图标注、指定场景重新标注、精确记录缩放裁剪坐标转换、逐视图评估、实现 A/B 和训练流程。以上均不应视为已完成。

原始图、处理图、大型权重和实验输出默认不提交 Git。共享数据应使用单独的数据存储方式；正式实验需要按场景划分训练/验证/测试集。

分来源操作说明：[手机照片](scripts/phone/README.md)、[GraspNet](scripts/graspnet/README.md)。legacy 是旧实验存档，不参与当前流程。
