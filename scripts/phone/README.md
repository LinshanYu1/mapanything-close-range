# 手机照片处理

在项目根目录运行。每个场景一个目录，场景内拍摄时保持物体位置不变，只移动相机。保留未画标记的原图；A/B 点通过标注工具点击，预览另存。

```text
data/raw/phone/
├── distances.conf
├── scene-001/
│   ├── view_001.jpg
│   └── view_002.jpg
└── scene-002/
    └── view_001.jpg
```

`distances.conf` 记录对应实际测量端点之间的距离，单位固定 inch：

```ini
[distances]
scene-001 = 17
scene-002 = 12
```

## 1. 图片预处理

默认扫描 `data/raw/phone/` 下所有非隐藏场景目录：

```bash
bash scripts/local.sh scripts/phone/prepare.py
```

只处理一个场景：

```bash
bash scripts/local.sh scripts/phone/prepare.py --scene scene-002
```

生成 `data/processed/phone/<场景>/` 图片及 `data/manifests/phone/<场景>.json` 清单。已有图片目录和清单时跳过，不覆盖、不自动检测原图更新；若只存在部分输出则报错。一个场景失败后继续其他场景，最终返回失败状态。

## 2. 点击 A/B 标注

当前标注入口仍然每次处理一个清单的参考图，并非自动逐场景弹窗：

```bash
bash scripts/local.sh scripts/phone/annotate.py --manifest data/manifests/phone/scene-002.json
```

依次点击实际测量的 A、B 端点，真实距离自动从 distances.conf 读取。输出 `data/annotations/phone/scene-002.json` 和 `scene-002.preview.png`。不修改模型输入图片。重新标注加 `--overwrite`；不指定清单时默认 scene-001。

## 3. 预测与评估

```bash
bash scripts/local.sh scripts/predict_directory.py --input data/processed/phone
```

一个目录一组，单图或多图自动选择。有标签则评估距离，没有标签只预测。输出目录在终端显示，汇总为 summary.csv 和 summary.json。

本机 scene-001 的原图曾手动加过黑点，但既有 processed 图片是之前保存的干净版本。默认跳过可保留这份输入；如果重新制作数据，应先恢复干净原图。

## 4. 简单验证方法与本次结果（2026-10-31）

本节验证的是“照片能处理成模型可加载的输入”，不加载预训练权重、不进行距离预测，也不要求重新点击 A/B。

在项目根目录运行：

```bash
# 第一步：扫描手机场景并准备图片
bash scripts/local.sh scripts/phone/prepare.py

# 第二步：通过官方图片加载器检查处理结果
bash scripts/local.sh scripts/predict_directory.py --input data/processed/phone --check-only
```

检查终端输出：每个场景应显示 `checked`，三张图片的组显示 `3 image(s) -> multi`。终端会打印检查输出目录，打开其中的 `summary.csv`，查看各场景 `status` 是否为 `checked`。任何 `failed` 都需要根据 error 列排查。

注意：若第一步显示 `skipped existing prepared scene`，表示沿用旧输出，不代表本次重新做了预处理。想测试全新输入，可以新增一个场景目录，并在 distances.conf 中填写其真实测量距离；不要覆盖已有场景。

本次实测过程：

1. 将 scene-001 的三张原始照片和距离配置复制到独立临时目录。
2. 仅在测试进程内把预处理函数的输出根目录和距离配置指向临时副本，重新生成图片与清单；未修改正式代码的默认路径。
3. 调用实际推理入口使用的 `prepare_inputs`，检查文件哈希、图片尺寸及官方 `load_images` 输出。
4. 再运行批量入口的 `--check-only`，验证场景发现和多图分组。

结果：3 张 RGB PNG，尺寸 518×392，官方加载器成功读取 3 个视图，批量状态为 `checked`。本次临时输入来自目前手动加过标记的原图，仅验证格式兼容；正式目录中原有干净 processed 图片没有被覆盖。

本次检查记录：`/private/tmp/data-branches-20261031-ey588qfi/phone_check/summary.csv`。临时目录可能被系统清理，日常复验请使用上面的命令生成新记录。
