# 共用工具

此目录预留给可直接运行的数据检查、可视化等工具，目前没有检查脚本。

数据来源专用的处理脚本放在 `../phone/` 或 `../graspnet/`。
共用的 Python 函数放在 `src/close_range/`。

目前可以通过批量预测入口仅检查输入，不加载模型：

```bash
bash scripts/local.sh scripts/predict_directory.py --input data/processed/phone --check-only
```
