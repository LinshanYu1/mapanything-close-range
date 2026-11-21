# 改动 A：全局与局部特征结合的尺度修正

## 实现位置

- `src/close_range/models/scale_correction.py`：注意力汇总局部特征、输出场景修正系数、安装模块。
- `third_party/map-anything/mapanything/models/mapanything/model.py`：`downstream_head` 中增加可选接入点。
- `tests/test_scale_correction.py`：初始化、梯度、批次隔离及真实 downstream_head 接入测试。

先正常加载预训练参数，再调用 `attach_scale_correction(model)`。没有安装 A 时，原模型保持原有路径。安装不会覆盖原尺度头参数；新增参数位于 `scale_correction.*`。

## 信息如何流动

1. 读取信息融合网络输出的尺度 token：原形状 `(B, C, T)`，接入点转为 `(B, T, C)`，沿 token 维取平均作为全局特征。
2. 读取最后一层局部 patch 特征：上游按视角拼接为 `(V*B, C, H, W)`，接入点展平并转为 `(V*B, P, C)`，重排为每个场景一组 `(B, V*P, C)`。这些是已经融合多视角信息的 patch 特征，并非原始像素。
3. 用全局特征产生 query，局部特征产生 key/value；softmax 注意力对所有视角的 patch 做加权汇总。
4. 拼接全局投影和局部汇总，经小型网络输出一个 `log_delta`。
5. `correction = exp(log_delta)`，`corrected_scale = original_scale * correction`，每个场景只输出一个系数 `(B,1)`。
6. 让原有代码继续将新尺度用于点云、深度和相机平移。方向、旋转不应随尺度改变。

最后一层权重、偏置均初始化为 0，因此初始修正系数恰为 1。这是整体尺度修正，不是局部形状修复。注意力权重是训练得到的信息权重，不直接等于物体可靠性。

## 调用方式

在 `scripts/local.sh` 配置的 Python 环境下：

```python
from mapanything.models import MapAnything
from close_range.common import REVISION
from close_range.models import attach_scale_correction

model = MapAnything.from_pretrained("facebook/map-anything", revision=REVISION)
model = model.to(device)
attach_scale_correction(model, hidden_dim=128, freeze_backbone=True)

# 第一阶段只训练 A。保持冻结骨干的确定性行为。
model.eval()
model.scale_correction.train()
optimizer = torch.optim.AdamW(model.scale_correction.parameters(), lr=1e-4)

# views 需遵循上游训练 forward 的输入约定，张量位于相同 device。
# 训练不能调用禁用梯度的 infer，也不能包在 inference_mode/no_grad 中。
predictions = model(views)
```

上述片段还需 `import torch`，并由训练程序提供 device、views、标签及 loss。此改动没有包含完整训练器。

默认冻结旧模型所有参数，新模块参数可训练。零初始化意味着第一次反向传播主要更新最后一层；它离开零后，前面的注意力和特征融合层才收到有效梯度，这是预期行为。

保存 A 参数：

```python
torch.save({"hidden_dim": 128, "state_dict": model.scale_correction.state_dict()}, "scale_a.pt")
```

恢复时先加载相同版本原模型、安装相同 hidden_dim 的 A，然后 `model.scale_correction.load_state_dict(...)`。既有 baseline 命令默认仍使用原模型，没有自动开启 A。

## 验证

```bash
bash scripts/local.sh -m unittest discover -s tests -v
```

2026-11-08：全部 9 项测试通过，其中 A 新增 4 项：

- 初始系数严格等于 1；一步优化能降低构造的尺度损失，后续注意力和全局投影能收到梯度。
- 多场景合批与分别处理得到一致结果，不混淆 view-major 布局；重排局部视角不改变汇总结果。
- 调用真实 `MapAnything.downstream_head`（轻量替代 dense/scale heads）：初始尺度一致，设置 log(2) 偏置后尺度翻倍并保留梯度。
- 安装 A 保留原参数、冻结骨干、仅新增模块可训练，重复安装报错。

尚未进行数据集训练，因此不能据此声称测距精度提高。

真实预训练模型检查（本机 MPS，scene-001 单图，关闭 AMP）：加载缓存权重后分别运行原模型、初始 A、强制 2 倍系数的 A。初始 A 的全部张量输出与原模型逐元素完全一致；`pts3d`、`pts3d_cam`、`depth_along_ray`、`metric_scaling_factor` 的 2 倍关系通过检查。当前配置新增 826,113 个参数。相机平移沿用上游同一尺度应用路径，本次没有单独数值断言平移。该检查不等于训练后的精度评估。

## 真实场景训练链路检查（2026-11-10）

复验命令：

```bash
bash scripts/local.sh scripts/tools/check_scale_a_training.py --steps 5
```

默认使用 phone/scene-001 的参考图及第一对有效标注，原模型冻结，只训练 A；每一步都调用允许梯度的完整模型 forward，不使用 infer。使用对数距离平方误差，AdamW 学习率 0.0005，并检查标签哈希、初始恒等、梯度有限且非零、骨干无梯度、最后损失下降。结果保存到新的 outputs/scale_a_smoke_<时间>/result.json，不保存或覆盖模型权重。

本次 17 项现有测试全部通过。真实样本参考距离 43.18 cm，5 次更新结果：

| 更新次数 | 预测距离 cm | 修正系数 | 损失 |
|---|---:|---:|---:|
| 0 | 71.5670 | 1.0000 | 0.255285 |
| 1 | 71.0512 | 0.9928 | 0.248027 |
| 2 | 69.4016 | 0.9697 | 0.225180 |
| 3 | 66.2307 | 0.9254 | 0.182985 |
| 4 | 61.3113 | 0.8567 | 0.122912 |
| 5 | 54.7186 | 0.7646 | 0.056087 |

报告：outputs/scale_a_smoke_20261110_172810/result.json。

这是单场景、单点对的训练拟合检查，不是独立测试集上的精度提升结果。检查使用原始预测点坐标，未做推理掩码过滤，也未验证多场景训练或 B 模块。
