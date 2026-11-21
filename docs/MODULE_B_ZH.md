# 改动 B：视角评分引导的全局注意力

## 实现位置

`src/close_range/models/view_fusion.py` 提供 `attach_view_fusion(model)`。必须先加载原预训练权重，再接入 B。适配当前 UniCeption 的 alternating-attention IFR 网络；不修改 site-packages，也不替换原有权重。像 A 一样使用 add_module 注册评分器，原网络 forward 显式调用；原参数的 state_dict 名称保持不变。

每个偶数编号的全局注意力层使用同一个 ViewScorer，奇数编号的帧内注意力保持原样。评分器从当前层归一化后的图像 patch 特征取平均，经 LayerNorm、Linear、GELU、Linear 输出每个视角的分数。每层重新计算，不跨 forward 缓存张量。

分数减去该场景各视角的平均值，表示相对视角偏好。每个视角的 patch 和属于该视角的 register token 使用同一分数；最后的全局 scale token 偏置为零。偏置形状为 B×1×1×N，沿 key 维度添加，避免显式构造 B×heads×N×N 的偏置。

普通注意力在 softmax 前加偏置；融合注意力使用浮点 attn_mask。本机 MPS 的融合 SDPA 暂不支持反向传播，因此在 MPS 接入 B 时，将全部信息融合层切换到等价的普通注意力（冻结的后续层仍需传递输入梯度）。CPU/CUDA 保留原选择。普通注意力会增加显存需求。保留原来的 Q/K 归一化、位置编码、缩放、dropout 和输出投影。评分器最后一层为零初始化。单视角时中心化评分为零，B 不调整单视角结果。

接入逻辑分为三处：

1. `src/close_range/models/view_fusion.py`：定义 ViewScorer，`attach_view_fusion` 将它注册为 `model.info_sharing.view_scorer`。
2. `third_party/uniception/uniception/models/info_sharing/alternating_attention_transformer.py`：在 IFR 全局注意力层中显式调用评分器，将分数展开为 key 偏置，再调用 `block(..., attention_bias=...)`。
3. `third_party/uniception/uniception/models/utils/transformer_blocks.py`：Block 把偏置传给 Attention，Attention 在 softmax 前加上偏置。

不再使用 ContextVar、动态类替换或浅拷贝适配器。项目随仓库保存 UniCeption 0.1.7 源码、许可证及两处修改的补丁；导入 close_range 自动优先使用它。已导入外部 UniCeption 的交互式会话需要重启。无需修改安装目录。

暂不支持梯度检查点和 torch.compile（未验证），保持原始 eager 配置。训练时不能用 no_grad 包裹 forward。原网络参数冻结不等于中间特征可以 detach，后续层必须保留向 B 反传的路径。

## 使用

```python
model = load_model(device)
attach_view_fusion(model, hidden_dim=128)
```

默认仅 B 可训练。A+B：先 attach_scale_correction，再 attach_view_fusion，然后显式将 model.scale_correction.requires_grad_(True)。B 接入会冻结已有参数，因此不能省略最后一步。

## 真实照片训练检查

在项目根目录运行，输出目录必须是新的：

```bash
bash scripts/local.sh scripts/tools/check_view_b_training.py \
  --manifest data/manifests/phone/scene-001.json \
  --steps 1 --output outputs/b_smoke
```

脚本要求至少三张图，选择参考图+第一张辅助图、参考图+第二张辅助图两组，使用参考图中同一对端点。支持 phone 和 GraspNet 标注。两组距离监督均使用对数距离平方损失；一致性项为两组预测对数距离之差的平方，默认权重 0.1。默认随机种子 0、AdamW 学习率 0.0005、weight decay 0、梯度范数裁剪 1。这是单场景链路检查，不是正式多场景训练器。

脚本检查零初始化距离等价、B 有限非零梯度、原网络无梯度；保存 training.json 和 modules.pt，重新载入模块参数，再对整组图执行预测及标准 mask 评估。训练检查使用原始点图的正且有限距离，不使用推理 mask；最终评估仍使用现有 mask 规则。损失不强制每一步下降，不能把短训练检查当作改善精度的证据。

A+B 检查增加 `--with-a`，A 和 B 均从恒等初始化开始。若要继续训练已有的 B/A+B，传 `--checkpoint`。

## 加载训练结果进行预测

```bash
bash scripts/local.sh scripts/tools/check_view_b_training.py \
  --manifest data/manifests/phone/scene-002.json \
  --checkpoint outputs/b_smoke/modules.pt \
  --steps 0 --output outputs/b_scene002
```

上述命令不更新参数，对全部输入视角预测，并评估标注点对。模型需先构建 B 适配器才能载入 B 权重；不支持直接用原始 MapAnything 加载 modules.pt。原通用 baseline 命令保持不变。

## 自动化测试

```bash
bash scripts/local.sh -m unittest discover -s tests -p 'test_view_fusion.py' -v
```

覆盖真实小型 UniCeption 网络的普通/融合注意力零初始化等价、原参数名字和值保留、仅全局层接收偏置、偏置 token 布局、非零梯度及参数更新、保存后 state_dict 恢复、两条注意力路径一致性、多场景分组、1/2/5 视角、固定参考图后辅助视角置换，以及一致性不能替代真实距离监督。

后续正式实验需混合 GraspNet 与 phone 的场景级训练/验证/测试划分、更多视角子集采样和 M0/M1/A/B/A+B 对照。评分是可学习偏置，未经验证不能解释为校准后的可靠性概率。
