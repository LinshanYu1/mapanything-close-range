# A、B 模块与训练流程说明

本文对应当前项目的随机多子集训练实现。一个场景是一条训练样本，一个图片组是该场景的一个视角子集，一个视角是一张图片。测量端点在下文称为 P、Q，以免与模块 A、B 混淆。

## 1. 我们改了什么

| 模块 | 作用 | 接入位置 | 输出 |
|---|---|---|---|
| A：全局与局部尺度修正 | 利用全局尺度特征和局部特征修正整体尺度 | 原尺度头和 scale_adaptor 输出之后 | 每个 batch 样本一个正数尺度系数 |
| B：视角评分 | 为本次输入的每张图片学习相对偏好，调整跨视角融合 | info_sharing 的全局跨视角注意力层 | 每个视角一个注意力偏置 |

原模型已经具备跨视角注意力，并非看不到整体信息。B 的动机是给冻结的预训练模型增加一个小的、可训练的融合调整入口。它是否提高准确性和子集稳定性，需要实验验证；不能假设增加模块就一定有效。

```text
一组 RGB 图片
    ↓
原图像编码器
    ↓
info_sharing：交替执行跨视角注意力和视角内注意力
    └── B：在跨视角注意力的 softmax 前加入视角偏置
    ↓
融合后的特征
    ├── 原稠密几何、姿态预测
    └── 原尺度头 → 原尺度 → A 的系数修正
    ↓
原模型后续几何计算
    ↓
参考图三维点图 → 提取 P、Q 三维坐标 → 预测距离
```

以下形状中的 `Bsz` 表示 batch 大小，不是模块 B；`V` 是本次图片组的视角数，`P` 是每张图的 patch 数，`C` 是特征维度。

## 2. 模块 A 的接入与计算

实现文件：`src/close_range/models/scale_correction.py`。
调用位置：`third_party/map-anything/mapanything/models/mapanything/model.py`，搜索 `self.scale_correction`。

加载原始预训练权重后，`attach_scale_correction()` 将新模块注册为：

```python
model.scale_correction
```

`add_module()` 只注册子模块，不会自动调用它。原模型中增加了显式调用：

```python
correction = self.scale_correction(global_tokens, local_tokens)
scale_final_output = scale_final_output * correction
```

输入：

- `global_tokens`：原尺度预测头使用的特征，形状 `[Bsz, T, C]`。
- `local_tokens`：稠密预测头输入的局部特征，形状 `[V * Bsz, P, C]`。

局部特征会按照原模型的视角优先排列方式，重新整理为 `[Bsz, V * P, C]`。A 对全局 tokens 取平均并归一化，生成 Query；从局部 tokens 生成 Key 和 Value；用注意力加权汇总局部特征，再与全局投影特征拼接。小网络最终预测 `log_delta`：

```python
correction = log_delta.exp()          # [Bsz, 1]
new_scale = original_scale * correction
```

最后一层权重、偏置为零初始化，初始系数为 1，因此初始不改变原尺度。

同一次模型调用中，所有视角共享一个系数。一个场景拆成多组分别调用模型时，各组可以得到不同系数，但使用同一套 A 参数。A 做统一尺度修正，不能独立纠正每个物体的形状和位置。

## 3. 模块 B 的接入与计算

实现文件：`src/close_range/models/view_fusion.py`。
调用位置：`third_party/uniception/uniception/models/info_sharing/alternating_attention_transformer.py`，搜索 `self.view_scorer`。

`attach_view_fusion()` 将网络注册到原模型的信息融合网络内：

```python
model.info_sharing.view_scorer
```

当前 `info_sharing` 使用 `MultiViewAlternatingAttentionTransformerIFR`。偶数层执行全局跨视角注意力，奇数层执行视角内注意力。B 只在跨视角分支中调用；同一个评分网络在不同全局层共享参数，每层根据当前特征重新评分。

### 3.1 从特征得到分数

当前层先对 tokens 做 `block.norm1`，整理成各视角，再把 patch tokens 传给评分器，输入形状为 `[Bsz, V, P, C]`。评分器不使用额外 register tokens 来计算均值。

```python
scores = self.net(patches.mean(dim=2)).squeeze(-1)
return scores - scores.mean(dim=1, keepdim=True)
```

处理步骤：

1. 对每张图片的 patch 特征取平均：`[Bsz,V,P,C] → [Bsz,V,C]`。
2. 经过 `LayerNorm → Linear → GELU → Linear`：得到 `[Bsz,V,1]`。
3. `squeeze(-1)` 去掉最后的单元素维度：得到 `[Bsz,V]`。
4. 减去本组视角分数的均值，得到零均值的相对偏置。

它不是概率，也没有人工评分标签。最后线性层零初始化，初始所有偏置为零。

### 3.2 分数如何影响原模型

每个视角的分数复制给该视角的 patch 和 register tokens。额外全局 tokens 的偏置补零。整理为 `[Bsz,1,1,Key数量]`，广播给全部注意力头和 Query。

```text
S = QKᵀ / sqrt(每个注意力头的维度)
W = softmax(S + attention_bias)
O = WV
```

B 的输出加在注意力分数 S 上，不直接加到 token 特征，也不直接加到 O。偏置改变读取各视角信息的权重，新的融合特征继续传给后续网络，最终影响几何、尺度和测距。

B 每次只处理当前图片组。例如 `[R,X,Y]` 会给三张图分别评分，不会同时看到另一个组 `[R,Z]`。不同组共享评分网络参数，不共享评分结果。

## 4. 一个场景如何生成多组输入

代码：`src/close_range/training/dataset.py`。

Dataset 接收 split 中的 manifest 路径列表，初始化时检查图片 SHA-256、尺寸、标注参考图、坐标范围和距离有效性。它不负责划分数据，也不在训练时重新计算真实距离。

开启 `use_consistency` 后，固定 manifest 的参考图，枚举至少包含一张辅助图的所有组合，从中不放回随机抽取 `N−1` 组。组内参考图始终在第一位。

例如五张图 `R,X,Y,Z,W`：

- 每组两张：4 种。
- 每组三张：6 种。
- 每组四张：4 种。
- 每组五张：1 种。
- 总计 15 种，随机选 4 种，每组长度不固定。

```text
组1：[R,X]
组2：[R,Y,Z]
组3：[R,X,W]
组4：[R,X,Y,Z,W]
```

这是示例，不保证每次抽到全图组、覆盖所有照片，或每种长度都被抽到。每个候选组合等概率；不是先等概率选择长度。

训练使用“seed、epoch、来源、场景编号”生成稳定随机种子，组合可以随轮次变化；验证固定 epoch=0，保持组合可复现。

| 图片数 | 返回组数 | 行为 |
|---|---:|---|
| 1 | 1 | 只包含参考图，一致性损失为零；B-only 训练跳过单图场景 |
| 2 | 1 | 两张图一起输入，一致性损失为零 |
| 3 | 2 | 从两个双图组和一个三图组中抽两组 |
| 5 | 4 | 从 15 个候选中抽四组 |

关闭 `use_consistency` 时，只返回一个包含全部视角的组。它控制采样方式，与 loss 中的 `consistency_weight` 是两个设置；权重为零不自动关闭子集采样。

当前枚举候选的数量是 `2^(N−1)−1`，适合每场景几张图片；大量视角时需要改成不穷举的采样。

## 5. 从多组输入得到距离

代码：`src/close_range/training/trainer.py` 的 `predict_distances()` 和 `compute_loss()`。

```python
distances = torch.stack([
    self.predict_distances(views, sample["points_xy"])
    for views in sample["view_groups"]
])
```

每组调用同一个模型，得到参考图三维点图。代码将标注像素坐标四舍五入，索引两个三维点，并计算欧氏距离：

```python
predictions = self.model(views, ...)
xyz = predictions[0]["pts3d"][0]
a = xyz[y_P, x_P]
b = xyz[y_Q, x_Q]
distance = torch.linalg.vector_norm(a - b, dim=-1)
```

辅助图无需标注同一对端点：它们帮助模型预测参考图的几何，测距始终在参考图上进行。该方法不等于显式验证辅助图上的像素对应关系。

设子集数为 G、标注点对数为 M：

```text
distances：[G, M]，每组对每个点对的预测，单位米。
target：   [M]，标注中的真实距离，单位米。
```

真实距离只用于 loss 和评分，不输入 A、B，也不用于对预测尺度做真值对齐。不是先平均各组距离再监督。

## 6. loss 如何计算

代码：`src/close_range/training/losses.py`。当前是对数距离平方损失，不是 Huber 稳健损失。

### 6.1 监督损失

所有子集的所有点对分别与真实距离比较，再取平均：

```python
log_distances = distances.log()
log_target = target.log()
supervised = (log_distances - log_target).square().mean()
```

即平均 `log(预测距离 / 真实距离)` 的平方。距离必须为有限正数。

### 6.2 子集一致性损失

每个测量点对，在不同子集之间比较对数距离。遍历全部无序子集配对，只计算一次，不与自身比较：

```python
i, j = torch.triu_indices(G, G, offset=1, device=distances.device)
consistency = (log_distances[i] - log_distances[j]).square().mean()
```

G=4 时有 6 对比较：1–2、1–3、1–4、2–3、2–4、3–4。G=1 时一致性为零。使用平均值，避免损失仅因组数变多而机械增大。

### 6.3 总损失

```python
loss = supervised + consistency_weight * consistency
```

命令行默认 A 的一致性权重为 0，B/AB 为 0.1；显式参数可以覆盖。A 默认仍使用多子集监督，只是不加一致性项。

准确性和一致性都需要关注：所有组都预测错但错得相同，也可以得到很小的一致性损失。

日志里的 `subset_gap_cm` 是所有组两两预测距离绝对差的平均值（厘米），用于解释稳定性，不是直接加入 loss 的量。

## 7. 梯度如何更新

当前训练器逐场景训练，相当于场景级 batch_size=1，没有跨场景梯度累积。一个场景包含多组输入，不代表多场景 batch。

```python
self.optimizer.zero_grad(set_to_none=True)
loss, metrics = self.compute_loss(sample)
loss.backward()
torch.nn.utils.clip_grad_norm_(
    self.parameters, max_norm=1.0, error_if_nonfinite=True
)
self.optimizer.step()
```

- `zero_grad`：清空上一场景的梯度。
- 多组前向：每组计算期间不更新参数；计算图保留到总 loss 反传。
- `backward`：计算总 loss 对参数的梯度，同一参数在各组计算中的贡献自动累加。
- 梯度裁剪：将可训练参数的梯度总范数限制到不超过 1，非有限梯度报错。
- `step`：AdamW 根据梯度更新参数，当前 weight_decay=0。

```text
所有子集的总 loss
    ↓
各组距离 → 三维点图
    ├── 修正尺度 → A 参数
    └── 原模型后续运算 → 注意力偏置 → B 参数
```

冻结原模型表示其参数不更新，不表示截断经过原模型的梯度。尤其 B 位于中间，loss 必须通过后面的网络运算才能传回 B。训练调用 `model(...)`，不能整体包在 `no_grad()` 或 `inference_mode()` 中，也不能用禁用梯度的 `infer()` 替代。

当前使用 `model.eval()` 关闭 dropout 等训练模式行为，但它不关闭 autograd。指标计算中的 `detach()` 只作用于日志副本，返回的总 loss 保留计算图。

| 训练变体 | 更新的参数 |
|---|---|
| A | scale_correction |
| B | info_sharing.view_scorer |
| AB | 两者；挂载 B 冻结参数后，会重新启用 A 的梯度 |

零初始化最后线性层时，初始步骤中前面某些新增层可能得到零梯度；这不等于整个模块没有训练，最后一层先更新后，梯度可以传到更前面的层。

## 8. 多个场景、多个 epoch 的执行顺序

```text
加载一次预训练模型，挂载 A/B
    ↓
每个 epoch 打乱场景顺序
    ↓
场景1：随机多组 → 多次前向 → 一个 loss → 更新一次
场景2：随机多组 → 多次前向 → 一个 loss → 更新一次
……
    ↓
验证集：固定组合，计算指标，不更新参数
    ↓
保存参数、配置和每轮指标
    ↓
下一轮继续使用已经更新的参数
```

场景之间共享同一模型；不会换场景就重新初始化。若所有场景均参与训练，94 个训练场景每轮更新 94 次。

同时保留更多组、更长图片组的计算图会增加显存和时间。当前没有自动微批处理或梯度累积，不应把这种实现描述成多场景批量训练。

## 9. 如何启动和保存

先安装依赖，准备 processed、annotations、manifests，生成固定划分。以下命令在项目根目录执行，CUDA 用于 Colab/NVIDIA，Mac 使用 mps：

```bash
python src/close_range/training/main.py \
  --variant AB \
  --split configs/mixed_split.json \
  --output-dir checkpoints/mixed_AB_new \
  --device cuda \
  --epochs 50 \
  --learning-rate 0.0005 \
  --consistency-weight 0.1 \
  --seed 443
```

训练 A/B 时修改 variant 和输出目录；A 若不使用一致性，显式设置权重为 0。请确保当前分支的划分器和 Dataset 支持所用数据来源。

不传 `--init-from` 时，加载原预训练权重并初始化新增模块。传入同变体的历史模型目录可以用已有模块参数热启动，优化器重新创建。A、B、AB 独立训练时，AB 不会自动加载之前分别训练的 A、B。

每轮输出：

- `model.pt`：最近一轮模块参数。
- `model_best.pt`：验证平均相对误差改善且无无效验证场景时保存。
- `model_config.json`：模型变体、基础模型版本等配置。
- `training_config.json`：训练设置和数据描述。
- `metrics/epoch_*.json`：每轮训练与验证指标。
- `checkpoints/latest.pt`、`best.pt`：用于完整恢复的检查点（含优化器等状态）。

推理权重主要保存新增模块，不重复保存冻结的完整基础模型。加载时仍需对应的预训练版本。Colab 若将 output-dir 指向已挂载 Drive 的目录，结果即保存到 Drive；当前未结束的一轮不保证保存。

随机子集规则已从旧版固定两组变更，旧指标不能不加说明地直接对比，新训练使用新输出目录。旧测试中的 views_1/views_2 和旧损失接口还需同步；小型张量测试通过不等于真实模型多组训练已验证。

## 10. 最终评估

使用本次训练划分中的 test，加载验证集选择的参数。原模型、A、B、AB 使用相同测试数据、端点和输入设置。

- 全部图片作为一组输入：评估实际多图测距准确性。
- 多个固定子集输入：评估子集稳定性，并同时报告准确性和无效预测。

训练器已有 subset_gap_cm；当前独立批量评估脚本主要比较单图/全部多图，尚未实现专门的多子集稳定性评估。不要把单图/多图比较直接等同于子集稳定性评估。
