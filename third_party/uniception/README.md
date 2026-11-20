# UniCeption 0.1.7 项目内源码

来源：本项目原运行环境安装的 PyPI `uniception==0.1.7`。保留上游源码及许可证；未修改 site-packages。仅两个文件加入 B 的显式接口，完整差异见 `module_b.patch`。

- `alternating_attention_transformer.py`：IFR 的全局层显式调用 `view_scorer`，将 `attention_bias` 传入 Block。
- `transformer_blocks.py`：Block 将可选偏置传给 Attention，Attention 在 softmax 前加偏置（融合路径使用 attn_mask）。默认 None 保持原计算。

项目导入 `close_range` 时优先加载本目录；直接使用第三方 MapAnything 时，可将此目录放在 PYTHONPATH 最前面。若会话已加载外部 UniCeption，请重启 Python。其余依赖仍按原 MapAnything 环境安装。

升级依赖时须重新核对补丁并运行测试。保留了预训练参数名字和形状，现有 B / A+B 模块权重可继续加载。梯度检查点及 torch.compile 尚未验证，接入函数仍拒绝梯度检查点配置。
