"""A/B 模块、优化器及轮次状态的原子保存；在完整 epoch 边界恢复。"""
import os
import json
import random
from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "close_range_training_v1",
        "parameters": {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        },
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        **metadata,
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer, expected_config):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "close_range_training_v1":
        raise ValueError("不是正式训练 checkpoint；旧 modules.pt 不包含完整训练状态")
    if state["config"] != expected_config:
        raise ValueError("恢复训练的配置或数据清单不一致")
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if parameters.keys() != state["parameters"].keys():
        raise ValueError("checkpoint 的可训练参数与当前模型不匹配")
    for name, parameter in parameters.items():
        if parameter.shape != state["parameters"][name].shape:
            raise ValueError(f"参数形状不匹配：{name}")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state["parameters"][name])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    if "cuda_rng" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state


def save_training_config(path, config):
    """保存可直接阅读的训练配置，与二进制权重分开。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)
