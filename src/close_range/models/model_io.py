"""保存和加载 MapAnything 的 A/B 扩展，不重复保存冻结的基础模型。"""
import json
import os
from pathlib import Path

import torch

from close_range.common import REVISION
from .scale_correction import attach_scale_correction
from .view_fusion import attach_view_fusion


def get_model_config(model, revision=REVISION):
    a = getattr(model, "scale_correction", None)
    b = getattr(model.info_sharing, "view_scorer", None)
    if a is None and b is None:
        raise ValueError("模型没有 A/B 模块")
    return {
        "format_version": 1,
        "architecture": "mapanything_close_range_v1",
        "base_model": "facebook/map-anything",
        "revision": revision,
        "variant": "AB" if a is not None and b is not None else "A" if a is not None else "B",
        "feature_dim": model.info_sharing.dim,
        "a_hidden_dim": None if a is None else a.hidden_dim,
        "b_hidden_dim": None if b is None else b.hidden_dim,
    }


def _modules(model):
    result = {}
    if hasattr(model, "scale_correction"):
        result["scale_correction"] = model.scale_correction
    if hasattr(model.info_sharing, "view_scorer"):
        result["info_sharing.view_scorer"] = model.info_sharing.view_scorer
    return result


def _validate_config(config):
    if config.get("format_version") != 1 or config.get("architecture") != "mapanything_close_range_v1":
        raise ValueError("不支持的模型配置版本")
    if config.get("base_model") != "facebook/map-anything" or not config.get("revision"):
        raise ValueError("必须指定基础模型及其版本")
    if config.get("variant") not in {"A", "B", "AB"}:
        raise ValueError("variant 必须为 A、B 或 AB")
    for key, enabled in (("feature_dim", True),
                         ("a_hidden_dim", config["variant"] in {"A", "AB"}),
                         ("b_hidden_dim", config["variant"] in {"B", "AB"})):
        value = config.get(key)
        if enabled and (type(value) is not int or value <= 0):
            raise ValueError(f"{key} 必须为正整数")
        if not enabled and value is not None:
            raise ValueError(f"未启用模块的 {key} 必须为 None")


def save_model(model, save_dir, model_config=None, weights_name="model.pt", *, epoch=None, global_step=None):
    """按模块 state_dict 导出，不受 requires_grad 状态影响。"""
    config = get_model_config(model) if model_config is None else dict(model_config)
    _validate_config(config)
    if config != get_model_config(model, config["revision"]):
        raise ValueError("模型结构与配置不一致")
    directory = Path(save_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if Path(weights_name).name != weights_name:
        raise ValueError("weights_name 必须为文件名")
    config_path = directory / "model_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("目录内已有不同模型配置，请使用新目录")
    parameters = {
        f"{prefix}.{name}": tensor.detach().cpu().clone()
        for prefix, module in _modules(model).items()
        for name, tensor in module.state_dict().items()
    }
    state = dict(format="close_range_model_v1", model_config=config,
                 parameters=parameters, epoch=epoch, global_step=global_step)
    temporary = directory / (weights_name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, directory / weights_name)
    temporary = directory / "model_config.json.tmp"
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, config_path)


def load_module_parameters(model, save_dir, weights_name="model_best.pt"):
    """恢复已构建模型的模块权重；严格检查名称、形状、类型及有限性。"""
    directory = Path(save_dir)
    config = json.loads((directory / "model_config.json").read_text())
    _validate_config(config)
    if config != get_model_config(model, config["revision"]):
        raise ValueError("目标模型结构与保存配置不一致")
    state = torch.load(directory / weights_name, map_location="cpu", weights_only=True)
    if state.get("format") != "close_range_model_v1" or state.get("model_config") != config:
        raise ValueError("权重文件与模型配置不匹配")
    modules = _modules(model)
    expected = {f"{prefix}.{name}": value for prefix, module in modules.items()
                for name, value in module.state_dict().items()}
    parameters = state["parameters"]
    if parameters.keys() != expected.keys():
        raise ValueError("模块参数缺失或存在多余参数")
    for name, value in parameters.items():
        if not torch.is_tensor(value) or value.shape != expected[name].shape:
            raise ValueError(f"参数形状错误：{name}")
        if value.dtype != expected[name].dtype or not torch.isfinite(value).all():
            raise ValueError(f"参数类型或数值错误：{name}")
    for prefix, module in modules.items():
        module.load_state_dict({name[len(prefix) + 1:]: value
                                for name, value in parameters.items()
                                if name.startswith(prefix + ".")}, strict=True)
    return state


def _load_base_model(config):
    from mapanything.models import MapAnything
    return MapAnything.from_pretrained(config["base_model"], revision=config["revision"])


def load_model(save_dir, device="mps", weights_name="model_best.pt"):
    """读取目录，加载基础权重、安装 A/B 并恢复模块；返回用于预测的模型。"""
    directory = Path(save_dir)
    config = json.loads((directory / "model_config.json").read_text())
    _validate_config(config)
    if not (directory / weights_name).is_file():
        raise FileNotFoundError(directory / weights_name)
    model = _load_base_model(config).to(device)
    if model.info_sharing.dim != config["feature_dim"]:
        raise ValueError("基础模型特征维度与保存配置不同")
    if config["variant"] in {"A", "AB"}:
        attach_scale_correction(model, hidden_dim=config["a_hidden_dim"])
    if config["variant"] in {"B", "AB"}:
        attach_view_fusion(model, hidden_dim=config["b_hidden_dim"])
    load_module_parameters(model, directory, weights_name)
    model.requires_grad_(False)
    model.eval()
    return model
