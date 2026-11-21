"""以米为单位的点对距离监督与跨视角一致性。"""
import torch


def distance_loss(distance_1, target, distance_2=None, consistency_weight=0.1):
    if consistency_weight < 0:
        raise ValueError("consistency_weight 必须非负")
    values = [distance_1, target]
    if distance_2 is not None:
        values.append(distance_2)
    for value in values:
        if value.shape != target.shape or value.numel() == 0:
            raise ValueError("距离形状必须一致且非空")
        if not torch.isfinite(value).all() or (value <= 0).any():
            raise ValueError("距离必须为有限正数")
    log1, truth = distance_1.log(), target.log()
    supervised = (log1 - truth).square().mean()
    consistency = supervised.new_zeros(())
    if distance_2 is not None:
        log2 = distance_2.log()
        supervised = 0.5 * (supervised + (log2 - truth).square().mean())
        consistency = (log1 - log2).square().mean()
    return {
        "loss": supervised + consistency_weight * consistency,
        "supervised": supervised,
        "consistency": consistency,
    }
