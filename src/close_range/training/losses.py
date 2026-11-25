"""以米为单位的点对距离监督与跨视角一致性。"""
import math
import torch


def distance_loss(distances, target, consistency_weight=0.1):
    """
    distances：[子集数量, 标注点对数量]
    target：   [标注点对数量]
    """
    if not math.isfinite(consistency_weight) or consistency_weight < 0:
        raise ValueError("consistency_weight 必须为有限非负数")

    if (
        distances.ndim != 2
        or target.ndim != 1
        or distances.shape[0] < 1
        or target.numel() == 0
        or distances.shape[1] != target.shape[0]
    ):
        raise ValueError("预测距离或真实距离的形状不正确")

    for value in (distances, target):
        if not torch.isfinite(value).all() or (value <= 0).any():
            raise ValueError("距离必须为有限正数")

    log_distances = distances.log()
    log_target = target.log()

    # 每个子集都与真实距离比较，再统一取平均。
    supervised = (log_distances - log_target).square().mean()

    consistency = supervised.new_zeros(())
    num_groups = distances.shape[0]

    if num_groups > 1:
        # 只选择不同子集的配对，不重复、不和自己比较。
        i, j = torch.triu_indices(
            num_groups,
            num_groups,
            offset=1,
            device=distances.device,
        )

        consistency = (
            log_distances[i] - log_distances[j]
        ).square().mean()

    return {
        "loss": supervised + consistency_weight * consistency,
        "supervised": supervised,
        "consistency": consistency,
    }
