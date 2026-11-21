"""Module A: one positive global-local scale correction per scene."""
import torch
from torch import nn


class GlobalLocalScaleCorrection(nn.Module):
    """Pool patch tokens with a scale-token query; predict residual log scale.

    Upstream stacks patch features in view-major order: (V * B, P, C).
    Scale tokens are (B, T, C). A single factor is shared by all views.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Feature dimensions must be positive")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.global_norm = nn.LayerNorm(feature_dim)
        self.local_norm = nn.LayerNorm(feature_dim)
        self.query = nn.Linear(feature_dim, hidden_dim)
        self.key = nn.Linear(feature_dim, hidden_dim)
        self.value = nn.Linear(feature_dim, hidden_dim)
        self.global_projection = nn.Linear(feature_dim, hidden_dim)
        self.fusion = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU())
        self.log_scale_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.log_scale_head.weight)
        nn.init.zeros_(self.log_scale_head.bias)

    def forward(self, global_tokens, local_tokens):
        if global_tokens.ndim != 3 or local_tokens.ndim != 3:
            raise ValueError("Expected scale (B,T,C) and local (V*B,P,C) tokens")
        batch, _, channels = global_tokens.shape
        if batch == 0 or channels != self.feature_dim or local_tokens.shape[-1] != channels:
            raise ValueError("Feature dimension/batch mismatch")
        if local_tokens.shape[0] % batch or local_tokens.shape[0] == 0 or local_tokens.shape[1] == 0:
            raise ValueError("Local tokens must contain complete nonempty view batches")
        # Respect upstream torch.cat(view_features, dim=0), not batch-major order.
        local = local_tokens.reshape(-1, batch, local_tokens.shape[1], channels)
        local = local.permute(1, 0, 2, 3).reshape(batch, -1, channels)
        global_feature = self.global_norm(global_tokens.mean(dim=1))
        local_feature = self.local_norm(local)
        query = self.query(global_feature).unsqueeze(1)
        keys = self.key(local_feature)
        weights = (query * keys).sum(dim=-1) / self.hidden_dim ** 0.5
        weights = weights.softmax(dim=-1)
        summary = (weights.unsqueeze(-1) * self.value(local_feature)).sum(dim=1)
        fused = torch.cat([self.global_projection(global_feature), summary], dim=-1)
        log_delta = self.log_scale_head(self.fusion(fused))
        return log_delta.exp()  # (B, 1); exactly 1 at initialization


def attach_scale_correction(model, hidden_dim=128, freeze_backbone=True):
    """Attach A AFTER loading original pretrained weights. Returns the same model.

    Training uses model(views), not the no-grad model.infer interface.
    Frozen backbone parameters still allow gradients through A; do not wrap the
    training forward in no_grad/inference_mode.
    """
    if hasattr(model, "scale_correction"):
        raise ValueError("Module A is already attached")
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    reference = next(model.scale_head.parameters())
    correction = GlobalLocalScaleCorrection(model.info_sharing.dim, hidden_dim)
    correction.to(device=reference.device, dtype=reference.dtype)
    correction.train(model.training)
    model.add_module("scale_correction", correction)
    return model
