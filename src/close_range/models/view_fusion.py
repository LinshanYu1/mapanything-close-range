"""Module B: a view scorer explicitly called by the information-sharing network."""
import torch
from torch import nn
from uniception.models.info_sharing.alternating_attention_transformer import (
    MultiViewAlternatingAttentionTransformerIFR,
)


class ViewScorer(nn.Module):
    """Shared across views and global layers; starts with exactly zero logits."""

    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError('Feature dimensions must be positive')
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, patches):
        scores = self.net(patches.mean(dim=2)).squeeze(-1)  # B,V
        # Relative view preference; no arbitrary common shift against scale tokens.
        return scores - scores.mean(dim=1, keepdim=True)


def attach_view_fusion(model, hidden_dim=128, freeze_backbone=True):
    """Attach B after loading pretrained weights, just like Module A.

    The project-local UniCeption forward explicitly calls view_scorer.
    For A+B, re-enable A parameters after this function freezes the backbone.
    """
    sharing = model.info_sharing
    if hasattr(sharing, "view_scorer"):
        raise ValueError("Module B is already attached")
    if type(sharing) is not MultiViewAlternatingAttentionTransformerIFR or not getattr(
        sharing, "supports_view_bias", False
    ):
        raise ValueError("B requires the project's patched UniCeption IFR implementation")
    if getattr(sharing, "gradient_checkpointing", False):
        raise ValueError("Disable gradient checkpointing for Module B")
    if freeze_backbone:
        model.requires_grad_(False)
    reference = next(sharing.parameters())
    scorer = ViewScorer(sharing.dim, hidden_dim).to(reference)
    scorer.train(sharing.training)
    sharing.add_module("view_scorer", scorer)
    sharing.view_fusion_enabled = True
    # Current MPS SDPA cannot backpropagate; downstream frozen layers also need
    # input gradients so that the distance loss can reach the scorer.
    if reference.device.type == "mps":
        for block in sharing.self_attention_blocks:
            block.attn.fused_attn = False
    return model


def distance_training_loss(distance_1, distance_2, target, consistency_weight=0.1):
    """Equal supervised weights plus log-distance consistency; positive distances only."""
    if consistency_weight < 0:
        raise ValueError('Consistency weight must be nonnegative')
    for value in (distance_1, distance_2, target):
        if not torch.isfinite(value).all() or (value <= 0).any():
            raise ValueError('Distances must be finite and positive')
    if distance_1.shape != distance_2.shape or target.shape != distance_1.shape:
        raise ValueError('Distance shapes must match; implicit broadcasting is not allowed')
    log1, log2, truth = distance_1.log(), distance_2.log(), target.log()
    supervised = 0.5 * ((log1 - truth).square() + (log2 - truth).square()).mean()
    consistency = (log1 - log2).square().mean()
    return supervised + consistency_weight * consistency, supervised, consistency
