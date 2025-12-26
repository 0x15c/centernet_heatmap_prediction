# losses.py
from __future__ import annotations
import torch


def centernet_focal_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    CenterNet-style focal loss for heatmaps.

    pred_logits: (N, C, H, W) raw logits
    target:      (N, C, H, W) target heatmaps in [0, 1] with peaks at 1
    """
    pred = torch.sigmoid(pred_logits)

    pos_mask = (target == 1).to(pred.dtype)
    neg_mask = (target < 1).to(pred.dtype)

    # Positive
    pos_loss = -torch.log(pred.clamp(min=eps)) * \
        ((1.0 - pred) ** alpha) * pos_mask

    # Negative (down-weight near positives by (1 - target)^beta)
    neg_weight = ((1.0 - target) ** beta) * neg_mask
    neg_loss = -torch.log((1.0 - pred).clamp(min=eps)) * \
        (pred ** alpha) * neg_weight

    loss = pos_loss.sum() + neg_loss.sum()
    num_pos = pos_mask.sum().clamp(min=1.0)  # avoid divide by 0
    return loss / num_pos


def heatmap_sparsity_loss(pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Penalize overall activation to reduce false positives.
    Uses mean probability over all pixels/channels.
    """
    pred = torch.sigmoid(pred_logits)
    return pred.mean()


def total_variation_loss(pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Spatial smoothness regularizer (TV loss) to reduce noisy heatmaps.
    """
    pred = torch.sigmoid(pred_logits)
    tv_h = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs().mean()
    tv_w = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


def combined_heatmap_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    lambda_sparse: float = 0.05,
    lambda_tv: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Total loss = focal + lambda_sparse*sparsity + lambda_tv*TV
    """
    focal = centernet_focal_loss(pred_logits, target)
    sparse = heatmap_sparsity_loss(
        pred_logits) if lambda_sparse > 0 else pred_logits.new_tensor(0.0)
    tv = total_variation_loss(
        pred_logits) if lambda_tv > 0 else pred_logits.new_tensor(0.0)
    total = focal + lambda_sparse * sparse + lambda_tv * tv
    logs = {
        "loss_focal": float(focal.detach().cpu()),
        "loss_sparse": float(sparse.detach().cpu()),
        "loss_tv": float(tv.detach().cpu()),
        "loss_total": float(total.detach().cpu()),
    }
    return total, logs
