"""Depth supervision losses for GeoViT guidance."""

from typing import Dict, Tuple

import torch


def scale_invariant_depth_loss(
    render_depth: torch.Tensor,
    target_depth: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.85,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute a scale-invariant log-depth loss.

    Args:
        render_depth: Predicted/rendered depth tensor.
        target_depth: Target depth tensor broadcastable to ``render_depth``.
        mask: Boolean or numeric validity mask broadcastable to the inputs.
        beta: Weight for the squared mean log-error term.
        eps: Minimum depth used before taking logarithms.

    Returns:
        A tuple of ``(loss, logs)``.  ``logs`` contains detached tensors for
        ``valid_ratio`` and ``mean_log_error``.
    """

    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    render_depth, target_depth, valid_mask = torch.broadcast_tensors(
        render_depth,
        target_depth,
        mask.bool(),
    )
    valid_mask = valid_mask & torch.isfinite(render_depth) & torch.isfinite(target_depth)
    valid_mask = valid_mask & (render_depth > eps) & (target_depth > eps)

    valid_ratio = valid_mask.to(dtype=render_depth.dtype).mean()
    if not torch.any(valid_mask):
        zero = render_depth.sum() * 0.0
        logs = {
            "valid_ratio": valid_ratio.detach(),
            "mean_log_error": zero.detach(),
        }
        return zero, logs

    log_error = torch.log(render_depth[valid_mask].clamp_min(eps)) - torch.log(
        target_depth[valid_mask].clamp_min(eps)
    )
    mean_log_error = log_error.mean()
    loss = (log_error.square().mean() - beta * mean_log_error.square()).clamp_min(0.0)

    logs = {
        "valid_ratio": valid_ratio.detach(),
        "mean_log_error": mean_log_error.detach(),
    }
    return loss, logs
