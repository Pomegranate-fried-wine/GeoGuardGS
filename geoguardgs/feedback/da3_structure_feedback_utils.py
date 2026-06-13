import torch
import torch.nn.functional as F

from lib.geovit.depth_anything_bridge import GeoViTDepthBridge


def make_da3_bridge(cfg_node):
    model_name = getattr(cfg_node, "model_dir", "") or getattr(cfg_node, "model_name", None)
    inference_kwargs = {
        "process_res": int(getattr(cfg_node, "process_res", 128)),
        "process_res_method": str(getattr(cfg_node, "process_res_method", "upper_bound_resize")),
    }
    return GeoViTDepthBridge(
        model_name=model_name,
        device=getattr(cfg_node, "device", "cuda"),
        use_cache=bool(getattr(cfg_node, "use_cache", True)),
        detach_cache=bool(getattr(cfg_node, "detach_cache", True)),
        include_confidence=bool(getattr(cfg_node, "include_confidence", True)),
        include_tokens=bool(getattr(cfg_node, "include_tokens", False)),
        local_files_only=bool(getattr(cfg_node, "local_files_only", True)),
        inference_kwargs=inference_kwargs,
    )


def _normalize_depth(depth, mask=None, eps=1e-6):
    if mask is None:
        mask = torch.isfinite(depth)
    mask = mask & torch.isfinite(depth)
    if not torch.any(mask):
        return depth * 0.0
    vals = depth[mask]
    lo = torch.quantile(vals.float(), 0.05)
    hi = torch.quantile(vals.float(), 0.95)
    return ((depth - lo) / (hi - lo).clamp_min(eps)).clamp(0.0, 1.0)


def _gradient_mag(depth):
    dx = F.pad(depth[..., :, 1:] - depth[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(depth[..., 1:, :] - depth[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-8)


def da3_structure_loss(
    rendered_depth,
    rendered_acc,
    da3_depth,
    feedback_weight,
    cfg_node,
):
    """Structure-only DA3 feedback on selected reliable boundary-risk pixels.

    This deliberately avoids pointwise DA3 depth fitting. The losses only ask
    rendered depth to preserve DA3 boundary strength and local relative order.
    """

    if feedback_weight is None:
        zero = rendered_depth.sum() * 0.0
        return zero, {"da3_structure_valid_pixels": 0}

    if da3_depth.ndim == 2:
        da3_depth = da3_depth.unsqueeze(0).unsqueeze(0)
    elif da3_depth.ndim == 3:
        da3_depth = da3_depth.unsqueeze(0) if da3_depth.shape[0] != 1 else da3_depth.unsqueeze(0)
    if rendered_depth.ndim == 3:
        rendered_depth = rendered_depth.unsqueeze(0)
    if rendered_acc.ndim == 3:
        rendered_acc = rendered_acc.unsqueeze(0)
    if feedback_weight.ndim == 3:
        feedback_weight = feedback_weight.unsqueeze(0)

    da3_depth = F.interpolate(da3_depth.float(), size=rendered_depth.shape[-2:], mode="bilinear", align_corners=False)
    feedback_weight = F.interpolate(
        feedback_weight.float(), size=rendered_depth.shape[-2:], mode="nearest"
    )

    base = feedback_weight > 1.0
    valid = base & torch.isfinite(rendered_depth) & torch.isfinite(da3_depth) & (rendered_acc > 0.03)
    valid_count = int(torch.count_nonzero(valid).item())
    if valid_count == 0:
        zero = rendered_depth.sum() * 0.0
        return zero, {"da3_structure_valid_pixels": 0}

    rd = _normalize_depth(rendered_depth / (rendered_acc + 1e-6), valid)
    dd = _normalize_depth(da3_depth, valid)
    rg = _gradient_mag(rd)
    dg = _gradient_mag(dd)

    weight = feedback_weight.clamp_min(0.0)
    edge_margin = float(getattr(cfg_node, "da3_edge_margin", 0.05))
    edge_loss = (F.relu(dg - rg - edge_margin) * weight)[valid].mean()

    # Local relative ranking: preserve DA3 near/far sign across right/down pairs
    ranking_losses = []
    ranking_margin = float(getattr(cfg_node, "da3_ranking_margin", 0.02))
    for shift_y, shift_x in [(0, 1), (1, 0)]:
        rd_a = rd[..., : rd.shape[-2] - shift_y or None, : rd.shape[-1] - shift_x or None]
        rd_b = rd[..., shift_y:, shift_x:]
        dd_a = dd[..., : dd.shape[-2] - shift_y or None, : dd.shape[-1] - shift_x or None]
        dd_b = dd[..., shift_y:, shift_x:]
        vw = weight[..., : weight.shape[-2] - shift_y or None, : weight.shape[-1] - shift_x or None]
        vm = valid[..., : valid.shape[-2] - shift_y or None, : valid.shape[-1] - shift_x or None]
        da3_delta = dd_b - dd_a
        render_delta = rd_b - rd_a
        confident = vm & (torch.abs(da3_delta) > ranking_margin)
        if torch.any(confident):
            ranking_losses.append((F.relu(ranking_margin - torch.sign(da3_delta) * render_delta) * vw)[confident].mean())
    ranking_loss = sum(ranking_losses) / max(len(ranking_losses), 1) if ranking_losses else edge_loss * 0.0

    # Boundary side consistency: DA3 edge high should not become rendered smooth.
    side_loss = (F.relu(torch.abs(dg - rg) - edge_margin) * weight)[valid].mean()

    total = (
        float(getattr(cfg_node, "da3_edge_weight", 1.0)) * edge_loss
        + float(getattr(cfg_node, "da3_ranking_weight", 1.0)) * ranking_loss
        + float(getattr(cfg_node, "da3_side_weight", 0.5)) * side_loss
    )
    logs = {
        "da3_structure_valid_pixels": valid_count,
        "da3_edge_loss": float(edge_loss.detach().item()),
        "da3_ranking_loss": float(ranking_loss.detach().item()),
        "da3_side_loss": float(side_loss.detach().item()),
    }
    return total, logs
