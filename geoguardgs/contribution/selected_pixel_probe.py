import argparse
import ast
import csv
import json
import os
import sys
import time
import zlib
from collections import defaultdict

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PWD", PROJECT_ROOT)


def parse_script_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--geometry-error-map-dir", default="output/local_formal/geometry_error_map")
    parser.add_argument("--v0-dir", default="output/local_formal/gaussian_responsibility_v0_1_A")
    parser.add_argument("--output-dir", default="output/local_formal/contribution_responsibility_debug_A5000")
    parser.add_argument("--views", nargs="+", default=["000002_0"])
    parser.add_argument("--regions-csv", default=None, help="Optional filtered_region_candidates.csv for batch region mode.")
    parser.add_argument("--feedback-signal", default=None, help="Optional guided feedback signal JSON that contains pixel_feedback_by_view.")
    parser.add_argument("--pixel-source", choices=["geometry_error_map", "feedback_signal"], default="geometry_error_map")
    parser.add_argument("--top-regions", type=int, default=5)
    parser.add_argument("--min-evidence-pixels", type=int, default=2)
    parser.add_argument("--high-error-quantile", type=float, default=0.95)
    parser.add_argument("--max-pixels", type=int, default=128)
    parser.add_argument("--pixel-bbox", nargs=4, type=int, default=None, help="Optional x0 y0 x1 y1 region for selecting high-error pixels.")
    parser.add_argument("--candidate-radius-scale", type=float, default=2.0)
    parser.add_argument("--max-candidate-gaussians", type=int, default=12000)
    parser.add_argument("--old-v0-pool-size", type=int, default=5000)
    parser.add_argument("--semantic-chunk-size", type=int, default=32)
    parser.add_argument("--contribution-backend", choices=["auto", "cuda", "semantic"], default="auto")
    parser.add_argument("--top-k-per-pixel", type=int, default=8)
    parser.add_argument("--top-gaussians", type=int, default=50)
    parser.add_argument("--counterfactual-top-k", type=int, default=5)
    parser.add_argument("--sanity-semantic-max-rows", type=int, default=256)
    parser.add_argument("--weaken-scale", type=float, default=0.0)
    parser.add_argument("--geometry-epsilon", type=float, default=0.1)
    parser.add_argument("--rgb-error-tolerance", type=float, default=0.02)
    parser.add_argument("--patch-radius", type=int, default=12)
    parser.add_argument("--lidar-dir", default="data/waymo/002/lidar_depth")
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--help-contribution", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.help_contribution:
        parser.print_help()
        sys.exit(0)
    sys.argv = [sys.argv[0]] + remaining
    return args


SCRIPT_ARGS = parse_script_args()

from lib.config import cfg  # noqa: E402
from lib.datasets.dataset import Dataset  # noqa: E402
from lib.models.scene import Scene  # noqa: E402
from lib.models.street_gaussian_model import StreetGaussianModel  # noqa: E402
from lib.utils.camera_utils import make_rasterizer  # noqa: E402
from lib.utils.general_utils import safe_state  # noqa: E402
from lib.utils.sh_utils import eval_sh  # noqa: E402
try:
    from diff_gaussian_rasterization import _C as rasterizer_C  # noqa: E402
except Exception:
    rasterizer_C = None


STABLE_ID_STRIDE = 10_000_000


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def stable_namespace_id(model_name):
    if model_name == "background":
        return 0
    if model_name.startswith("obj_"):
        try:
            return int(model_name.split("_", 1)[1]) + 1
        except ValueError:
            return 9_000_000
    return 9_500_000 + (zlib.crc32(model_name.encode("utf-8")) % 100_000)


def build_stable_ids(pc, count):
    stable_ids = np.full(count, -1, dtype=np.int64)
    model_names = np.full(count, "unknown", dtype="<U64")
    model_local = np.full(count, -1, dtype=np.int64)
    for model_name, span in getattr(pc, "graph_gaussian_range", {}).items():
        start, end = int(span[0]), int(span[1])
        n = max(0, end - start)
        local = np.arange(n, dtype=np.int64)
        stable_ids[start:end] = stable_namespace_id(model_name) * STABLE_ID_STRIDE + local
        model_names[start:end] = model_name
        model_local[start:end] = local
    fallback = stable_ids < 0
    if np.any(fallback):
        rows = np.flatnonzero(fallback).astype(np.int64)
        stable_ids[fallback] = 9_999_999 * STABLE_ID_STRIDE + rows
        model_local[fallback] = rows
    return stable_ids, model_names, model_local


def load_components(root, stem):
    view_dir = os.path.join(root, stem)
    comp = np.load(os.path.join(view_dir, "geometry_error_components.npz"))
    error_map = np.load(os.path.join(view_dir, "geometry_error_map.npy")).astype(np.float32)
    return {
        "error_map": error_map,
        "valid": comp["valid_lidar_mask"].astype(bool),
        "A_abs_error": comp["A_abs_error"].astype(np.float32) if "A_abs_error" in comp.files else error_map,
    }


def load_lidar(path, shape):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == () and isinstance(data.item(), dict):
        data = data.item()
    if isinstance(data, dict) and "mask" in data and "value" in data:
        mask = np.asarray(data["mask"]).astype(bool)
        depth = np.zeros(mask.shape, dtype=np.float32)
        depth[mask] = np.asarray(data["value"], dtype=np.float32).reshape(-1)
    else:
        depth = np.asarray(data, dtype=np.float32).squeeze()
        mask = np.isfinite(depth) & (depth > 0)
    if depth.shape[:2] != shape:
        depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return depth, mask


def load_original_rgb(camera, shape=None):
    image = to_numpy(camera.original_image)
    image = np.transpose(image[:3], (1, 2, 0))
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    if shape is not None and image.shape[:2] != shape:
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return image


def parse_bbox(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [int(v) for v in value]
    parsed = ast.literal_eval(str(value))
    return [int(v) for v in parsed]


def load_region_specs(args):
    if not args.regions_csv:
        return [
            {
                "view_id": stem,
                "region_id": "manual",
                "bbox": args.pixel_bbox,
                "region_type": "manual",
                "review_score": None,
            }
            for stem in args.views
        ]
    rows = []
    with open(args.regions_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["bbox"] = parse_bbox(row.get("bbox"))
            try:
                row["review_score"] = float(row.get("review_score", 0.0))
            except Exception:
                row["review_score"] = 0.0
            rows.append(row)
    rows.sort(key=lambda item: item.get("review_score") or 0.0, reverse=True)
    return rows[: args.top_regions]


def project_points(points, camera):
    points_h = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
    clip = torch.matmul(points_h, camera.full_proj_transform)
    ndc = clip[:, :3] / (clip[:, 3:4] + 1e-8)
    x = (ndc[:, 0] * 0.5 + 0.5) * float(camera.image_width)
    y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * float(camera.image_height)
    return torch.stack([x, y], dim=-1)


def camera_depth(points, camera):
    points_h = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
    view = torch.matmul(points_h, camera.world_view_transform)
    return view[:, 2]


def select_high_error_pixels(error_map, valid, max_pixels, quantile, bbox=None):
    mask = valid & np.isfinite(error_map)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        region = np.zeros_like(mask, dtype=bool)
        x0 = max(0, min(mask.shape[1], x0))
        x1 = max(0, min(mask.shape[1], x1))
        y0 = max(0, min(mask.shape[0], y0))
        y1 = max(0, min(mask.shape[0], y1))
        region[y0:y1, x0:x1] = True
        mask &= region
    if not np.any(mask):
        return np.zeros((0, 2), dtype=np.int32)
    threshold = float(np.quantile(error_map[mask], quantile))
    ys, xs = np.where(mask & (error_map >= threshold))
    if len(xs) == 0:
        ys, xs = np.where(mask)
    values = error_map[ys, xs]
    order = np.argsort(values)[::-1][:max_pixels]
    return np.stack([xs[order], ys[order]], axis=1).astype(np.int32)


def load_feedback_pixels(signal_path, stem, bbox=None, max_pixels=128):
    if not signal_path or not os.path.exists(signal_path):
        return np.zeros((0, 2), dtype=np.int32)
    with open(signal_path, "r", encoding="utf-8") as f:
        signal = json.load(f)
    pixels = []
    for record in signal.get("pixel_feedback_by_view", []):
        if str(record.get("view_id", "")) != str(stem):
            continue
        for item in record.get("bad_pixels", []):
            if len(item) >= 2:
                pixels.append([int(item[0]), int(item[1])])
    if not pixels:
        return np.zeros((0, 2), dtype=np.int32)
    arr = np.asarray(pixels, dtype=np.int32)
    if bbox is not None:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        keep = (arr[:, 0] >= x0) & (arr[:, 0] < x1) & (arr[:, 1] >= y0) & (arr[:, 1] < y1)
        arr = arr[keep]
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    # Stable deterministic sub-sampling: keep spatially sorted pixels, which are already risk-ranked in signal order.
    _, unique_idx = np.unique(arr[:, 1].astype(np.int64) * 1_000_000 + arr[:, 0].astype(np.int64), return_index=True)
    arr = arr[np.sort(unique_idx)]
    return arr[:max_pixels].astype(np.int32)


def candidate_pool_from_pixels(centers, radii, visibility, pixels, args):
    visible = np.flatnonzero(visibility & np.isfinite(centers[:, 0]) & np.isfinite(centers[:, 1]) & (radii > 0))
    selected = []
    for start in range(0, len(visible), 8192):
        rows = visible[start : start + 8192]
        c = centers[rows]
        r = np.maximum(radii[rows] * args.candidate_radius_scale, 2.0)
        keep = np.zeros(len(rows), dtype=bool)
        for p_start in range(0, len(pixels), 64):
            p = pixels[p_start : p_start + 64].astype(np.float32)
            dx = c[:, None, 0] - p[None, :, 0]
            dy = c[:, None, 1] - p[None, :, 1]
            keep |= np.any((dx * dx + dy * dy) <= (r[:, None] * r[:, None]), axis=1)
        selected.append(rows[keep])
    if not selected:
        return np.zeros(0, dtype=np.int64)
    pool = np.unique(np.concatenate(selected))
    if len(pool) > args.max_candidate_gaussians:
        # Prefer smaller screen-distance candidates around the selected high-error pixels.
        px = pixels.astype(np.float32)
        c = centers[pool]
        d2 = np.min((c[:, None, 0] - px[None, :, 0]) ** 2 + (c[:, None, 1] - px[None, :, 1]) ** 2, axis=1)
        pool = pool[np.argsort(d2)[: args.max_candidate_gaussians]]
    return pool.astype(np.int64)


def old_v0_pool(v0_dir, stem, limit):
    path = os.path.join(v0_dir, stem, "gaussian_responsibility_v0.npz")
    if not os.path.exists(path) or limit <= 0:
        return np.zeros(0, dtype=np.int64)
    data = np.load(path, allow_pickle=True)
    if "view_local_gaussian_indices" in data.files:
        rows = data["view_local_gaussian_indices"].astype(np.int64)
    else:
        rows = data["gaussian_ids"].astype(np.int64)
    scores = data["responsibility_scores"].astype(np.float32)
    order = np.argsort(scores)[::-1][: min(limit, len(scores))]
    return rows[order].astype(np.int64)


def prepare_raster_inputs(camera, pc, opacity_override=None, semantics=None):
    means3D = pc.get_xyz
    opacity = pc.get_opacity if opacity_override is None else opacity_override
    scales = None
    rotations = None
    cov3D_precomp = None
    if cfg.render.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(cfg.render.scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if cfg.render.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = pc.get_xyz - camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        colors_precomp = torch.clamp_min(eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) + 0.5, 0.0)
    else:
        try:
            shs = pc.get_features
        except Exception:
            colors_precomp = pc.get_colors(camera.camera_center)

    bg_color = torch.zeros(3, device=means3D.device, dtype=torch.float32)
    rasterizer = make_rasterizer(camera, pc.max_sh_degree, bg_color, cfg.render.scaling_modifier)
    means2D = None
    kwargs = dict(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        shs=shs,
        colors_precomp=colors_precomp,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    if semantics is not None:
        kwargs["semantics"] = semantics
    return rasterizer, kwargs


def render_custom(camera, pc, opacity_override=None, semantics=None):
    rasterizer, kwargs = prepare_raster_inputs(camera, pc, opacity_override=opacity_override, semantics=semantics)
    result = rasterizer(**kwargs)
    color, radii, depth, alpha, semantic = result
    return {"rgb": color, "radii": radii, "depth": depth, "acc": alpha, "semantic": semantic}


def cuda_contribution_available():
    return rasterizer_C is not None and hasattr(rasterizer_C, "rasterize_gaussians_contrib")


def empty_like_if_none(tensor, device):
    if tensor is not None:
        return tensor
    return torch.Tensor([]).to(device)


def capture_contributions_cuda(camera, pc, pixels, top_k):
    device = pc.get_xyz.device
    rasterizer, kwargs = prepare_raster_inputs(camera, pc, semantics=None)
    settings = rasterizer.raster_settings
    means3D = kwargs["means3D"]
    selected_pixels = torch.as_tensor(pixels, device=device, dtype=torch.int32).contiguous()
    shs = empty_like_if_none(kwargs.get("shs"), device)
    colors_precomp = empty_like_if_none(kwargs.get("colors_precomp"), device)
    semantics = torch.zeros((means3D.shape[0], 0), device=device, dtype=torch.float32)
    scales = empty_like_if_none(kwargs.get("scales"), device)
    rotations = empty_like_if_none(kwargs.get("rotations"), device)
    cov3D_precomp = empty_like_if_none(kwargs.get("cov3D_precomp"), device)
    with torch.no_grad():
        ids, alpha, transmittance, weight, depth, depth_order = rasterizer_C.rasterize_gaussians_contrib(
            settings.bg,
            means3D,
            colors_precomp,
            semantics,
            kwargs["opacities"],
            scales,
            rotations,
            settings.scale_modifier,
            cov3D_precomp,
            settings.viewmatrix,
            settings.projmatrix,
            settings.tanfovx,
            settings.tanfovy,
            settings.image_height,
            settings.image_width,
            shs,
            settings.sh_degree,
            settings.campos,
            settings.prefiltered,
            settings.debug,
            selected_pixels,
            int(top_k),
        )
    return {
        "ids": to_numpy(ids).astype(np.int64),
        "alpha": to_numpy(alpha).astype(np.float32),
        "transmittance": to_numpy(transmittance).astype(np.float32),
        "weight": to_numpy(weight).astype(np.float32),
        "depth": to_numpy(depth).astype(np.float32),
        "depth_order": to_numpy(depth_order).astype(np.int32),
    }


def capture_contributions(camera, pc, pool_rows, pixels, args):
    device = pc.get_xyz.device
    n = pc.get_xyz.shape[0]
    py = torch.as_tensor(pixels[:, 1], device=device, dtype=torch.long)
    px = torch.as_tensor(pixels[:, 0], device=device, dtype=torch.long)
    pixel_weights = np.zeros((len(pixels), len(pool_rows)), dtype=np.float32)
    for offset in range(0, len(pool_rows), args.semantic_chunk_size):
        chunk = pool_rows[offset : offset + args.semantic_chunk_size]
        semantics = torch.zeros((n, len(chunk)), device=device, dtype=torch.float32)
        semantics[torch.as_tensor(chunk, device=device, dtype=torch.long), torch.arange(len(chunk), device=device)] = 1.0
        with torch.no_grad():
            sem = render_custom(camera, pc, semantics=semantics)["semantic"]
            weights = sem[:, py, px].transpose(0, 1).contiguous()
        pixel_weights[:, offset : offset + len(chunk)] = to_numpy(weights)
        del semantics, sem, weights
        torch.cuda.empty_cache()
    return pixel_weights


def summarize_contributions(pool_rows, stable_ids, model_names, model_local, depths, pixels, weights, error_map, top_k):
    per_pixel = []
    responsibility = defaultdict(float)
    support = defaultdict(int)
    max_weight = defaultdict(float)
    for p_idx, (x, y) in enumerate(pixels):
        w = weights[p_idx]
        nz = np.flatnonzero(w > 1e-6)
        if nz.size == 0:
            per_pixel.append({"x": int(x), "y": int(y), "geometry_error": float(error_map[y, x]), "contributors": []})
            continue
        order = nz[np.argsort(w[nz])[::-1][:top_k]]
        contributors = []
        for local in order:
            row = int(pool_rows[local])
            gid = int(stable_ids[row])
            weight = float(w[local])
            err = float(error_map[y, x])
            responsibility[gid] += weight * err
            support[gid] += 1
            max_weight[gid] = max(max_weight[gid], weight)
            contributors.append(
                {
                    "gaussian_id": gid,
                    "view_local_index": row,
                    "model_name": str(model_names[row]),
                    "model_local_index": int(model_local[row]),
                    "alpha_transmittance_weight": weight,
                    "depth_order": int(np.where(order == local)[0][0]),
                    "gaussian_depth": float(depths[row]),
                    "rendered_depth_contribution": float(depths[row] * weight),
                }
            )
        per_pixel.append({"x": int(x), "y": int(y), "geometry_error": float(error_map[y, x]), "contributors": contributors})
    rows = []
    for gid, score in responsibility.items():
        rows.append(
            {
                "gaussian_id": int(gid),
                "contribution_responsibility": float(score),
                "support_pixel_count": int(support[gid]),
                "max_contribution_weight": float(max_weight[gid]),
            }
        )
    rows.sort(key=lambda item: item["contribution_responsibility"], reverse=True)
    return per_pixel, rows


def summarize_cuda_contributions(stable_ids, model_names, model_local, pixels, cuda_data, error_map):
    per_pixel = []
    responsibility = defaultdict(float)
    support = defaultdict(int)
    max_weight = defaultdict(float)
    ids = cuda_data["ids"]
    weights = cuda_data["weight"]
    for p_idx, (x, y) in enumerate(pixels):
        contributors = []
        for k in range(ids.shape[1]):
            row = int(ids[p_idx, k])
            weight = float(weights[p_idx, k])
            if row < 0 or weight <= 1e-6:
                continue
            gid = int(stable_ids[row])
            err = float(error_map[y, x])
            responsibility[gid] += weight * err
            support[gid] += 1
            max_weight[gid] = max(max_weight[gid], weight)
            contributors.append(
                {
                    "gaussian_id": gid,
                    "view_local_index": row,
                    "model_name": str(model_names[row]),
                    "model_local_index": int(model_local[row]),
                    "alpha": float(cuda_data["alpha"][p_idx, k]),
                    "transmittance": float(cuda_data["transmittance"][p_idx, k]),
                    "alpha_transmittance_weight": weight,
                    "depth_order": int(cuda_data["depth_order"][p_idx, k]),
                    "gaussian_depth": float(cuda_data["depth"][p_idx, k]),
                    "rendered_depth_contribution": float(cuda_data["depth"][p_idx, k] * weight),
                }
            )
        per_pixel.append({"x": int(x), "y": int(y), "geometry_error": float(error_map[y, x]), "contributors": contributors})
    rows = []
    for gid, score in responsibility.items():
        rows.append(
            {
                "gaussian_id": int(gid),
                "contribution_responsibility": float(score),
                "support_pixel_count": int(support[gid]),
                "max_contribution_weight": float(max_weight[gid]),
            }
        )
    rows.sort(key=lambda item: item["contribution_responsibility"], reverse=True)
    return per_pixel, rows


def summarize_array(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "max": None, "p95": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
    }


def compute_cuda_raster_sanity(cuda_data, pixels, rendered_acc, rendered_depth):
    ids = cuda_data["ids"]
    valid = ids >= 0
    weights = np.where(valid, cuda_data["weight"], 0.0).astype(np.float64)
    depths = np.where(valid, cuda_data["depth"], 0.0).astype(np.float64)
    topk_acc = np.sum(weights, axis=1)
    topk_depth_contrib = np.sum(weights * depths, axis=1)
    acc_vals = rendered_acc[pixels[:, 1], pixels[:, 0]].astype(np.float64)
    depth_vals = rendered_depth[pixels[:, 1], pixels[:, 0]].astype(np.float64)
    eps = 1e-8
    return {
        "note": "top-K sparse dump is compared against full rendered acc/depth. Residuals are expected if top_k does not cover all contributors.",
        "selected_pixel_count": int(len(pixels)),
        "topk_per_pixel": int(ids.shape[1]) if ids.ndim == 2 else 0,
        "valid_contribution_entries": int(np.count_nonzero(valid)),
        "topk_sum_weight_vs_rendered_acc_abs": summarize_array(np.abs(topk_acc - acc_vals)),
        "topk_sum_weight_over_rendered_acc": summarize_array(topk_acc / np.maximum(acc_vals, eps)),
        "topk_depth_contrib_vs_rendered_depth_abs": summarize_array(np.abs(topk_depth_contrib - depth_vals)),
        "topk_depth_contrib_over_rendered_depth": summarize_array(topk_depth_contrib / np.maximum(np.abs(depth_vals), eps)),
        "per_pixel_preview": [
            {
                "x": int(x),
                "y": int(y),
                "rendered_acc": float(acc_vals[i]),
                "topk_sum_weight": float(topk_acc[i]),
                "acc_abs_residual": float(abs(topk_acc[i] - acc_vals[i])),
                "rendered_depth": float(depth_vals[i]),
                "topk_depth_contribution": float(topk_depth_contrib[i]),
                "depth_abs_residual": float(abs(topk_depth_contrib[i] - depth_vals[i])),
            }
            for i, (x, y) in enumerate(pixels[: min(12, len(pixels))])
        ],
    }


def compare_cuda_with_semantic_shared(camera, pc, pixels, cuda_data, args):
    cuda_ids = cuda_data["ids"]
    valid_rows = np.unique(cuda_ids[cuda_ids >= 0].astype(np.int64))
    if valid_rows.size == 0:
        return {"tested_rows": 0, "shared_entries": 0, "reason": "no CUDA contributors"}
    truncated = valid_rows.size > args.sanity_semantic_max_rows
    rows = valid_rows[: args.sanity_semantic_max_rows]
    semantic_weights = capture_contributions(camera, pc, rows, pixels, args)
    row_to_col = {int(row): idx for idx, row in enumerate(rows)}
    abs_diffs = []
    rel_diffs = []
    shared_entries = 0
    skipped_not_tested = 0
    previews = []
    for p_idx, (x, y) in enumerate(pixels):
        for k in range(cuda_ids.shape[1]):
            row = int(cuda_ids[p_idx, k])
            cuda_w = float(cuda_data["weight"][p_idx, k])
            if row < 0 or cuda_w <= 1e-8:
                continue
            col = row_to_col.get(row)
            if col is None:
                skipped_not_tested += 1
                continue
            sem_w = float(semantic_weights[p_idx, col])
            diff = abs(cuda_w - sem_w)
            abs_diffs.append(diff)
            rel_diffs.append(diff / max(abs(cuda_w), abs(sem_w), 1e-8))
            shared_entries += 1
            if len(previews) < 16:
                previews.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "view_local_index": row,
                        "cuda_weight": cuda_w,
                        "semantic_weight": sem_w,
                        "abs_diff": diff,
                    }
                )
    return {
        "tested_rows": int(len(rows)),
        "available_cuda_unique_rows": int(valid_rows.size),
        "truncated_by_max_rows": bool(truncated),
        "shared_entries": int(shared_entries),
        "skipped_entries_not_tested": int(skipped_not_tested),
        "abs_diff": summarize_array(abs_diffs),
        "relative_diff": summarize_array(rel_diffs),
        "preview": previews,
    }


def analyze_semantic_pool_gap(pool, cuda_data, stable_ids):
    cuda_rows = np.unique(cuda_data["ids"][cuda_data["ids"] >= 0].astype(np.int64))
    pool_set = set(int(x) for x in pool.tolist())
    missing_rows = [int(row) for row in cuda_rows if int(row) not in pool_set]
    return {
        "cuda_unique_contributor_rows": int(len(cuda_rows)),
        "screen_space_pool_count": int(len(pool)),
        "cuda_rows_missing_from_screen_space_pool_count": int(len(missing_rows)),
        "cuda_rows_missing_from_screen_space_pool_sample": missing_rows[:20],
        "stable_ids_missing_from_screen_space_pool_sample": [int(stable_ids[row]) for row in missing_rows[:20]],
        "interpretation": "If CUDA contributors are missing from the semantic fallback pool, CUDA-vs-semantic top-K differences can be caused by pool truncation rather than id or pixel-coordinate mismatch.",
    }


def classify_counterfactual(geometry_reduction, rgb_error_increase, args):
    if geometry_reduction > args.geometry_epsilon and rgb_error_increase <= args.rgb_error_tolerance:
        return "bad_contributor"
    if geometry_reduction < -args.geometry_epsilon:
        return "good_contributor"
    return "neutral_contributor"


def counterfactual_probe(
    camera,
    pc,
    top_rows,
    pixels,
    lidar_depth,
    lidar_mask,
    original_depth,
    original_rgb,
    base_rgb,
    args,
    out_dir=None,
    bbox=None,
):
    base_opacity = pc.get_opacity
    valid = lidar_mask[pixels[:, 1], pixels[:, 0]] & np.isfinite(lidar_depth[pixels[:, 1], pixels[:, 0]])
    if not np.any(valid):
        return []
    lidar_values = lidar_depth[pixels[:, 1], pixels[:, 0]]
    base_depth_values = original_depth[pixels[:, 1], pixels[:, 0]]
    base_error = np.abs(base_depth_values - lidar_values)
    original_rgb_values = original_rgb[pixels[:, 1], pixels[:, 0], :]
    base_rgb_values = base_rgb[pixels[:, 1], pixels[:, 0], :]
    base_rgb_error = np.mean(np.abs(base_rgb_values - original_rgb_values), axis=1)
    results = []
    for row in top_rows:
        opacity = base_opacity.clone()
        opacity[row] = opacity[row] * float(args.weaken_scale)
        with torch.no_grad():
            weakened = render_custom(camera, pc, opacity_override=opacity)
            depth_np = np.squeeze(to_numpy(weakened["depth"]).astype(np.float32))
            rgb_np = np.transpose(to_numpy(weakened["rgb"]).astype(np.float32), (1, 2, 0))
            new_depth_values = depth_np[pixels[:, 1], pixels[:, 0]]
        new_error = np.abs(new_depth_values - lidar_values)
        new_rgb_values = rgb_np[pixels[:, 1], pixels[:, 0], :]
        new_rgb_error = np.mean(np.abs(new_rgb_values - original_rgb_values), axis=1)
        delta = base_error[valid] - new_error[valid]
        rgb_delta = new_rgb_error[valid] - base_rgb_error[valid]
        depth_change = np.abs(new_depth_values[valid] - base_depth_values[valid])
        mean_geometry_reduction = float(np.mean(delta))
        mean_rgb_error_increase = float(np.mean(rgb_delta))
        visual_path = None
        if args.save_visuals and out_dir is not None and bbox is not None:
            visual_dir = ensure_dir(os.path.join(out_dir, "counterfactual_before_after"))
            visual_path = os.path.join(visual_dir, f"row_{int(row)}_before_after.png")
            save_counterfactual_sheet(
                visual_path,
                original_rgb,
                base_rgb,
                rgb_np,
                original_depth,
                depth_np,
                lidar_depth,
                lidar_mask,
                bbox,
                pixels,
            )
        results.append(
            {
                "view_local_index": int(row),
                "mean_error_before": float(np.mean(base_error[valid])),
                "mean_error_after": float(np.mean(new_error[valid])),
                "mean_error_reduction": mean_geometry_reduction,
                "improved_pixel_ratio": float(np.mean(delta > 0)),
                "mean_rgb_error_before": float(np.mean(base_rgb_error[valid])),
                "mean_rgb_error_after": float(np.mean(new_rgb_error[valid])),
                "mean_rgb_error_increase": mean_rgb_error_increase,
                "mean_rendered_depth_change": float(np.mean(depth_change)),
                "counterfactual_label": classify_counterfactual(mean_geometry_reduction, mean_rgb_error_increase, args),
                "before_after_visual": visual_path,
            }
        )
        del opacity, weakened
        torch.cuda.empty_cache()
    return results


def compare_with_old_v0(v0_dir, stem, contribution_rows, stable_ids):
    path = os.path.join(v0_dir, stem, "gaussian_responsibility_v0.npz")
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=True)
    old_ids = data["gaussian_ids"][data["topk_indices"][:100]].astype(np.int64)
    new_ids = np.asarray([r["gaussian_id"] for r in contribution_rows[:100]], dtype=np.int64)
    overlap = len(set(int(x) for x in old_ids) & set(int(x) for x in new_ids))
    new_set = set(int(x) for x in new_ids)
    old_set = set(int(x) for x in old_ids)
    return {
        "old_top100_count": int(len(old_ids)),
        "new_top100_count": int(len(new_ids)),
        "top100_id_overlap_count": int(overlap),
        "old_top_excluded_sample": [int(x) for x in old_ids if int(x) not in new_set][:20],
        "new_top_not_in_old_sample": [int(x) for x in new_ids if int(x) not in old_set][:20],
        "note": "Old V0 ids may be legacy view-local ids; compare is indicative unless V0 was regenerated with stable ids.",
    }


def save_overlay(path, camera, pixels):
    image = to_numpy(camera.original_image)
    image = np.transpose(image[:3], (1, 2, 0))
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for x, y in pixels:
        cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.imwrite(path, image)


def normalize_u8(values, valid=None):
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values) if valid is None else (valid & np.isfinite(values))
    out = np.zeros(values.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(values[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        out[valid] = np.clip((values[valid] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


def crop_with_bbox(image, bbox, pad=12):
    x0, y0, x1, y1 = bbox
    h, w = image.shape[:2]
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    return image[y0:y1, x0:x1]


def save_counterfactual_sheet(path, original_rgb, base_rgb, weakened_rgb, base_depth, weakened_depth, lidar_depth, lidar_mask, bbox, pixels):
    valid = lidar_mask & np.isfinite(lidar_depth)
    base_depth_error = np.zeros_like(base_depth, dtype=np.float32)
    weakened_depth_error = np.zeros_like(weakened_depth, dtype=np.float32)
    base_depth_error[valid] = np.abs(base_depth[valid] - lidar_depth[valid])
    weakened_depth_error[valid] = np.abs(weakened_depth[valid] - lidar_depth[valid])

    original = cv2.cvtColor((np.clip(original_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    base = cv2.cvtColor((np.clip(base_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    weakened = cv2.cvtColor((np.clip(weakened_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    rgb_before = cv2.applyColorMap(normalize_u8(np.mean(np.abs(base_rgb - original_rgb), axis=2)), cv2.COLORMAP_INFERNO)
    rgb_after = cv2.applyColorMap(normalize_u8(np.mean(np.abs(weakened_rgb - original_rgb), axis=2)), cv2.COLORMAP_INFERNO)
    depth_before = cv2.applyColorMap(normalize_u8(base_depth_error, valid), cv2.COLORMAP_TURBO)
    depth_after = cv2.applyColorMap(normalize_u8(weakened_depth_error, valid), cv2.COLORMAP_TURBO)
    for canvas in [base, weakened, rgb_before, rgb_after, depth_before, depth_after]:
        for x, y in pixels:
            cv2.circle(canvas, (int(x), int(y)), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    panels = [crop_with_bbox(img, bbox) for img in [original, base, weakened, rgb_before, rgb_after, depth_before, depth_after]]
    h = max(p.shape[0] for p in panels)
    w = max(p.shape[1] for p in panels)
    panels = [cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in panels]
    blank = np.zeros_like(panels[0])
    row1 = np.concatenate([panels[0], panels[1], panels[2]], axis=1)
    row2 = np.concatenate([panels[3], panels[4], blank], axis=1)
    row3 = np.concatenate([panels[5], panels[6], blank], axis=1)
    cv2.imwrite(path, np.concatenate([row1, row2, row3], axis=0))


def save_region_contact_sheet(path, original_rgb, base_rgb, error_map, pixels, bbox):
    error_vis = cv2.applyColorMap(normalize_u8(error_map, np.isfinite(error_map)), cv2.COLORMAP_TURBO)
    original = (np.clip(original_rgb, 0, 1) * 255).astype(np.uint8)
    rendered = (np.clip(base_rgb, 0, 1) * 255).astype(np.uint8)
    rgb_error = normalize_u8(np.mean(np.abs(base_rgb - original_rgb), axis=2))
    rgb_error = cv2.applyColorMap(rgb_error, cv2.COLORMAP_INFERNO)
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
    for x, y in pixels:
        cv2.circle(rendered, (int(x), int(y)), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(error_vis, (int(x), int(y)), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    panels = [crop_with_bbox(img, bbox) for img in [original, rendered, rgb_error, error_vis]]
    h = max(p.shape[0] for p in panels)
    w = max(p.shape[1] for p in panels)
    resized = [cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in panels]
    sheet = np.concatenate([np.concatenate(resized[:2], axis=1), np.concatenate(resized[2:], axis=1)], axis=0)
    cv2.imwrite(path, sheet)


def write_training_feedback_signal(output_dir, summaries):
    signal_path = os.path.join(output_dir, "guided_training_feedback_signal.json")
    table_path = os.path.join(output_dir, "bad_good_neutral_contributor_table.csv")
    bad, good, neutral, low_evidence, region_signals = [], [], [], [], []
    for summary in summaries:
        stem = summary.get("stem")
        region_id = summary.get("region_id")
        region_key = f"{stem}:region{region_id}"
        if summary.get("status") == "low_evidence":
            low_evidence.append(
                {
                    "region_key": region_key,
                    "view_id": stem,
                    "region_id": region_id,
                    "reason": summary.get("reason"),
                    "selected_pixel_count": summary.get("selected_pixel_count"),
                    "recommendation": "skip_feedback_until_more_valid_lidar_evidence",
                }
            )
            continue
        label_counts = summary.get("counterfactual_label_counts", {})
        region_action = "skip"
        if label_counts.get("bad_contributor", 0) > 0:
            region_action = "increase_local_geometry_supervision"
        elif label_counts.get("good_contributor", 0) > 0:
            region_action = "protect_current_contributors"
        region_signals.append(
            {
                "region_key": region_key,
                "view_id": stem,
                "region_id": region_id,
                "region_type": summary.get("region_type"),
                "selected_pixel_count": summary.get("selected_pixel_count"),
                "bad_count": int(label_counts.get("bad_contributor", 0)),
                "good_count": int(label_counts.get("good_contributor", 0)),
                "neutral_count": int(label_counts.get("neutral_contributor", 0)),
                "recommended_feedback": region_action,
            }
        )
        for item in summary.get("counterfactual", []):
            label = item.get("counterfactual_label", "neutral_contributor")
            row = {
                "region_key": region_key,
                "view_id": stem,
                "region_id": region_id,
                "region_type": summary.get("region_type"),
                "gaussian_id": int(item.get("gaussian_id", -1)),
                "view_local_index": int(item.get("view_local_index", -1)),
                "model_name": item.get("model_name"),
                "counterfactual_label": label,
                "mean_error_before": item.get("mean_error_before"),
                "mean_error_after": item.get("mean_error_after"),
                "mean_error_reduction": item.get("mean_error_reduction"),
                "improved_pixel_ratio": item.get("improved_pixel_ratio"),
                "mean_rgb_error_increase": item.get("mean_rgb_error_increase"),
                "recommended_feedback": (
                    "upweight_local_geometry_supervision"
                    if label == "bad_contributor"
                    else "protect_or_preserve"
                    if label == "good_contributor"
                    else "skip"
                ),
            }
            if label == "bad_contributor":
                bad.append(row)
            elif label == "good_contributor":
                good.append(row)
            else:
                neutral.append(row)
    payload = {
        "debug_only": True,
        "do_not_start_training": True,
        "source": "CUDA selected-pixel contribution dump + temporary counterfactual opacity weakening",
        "policy_summary": {
            "bad_contributor_or_bad_region": "future guided training may increase reliable local geometry supervision",
            "good_contributor": "protect from weakening/removal; do not use as bad-geometry target",
            "neutral_or_low_evidence": "skip until stronger evidence is available",
        },
        "counts": {
            "regions": int(len(summaries)),
            "ok_regions": int(sum(1 for s in summaries if s.get("status") == "ok")),
            "low_evidence_regions": int(len(low_evidence)),
            "bad_contributors": int(len(bad)),
            "good_contributors": int(len(good)),
            "neutral_contributors": int(len(neutral)),
        },
        "bad_contributors": bad,
        "good_contributors": good,
        "neutral_contributors": neutral,
        "low_evidence_regions": low_evidence,
        "region_signals": region_signals,
        "next_experiment_note": "Use this file only as a proposal for plain-vs-guided continue-training. It does not modify loss weights or Gaussian parameters by itself.",
    }
    with open(signal_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    rows = bad + good + neutral
    with open(table_path, "w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write("region_key,view_id,region_id,gaussian_id,counterfactual_label,recommended_feedback\n")
    return {"json": signal_path, "csv": table_path, "counts": payload["counts"]}


def run_region(region, camera, pc, out_root, args):
    stem = region["view_id"]
    region_id = str(region.get("region_id", "manual"))
    safe_region_type = str(region.get("region_type", "manual")).replace("/", "_")
    out_dir = ensure_dir(os.path.join(out_root, f"{stem}_region{region_id}_{safe_region_type}"))
    summary_path = os.path.join(out_dir, "contribution_responsibility_summary.json")
    if args.skip_existing and os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    comps = load_components(args.geometry_error_map_dir, stem)
    error_map = comps["error_map"]
    bbox = region.get("bbox") or args.pixel_bbox
    if args.pixel_source == "feedback_signal":
        pixels = load_feedback_pixels(args.feedback_signal, stem, bbox=bbox, max_pixels=args.max_pixels)
    else:
        pixels = select_high_error_pixels(error_map, comps["valid"], args.max_pixels, args.high_error_quantile, bbox)
    if len(pixels) < args.min_evidence_pixels:
        summary = {
            "stem": stem,
            "region_id": region_id,
            "region_type": region.get("region_type"),
            "status": "low_evidence",
            "reason": (
                "not enough DA3 feedback pixels in region"
                if args.pixel_source == "feedback_signal"
                else "not enough LiDAR-valid high-error pixels"
            ),
            "selected_pixel_count": int(len(pixels)),
            "min_evidence_pixels": int(args.min_evidence_pixels),
            "pixel_bbox": bbox,
            "pixel_source": args.pixel_source,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    timings = {}
    t0 = time.perf_counter()

    pc.set_visibility(list(pc.model_name_id.keys()))
    pc.parse_camera(camera)
    with torch.no_grad():
        base = render_custom(camera, pc)
        radii = to_numpy(base["radii"]).astype(np.float32)
        visibility = radii > 0
        centers = to_numpy(project_points(pc.get_xyz, camera)).astype(np.float32)
        depths = to_numpy(camera_depth(pc.get_xyz, camera)).astype(np.float32)
        original_depth = np.squeeze(to_numpy(base["depth"]).astype(np.float32))
        original_acc = np.squeeze(to_numpy(base["acc"]).astype(np.float32))
        base_rgb = np.transpose(to_numpy(base["rgb"]).astype(np.float32), (1, 2, 0))
        original_rgb = load_original_rgb(camera, base_rgb.shape[:2])
    timings["baseline_render_sec"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    visible_rows = np.flatnonzero(visibility)
    sanity_rows = visible_rows[: min(8, len(visible_rows))]
    semantic_probe_sanity = {"tested_rows": int(len(sanity_rows)), "max_semantic": 0.0, "sum_semantic": 0.0}
    if len(sanity_rows):
        semantics = torch.zeros((pc.get_xyz.shape[0], len(sanity_rows)), device=pc.get_xyz.device, dtype=torch.float32)
        semantics[torch.as_tensor(sanity_rows, device=pc.get_xyz.device, dtype=torch.long), torch.arange(len(sanity_rows), device=pc.get_xyz.device)] = 1.0
        with torch.no_grad():
            sem = render_custom(camera, pc, semantics=semantics)["semantic"]
        sem_np = to_numpy(sem)
        semantic_probe_sanity = {
            "tested_rows": int(len(sanity_rows)),
            "max_semantic": float(np.max(sem_np)) if sem_np.size else 0.0,
            "sum_semantic": float(np.sum(sem_np)) if sem_np.size else 0.0,
        }
        del semantics, sem

    stable_ids, model_names, model_local = build_stable_ids(pc, len(radii))
    pool = candidate_pool_from_pixels(centers, radii, visibility, pixels, args)
    old_pool = old_v0_pool(args.v0_dir, stem, args.old_v0_pool_size)
    if old_pool.size:
        pool = np.unique(np.concatenate([pool, old_pool]))
    timings["candidate_pool_sec"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    contribution_backend = "semantic"
    cuda_data = None
    if args.contribution_backend in ("auto", "cuda") and cuda_contribution_available():
        cuda_data = capture_contributions_cuda(camera, pc, pixels, args.top_k_per_pixel)
        contribution_backend = "cuda"
    elif args.contribution_backend == "cuda":
        raise RuntimeError("CUDA contribution dump requested but rasterize_gaussians_contrib is not available. Rebuild diff_gaussian_rasterization.")
    if contribution_backend == "semantic":
        weights = capture_contributions(camera, pc, pool, pixels, args) if len(pool) else np.zeros((len(pixels), 0), dtype=np.float32)
    else:
        weights = cuda_data["weight"]
    timings["contribution_capture_sec"] = time.perf_counter() - t2
    if contribution_backend == "semantic":
        timings["semantic_probe_sec"] = timings["contribution_capture_sec"]
    else:
        timings["cuda_dump_sec"] = timings["contribution_capture_sec"]
    t3 = time.perf_counter()
    if contribution_backend == "cuda":
        per_pixel, contribution_rows = summarize_cuda_contributions(stable_ids, model_names, model_local, pixels, cuda_data, error_map)
        cuda_raster_sanity = compute_cuda_raster_sanity(cuda_data, pixels, original_acc, original_depth)
        cuda_semantic_shared_check = compare_cuda_with_semantic_shared(camera, pc, pixels, cuda_data, args)
        semantic_pool_gap = analyze_semantic_pool_gap(pool, cuda_data, stable_ids)
    else:
        per_pixel, contribution_rows = summarize_contributions(
            pool, stable_ids, model_names, model_local, depths, pixels, weights, error_map, args.top_k_per_pixel
        )
        cuda_raster_sanity = {"available": False, "reason": "semantic backend used"}
        cuda_semantic_shared_check = {"available": False, "reason": "semantic backend used"}
        semantic_pool_gap = {"available": False, "reason": "semantic backend used"}

    top_view_rows = []
    gid_to_row = {int(gid): int(row) for row, gid in enumerate(stable_ids)}
    for item in contribution_rows[: args.counterfactual_top_k]:
        row = gid_to_row.get(int(item["gaussian_id"]))
        if row is not None:
            top_view_rows.append(row)
    lidar_depth, lidar_mask = load_lidar(os.path.join(args.lidar_dir, f"{stem}.npy"), error_map.shape)
    counterfactual = counterfactual_probe(
        camera,
        pc,
        top_view_rows,
        pixels,
        lidar_depth,
        lidar_mask,
        original_depth,
        original_rgb,
        base_rgb,
        args,
        out_dir=out_dir,
        bbox=bbox,
    )
    for item in counterfactual:
        gid = int(stable_ids[item["view_local_index"]])
        item["gaussian_id"] = gid
        item["model_name"] = str(model_names[item["view_local_index"]])
    timings["counterfactual_sec"] = time.perf_counter() - t3
    timings["total_sec"] = time.perf_counter() - t0

    np.savez_compressed(
        os.path.join(out_dir, "contribution_responsibility_debug.npz"),
        selected_pixels=pixels,
        candidate_view_local_indices=pool,
        candidate_gaussian_ids=stable_ids[pool] if len(pool) else np.zeros(0, dtype=np.int64),
        contribution_weights=weights,
        cuda_contribution_ids=cuda_data["ids"] if cuda_data is not None else np.zeros((0, 0), dtype=np.int64),
        cuda_alpha=cuda_data["alpha"] if cuda_data is not None else np.zeros((0, 0), dtype=np.float32),
        cuda_transmittance=cuda_data["transmittance"] if cuda_data is not None else np.zeros((0, 0), dtype=np.float32),
        cuda_depth=cuda_data["depth"] if cuda_data is not None else np.zeros((0, 0), dtype=np.float32),
        cuda_depth_order=cuda_data["depth_order"] if cuda_data is not None else np.zeros((0, 0), dtype=np.int32),
        geometry_errors=error_map[pixels[:, 1], pixels[:, 0]],
    )
    with open(os.path.join(out_dir, "per_pixel_topk_contributors.json"), "w", encoding="utf-8") as f:
        json.dump(per_pixel, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "top_contribution_gaussians.json"), "w", encoding="utf-8") as f:
        json.dump(contribution_rows[: args.top_gaussians], f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "counterfactual_probe.json"), "w", encoding="utf-8") as f:
        json.dump(counterfactual, f, indent=2, ensure_ascii=False)
    if args.save_visuals:
        save_overlay(os.path.join(out_dir, "selected_high_error_pixels.png"), camera, pixels)
        if bbox is not None:
            save_region_contact_sheet(os.path.join(out_dir, "region_contact_sheet.png"), original_rgb, base_rgb, error_map, pixels, bbox)

    summary = {
        "stem": stem,
        "region_id": region_id,
        "region_type": region.get("region_type"),
        "input_region": region,
        "status": "ok",
        "selected_pixel_count": int(len(pixels)),
        "pixel_bbox": bbox,
        "pixel_source": args.pixel_source,
        "selected_pixel_acc": {
            "mean": float(np.mean(original_acc[pixels[:, 1], pixels[:, 0]])) if len(pixels) else None,
            "max": float(np.max(original_acc[pixels[:, 1], pixels[:, 0]])) if len(pixels) else None,
            "min": float(np.min(original_acc[pixels[:, 1], pixels[:, 0]])) if len(pixels) else None,
        },
        "candidate_pool_count": int(len(pool)),
        "contribution_backend": contribution_backend,
        "cuda_contribution_available": bool(cuda_contribution_available()),
        "nonzero_contribution_gaussian_count": int(len(contribution_rows)),
        "top_contribution_gaussians": contribution_rows[:10],
        "counterfactual": counterfactual,
        "counterfactual_label_counts": dict(
            (label, int(sum(1 for item in counterfactual if item.get("counterfactual_label") == label)))
            for label in ["bad_contributor", "good_contributor", "neutral_contributor"]
        ),
        "old_v0_comparison": compare_with_old_v0(args.v0_dir, stem, contribution_rows, stable_ids),
        "semantic_probe_sanity": semantic_probe_sanity,
        "cuda_raster_sanity": cuda_raster_sanity,
        "cuda_semantic_shared_weight_check": cuda_semantic_shared_check,
        "semantic_pool_gap_analysis": semantic_pool_gap,
        "id_mapping": {
            "cuda_row_id": "Row index in the concatenated StreetGaussianModel tensor used by the rasterizer for this render call.",
            "view_local_index": "Alias of cuda_row_id in this debug script; it is render-call local and should not be used for multi-view aggregation.",
            "stable_gaussian_id": "Namespace-stabilized id: stable_namespace_id(model_name) * 10000000 + model_local_index.",
            "model_namespace": "model_name from pc.graph_gaussian_range, e.g. background or obj_<track_id>.",
            "aggregation_rule": "Use stable_gaussian_id plus model_name/model_local_index for cross-view summaries; do not aggregate legacy view-local ids.",
        },
        "timings_sec": timings,
        "method": {
            "contribution_capture": (
                "Uses debug-only CUDA selected-pixel rasterizer dump. For each selected pixel, "
                "the rasterizer scans the sorted per-tile Gaussian list and records top-K ids, "
                "alpha, transmittance, T*alpha contribution weight, depth, and depth order."
                if contribution_backend == "cuda"
                else "Uses rasterizer semantic channels as one-hot Gaussian probes. Returned semantic value equals the original rasterizer alpha*T contribution weight for each probed Gaussian."
            ),
            "candidate_pool": (
                "Only used for old V0 comparison / semantic fallback; CUDA responsibility scores are computed from selected-pixel rasterizer contributions, not screen-space overlap."
                if contribution_backend == "cuda"
                else "A loose screen-space pool is used only to limit debug cost; responsibility scores use real rasterizer contributions, not overlap."
            ),
            "responsibility_formula": "R_i = sum_p (T_i(p) * alpha_i(p) * E_geo(p)) over selected high-error LiDAR-valid pixels.",
            "counterfactual_formula": "C_i = E_with_i - E_with_weakened_i over selected pixels; positive means weakening reduced error.",
            "depth_order": "For CUDA backend, depth_order is the Gaussian position in the rasterizer's sorted per-tile traversal for that pixel. It is the alpha-compositing traversal order used by this rasterizer, but it should be interpreted as rasterizer order rather than a semantic object/layer id.",
        },
        "paths": {
            "npz": os.path.join(out_dir, "contribution_responsibility_debug.npz"),
            "per_pixel_topk": os.path.join(out_dir, "per_pixel_topk_contributors.json"),
            "top_gaussians": os.path.join(out_dir, "top_contribution_gaussians.json"),
            "counterfactual": os.path.join(out_dir, "counterfactual_probe.json"),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    cfg.render.save_image = False
    cfg.render.save_video = False
    safe_state(cfg.eval.quiet)
    ensure_dir(SCRIPT_ARGS.output_dir)
    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        cameras = scene.getTrainCameras()
        camera_by_name = {camera.image_name: camera for camera in cameras}
        regions = load_region_specs(SCRIPT_ARGS)
        summaries = []
        for region in regions:
            stem = region["view_id"]
            camera = camera_by_name.get(stem)
            if camera is None:
                print(f"Skipping {stem}: camera not found")
                continue
            print(f"Contribution debug for {stem} region {region.get('region_id', 'manual')}")
            summaries.append(run_region(region, camera, gaussians, SCRIPT_ARGS.output_dir, SCRIPT_ARGS))
    feedback_signal = write_training_feedback_signal(SCRIPT_ARGS.output_dir, summaries)
    aggregate = {
        "views": [s["stem"] for s in summaries],
        "regions": [f"{s.get('stem')}:region{s.get('region_id')}" for s in summaries],
        "view_count": len(summaries),
        "ok_region_count": int(sum(1 for s in summaries if s.get("status") == "ok")),
        "low_evidence_region_count": int(sum(1 for s in summaries if s.get("status") == "low_evidence")),
        "cuda_backend_region_count": int(sum(1 for s in summaries if s.get("contribution_backend") == "cuda")),
        "semantic_backend_region_count": int(sum(1 for s in summaries if s.get("contribution_backend") == "semantic")),
        "total_nonzero_contribution_gaussians": int(sum(s.get("nonzero_contribution_gaussian_count", 0) for s in summaries)),
        "total_bad_contributors": int(sum(s.get("counterfactual_label_counts", {}).get("bad_contributor", 0) for s in summaries)),
        "total_good_contributors": int(sum(s.get("counterfactual_label_counts", {}).get("good_contributor", 0) for s in summaries)),
        "total_neutral_contributors": int(sum(s.get("counterfactual_label_counts", {}).get("neutral_contributor", 0) for s in summaries)),
        "mean_old_new_top100_overlap": float(
            np.mean(
                [
                    s.get("old_v0_comparison", {}).get("top100_id_overlap_count", 0)
                    / max(s.get("old_v0_comparison", {}).get("new_top100_count", 1), 1)
                    for s in summaries
                    if s.get("status") == "ok"
                ]
            )
        )
        if any(s.get("status") == "ok" for s in summaries)
        else None,
        "timings_sec": {
            "total": float(sum(s.get("timings_sec", {}).get("total_sec", 0.0) for s in summaries)),
            "contribution_capture": float(sum(s.get("timings_sec", {}).get("contribution_capture_sec", 0.0) for s in summaries)),
            "cuda_dump": float(sum(s.get("timings_sec", {}).get("cuda_dump_sec", 0.0) for s in summaries)),
            "semantic_probe": float(sum(s.get("timings_sec", {}).get("semantic_probe_sec", 0.0) for s in summaries)),
            "counterfactual": float(sum(s.get("timings_sec", {}).get("counterfactual_sec", 0.0) for s in summaries)),
        },
        "debug_only": True,
        "selected_pixel_contribution_dump": (
            "Debug-only sparse query. CUDA backend records top-K Gaussian ids, alpha, transmittance, "
            "T*alpha contribution weight, depth, and depth order for selected high-error pixels. "
            "Semantic one-hot probe remains available as fallback."
        ),
        "guided_training_feedback_signal": feedback_signal,
    }
    with open(os.path.join(SCRIPT_ARGS.output_dir, "contribution_responsibility_all_views_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"aggregate": aggregate, "frames": summaries}, f, indent=2, ensure_ascii=False)
    print(f"Saved contribution debug outputs under: {SCRIPT_ARGS.output_dir}")


if __name__ == "__main__":
    main()
