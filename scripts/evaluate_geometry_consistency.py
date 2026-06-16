#!/usr/bin/env python3
"""Geometry consistency evaluation for final GeoFeedback-GS/StreetGS checkpoints.

This script is independent from training.  It loads trained checkpoints through
the StreetGS evaluation path, renders held-out views, and compares expected
rendered depth against held-out geometry references.

Implemented scopes:
- full_image
- object_region, using camera.guidance["obj_bound"] when available
- background_region
- selected_region, using feedback selected/risk masks when available

Depth convention:
StreetGS rasterizer returns accumulated depth and alpha/accumulation.  The
default evaluated depth is expected depth: raw_depth / (acc + eps).  The raw
accumulated depth can be selected with --depth-mode raw for debugging.
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


SCOPES = ["full_image", "object_region", "background_region", "selected_region"]
METRICS = [
    "lidar_absrel",
    "lidar_rmse",
    "lidar_mae",
    "lidar_delta1",
    "da3_absrel_aligned",
    "da3_rmse_aligned",
    "da3_mae_aligned",
    "da3_spearman",
    "da3_order_accuracy",
    "depth_edge_f1_rgb",
    "edge_precision_rgb",
    "edge_recall_rgb",
    "edge_chamfer_rgb",
    "depth_edge_f1_da3",
    "edge_precision_da3",
    "edge_recall_da3",
    "edge_chamfer_da3",
    "object_boundary_depth_jump_consistency",
]
PER_VIEW_FIELDS = [
    "experiment",
    "split",
    "scope",
    "view_index",
    "cam_id",
    "frame",
    "frame_idx",
    "image_name",
    "depth_mode",
    "rendered_depth_type",
    "mask_type",
    "scope_pixel_count",
    "lidar_valid_pixel_count",
    "da3_valid_pixel_count",
    "edge_valid_pixel_count",
    "selected_pixel_count",
    "responsible_gaussian_group_count",
    "status",
    "warning",
    *METRICS,
]
SUMMARY_FIELDS = ["experiment", "scope", "split", "view_count"]
for metric in METRICS:
    SUMMARY_FIELDS.extend([
        f"{metric}_mean",
        f"{metric}_median",
        f"{metric}_std",
        f"{metric}_min",
        f"{metric}_max",
        f"{metric}_valid_count",
    ])


def _deep_merge(base, override):
    result = dict(base or {})
    for key, value in (override or {}).items():
        if key == "_BASE_":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_merged_config(config_path):
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        current = yaml.safe_load(f) or {}
    base_ref = current.get("_BASE_")
    if not base_ref:
        return current
    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = (config_path.parent / base_path).resolve()
    return _deep_merge(_load_merged_config(base_path), current)


def _materialize_config(repo_root, config_path, out_dir, loaded_iter):
    payload = _load_merged_config(config_path)
    payload["workspace"] = str(repo_root)
    payload["mode"] = "evaluate"
    payload["loaded_iter"] = int(loaded_iter)
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        payload["gpus"] = [-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(config_path).stem}_geometry_eval.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return out_path, payload


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _append_csv(path, row, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()


def _read_csv(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _summarize(rows, experiment, scope, split):
    subset = [
        r for r in rows
        if r.get("scope") == scope and r.get("split") == split and r.get("status") in {"valid", "partial"}
    ]
    out = {"experiment": experiment, "scope": scope, "split": split, "view_count": len(subset)}
    for metric in METRICS:
        values = [_to_float(r.get(metric)) for r in subset]
        values = [v for v in values if v is not None]
        out[f"{metric}_valid_count"] = len(values)
        if not values:
            for stat in ["mean", "median", "std", "min", "max"]:
                out[f"{metric}_{stat}"] = ""
            continue
        values_sorted = sorted(values)
        mid = len(values_sorted) // 2
        mean = sum(values) / len(values)
        median = values_sorted[mid] if len(values_sorted) % 2 else 0.5 * (values_sorted[mid - 1] + values_sorted[mid])
        out[f"{metric}_mean"] = mean
        out[f"{metric}_median"] = median
        out[f"{metric}_std"] = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        out[f"{metric}_min"] = min(values)
        out[f"{metric}_max"] = max(values)
    return out


def _np_resize(arr, shape, interpolation):
    import cv2
    import numpy as np

    arr = np.asarray(arr)
    if arr.shape == shape:
        return arr
    return cv2.resize(arr, (shape[1], shape[0]), interpolation=interpolation)


def _tensor_to_np(tensor):
    return tensor.detach().float().cpu().numpy().squeeze()


def _load_lidar_depth(source_path, image_name, shape):
    import cv2
    import numpy as np

    path = Path(source_path) / "lidar_depth" / f"{image_name}.npy"
    if not path.exists():
        return None, None, f"missing_lidar:{path}"
    try:
        payload = np.load(path, allow_pickle=True)
        if hasattr(payload, "item"):
            payload = payload.item()
        mask = np.asarray(payload["mask"]).astype(bool)
        value = np.asarray(payload["value"]).reshape(-1).astype(np.float32)
        depth = np.zeros(mask.shape, dtype=np.float32)
        if int(mask.sum()) != int(value.size):
            value = value[: int(mask.sum())]
        depth[mask] = value
        if depth.shape != shape:
            depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        return depth, mask, ""
    except Exception as exc:
        return None, None, f"lidar_load_failed:{exc}"


def _safe_metrics(pred, gt, mask):
    import numpy as np

    valid = mask & np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    count = int(valid.sum())
    if count == 0:
        return {}, 0
    p = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)
    diff = p - g
    ratio = np.maximum(p / np.maximum(g, 1e-8), g / np.maximum(p, 1e-8))
    return {
        "absrel": float(np.mean(np.abs(diff) / np.maximum(g, 1e-8))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "delta1": float(np.mean(ratio < 1.25)),
    }, count


def _fit_scale_shift(reference, target, mask):
    """Fit aligned = s * target + b to reference by least squares."""
    import numpy as np

    valid = mask & np.isfinite(reference) & np.isfinite(target) & (reference > 0) & (target > 0)
    if int(valid.sum()) < 8:
        return None, None, valid
    x = target[valid].reshape(-1).astype(np.float64)
    y = reference[valid].reshape(-1).astype(np.float64)
    a = np.stack([x, np.ones_like(x)], axis=1)
    try:
        s, b = np.linalg.lstsq(a, y, rcond=None)[0]
    except Exception:
        return None, None, valid
    return float(s), float(b), valid


def _rankdata(values):
    import numpy as np

    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(a, b, mask, max_samples=20000, seed=13):
    import numpy as np

    valid = mask & np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    idx = np.flatnonzero(valid.reshape(-1))
    if idx.size < 8:
        return ""
    rng = np.random.default_rng(seed)
    if idx.size > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    av = a.reshape(-1)[idx].astype(np.float64)
    bv = b.reshape(-1)[idx].astype(np.float64)
    ar = _rankdata(av)
    br = _rankdata(bv)
    ar -= ar.mean()
    br -= br.mean()
    denom = np.sqrt(np.sum(ar ** 2) * np.sum(br ** 2))
    if denom <= 1e-12:
        return ""
    return float(np.sum(ar * br) / denom)


def _order_accuracy(a, b, mask, pairs=10000, seed=17):
    import numpy as np

    valid = mask & np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    idx = np.flatnonzero(valid.reshape(-1))
    if idx.size < 16:
        return ""
    rng = np.random.default_rng(seed)
    n = min(pairs, idx.size * 2)
    i1 = rng.choice(idx, size=n, replace=True)
    i2 = rng.choice(idx, size=n, replace=True)
    av = a.reshape(-1)
    bv = b.reshape(-1)
    da = av[i1] - av[i2]
    db = bv[i1] - bv[i2]
    keep = (np.abs(da) > 1e-6) & (np.abs(db) > 1e-6)
    if int(keep.sum()) == 0:
        return ""
    return float(np.mean(np.sign(da[keep]) == np.sign(db[keep])))


def _edge_map(arr, mask=None, quantile=0.88):
    import cv2
    import numpy as np

    x = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(x)
    if mask is not None:
        valid &= mask.astype(bool)
    if int(valid.sum()) < 8:
        return np.zeros(x.shape, dtype=bool)
    lo, hi = np.percentile(x[valid], [2, 98])
    xn = np.zeros_like(x, dtype=np.float32)
    denom = max(float(hi - lo), 1e-6)
    xn[valid] = np.clip((x[valid] - lo) / denom, 0.0, 1.0)
    gx = cv2.Sobel(xn, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(xn, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    threshold = np.quantile(mag[valid], quantile)
    return (mag >= threshold) & valid


def _rgb_edge(rgb):
    import cv2
    import numpy as np

    img = np.asarray(rgb)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 60, 150).astype(bool)


def _edge_metrics(pred_edge, ref_edge, valid_mask, tolerance=2):
    import cv2
    import numpy as np

    pred = pred_edge.astype(bool) & valid_mask.astype(bool)
    ref = ref_edge.astype(bool) & valid_mask.astype(bool)
    if int(pred.sum()) == 0 or int(ref.sum()) == 0:
        return {"f1": "", "precision": "", "recall": "", "chamfer": ""}, int(valid_mask.sum())
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=np.uint8)
    ref_dil = cv2.dilate(ref.astype(np.uint8), kernel).astype(bool)
    pred_dil = cv2.dilate(pred.astype(np.uint8), kernel).astype(bool)
    precision = float((pred & ref_dil).sum() / max(pred.sum(), 1))
    recall = float((ref & pred_dil).sum() / max(ref.sum(), 1))
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    dist_to_ref = cv2.distanceTransform((~ref).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_pred = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, 3)
    chamfer = 0.5 * (float(dist_to_ref[pred].mean()) + float(dist_to_pred[ref].mean()))
    return {"f1": f1, "precision": precision, "recall": recall, "chamfer": chamfer}, int(valid_mask.sum())


def _boundary_jump_consistency(depth, obj_mask):
    import cv2
    import numpy as np

    mask = obj_mask.astype(bool)
    if int(mask.sum()) == 0:
        return ""
    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)
    if int(boundary.sum()) == 0:
        return ""
    edge = _edge_map(depth, np.isfinite(depth) & (depth > 0), quantile=0.85)
    edge_dil = cv2.dilate(edge.astype(np.uint8), kernel).astype(bool)
    return float((boundary & edge_dil).sum() / max(boundary.sum(), 1))


def _visualize_depth(depth, mask=None):
    import cv2
    import numpy as np

    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    if mask is not None:
        valid &= mask.astype(bool)
    out = np.zeros((*d.shape, 3), dtype=np.uint8)
    if int(valid.sum()) == 0:
        return out
    lo, hi = np.percentile(d[valid], [2, 98])
    norm = np.zeros_like(d, dtype=np.float32)
    norm[valid] = np.clip((d[valid] - lo) / max(float(hi - lo), 1e-6), 0, 1)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[valid] = color[valid]
    return out


def _save_panel(path, items):
    import cv2
    import numpy as np

    tiles = []
    for title, img in items:
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if arr.shape[-1] == 3:
            tile = arr.copy()
        else:
            tile = arr[..., :3].copy()
        if tile.dtype != np.uint8:
            tile = np.clip(tile, 0, 255).astype(np.uint8)
        cv2.putText(tile, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(tile, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    h = min(t.shape[0] for t in tiles)
    tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    panel = np.concatenate(tiles, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)


def _make_feedback_index(repo_root, model_path):
    import numpy as np

    root = Path(model_path or "")
    if root and not root.is_absolute():
        root = repo_root / root
    root = root / "feedback_controller"
    latest_mask = None
    selected_count = ""
    group_count = ""
    if not root.exists():
        return {"mask": None, "selected_pixel_count": "", "responsible_gaussian_group_count": ""}
    for iter_dir in sorted(root.glob("iter_*")):
        manifest_candidates = list(iter_dir.glob("*manifest*.json")) + list(iter_dir.glob("*summary*.json")) + list(iter_dir.glob("*audit*.json"))
        for path in manifest_candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            selected_count = payload.get("selected_pixel_count", payload.get("selected_pixels", selected_count))
            group_count = payload.get("responsible_gaussian_group_count", payload.get("responsible_group_count", group_count))
        npy_candidates = list(iter_dir.rglob("*risk*.npy")) + list(iter_dir.rglob("*mask*.npy"))
        if npy_candidates:
            try:
                arr = np.load(npy_candidates[-1], allow_pickle=True)
                latest_mask = np.asarray(arr).squeeze()
            except Exception:
                pass
    if latest_mask is not None:
        latest_mask = latest_mask > 0
    return {
        "mask": latest_mask,
        "selected_pixel_count": selected_count,
        "responsible_gaussian_group_count": group_count,
    }


def _init_da3_bridge(enable_da3, payload, torch):
    if not enable_da3:
        return None
    from lib.geovit.depth_anything_bridge import GeoViTDepthBridge

    geovit = payload.get("geovit", {}) or {}
    model_name = geovit.get("model_dir") or geovit.get("model_name")
    kwargs = {}
    if geovit.get("local_files_only", False):
        kwargs["local_files_only"] = True
    bridge = GeoViTDepthBridge(
        model_name=model_name,
        device=torch.device("cuda"),
        use_cache=True,
        include_confidence=True,
        **kwargs,
    )
    return bridge.eval()


def _geometry_rows_for_view(
    experiment,
    split,
    idx,
    camera,
    rendered_depth,
    acc,
    rgb,
    gt_rgb,
    obj_mask,
    lidar_depth,
    lidar_mask,
    da3_depth,
    selected_mask,
    feedback_meta,
    args,
):
    import cv2
    import numpy as np

    h, w = rendered_depth.shape
    full_mask = np.isfinite(rendered_depth) & (rendered_depth > 0) & np.isfinite(acc) & (acc > args.min_acc)
    if lidar_mask is not None:
        lidar_mask = _np_resize(lidar_mask.astype(np.uint8), (h, w), cv2.INTER_NEAREST).astype(bool)
    if lidar_depth is not None:
        lidar_depth = _np_resize(lidar_depth, (h, w), cv2.INTER_NEAREST)
    obj_mask = _np_resize(obj_mask.astype(np.uint8), (h, w), cv2.INTER_NEAREST).astype(bool)
    bg_mask = full_mask & (~obj_mask)
    if selected_mask is not None:
        selected_mask = _np_resize(selected_mask.astype(np.uint8), (h, w), cv2.INTER_NEAREST).astype(bool)
    else:
        selected_mask = np.zeros((h, w), dtype=bool)

    da3_aligned = None
    da3_valid_mask = np.zeros((h, w), dtype=bool)
    if da3_depth is not None:
        da3_depth = _np_resize(da3_depth, (h, w), cv2.INTER_LINEAR)
        s, b, da3_valid_mask = _fit_scale_shift(rendered_depth, da3_depth, full_mask)
        if s is not None:
            da3_aligned = s * da3_depth + b

    rgb_ref_edge = _rgb_edge(gt_rgb)
    depth_edge = _edge_map(rendered_depth, full_mask)
    da3_edge = _edge_map(da3_aligned, da3_valid_mask) if da3_aligned is not None else None

    scopes = {
        "full_image": ("full_image_valid_acc", full_mask),
        "object_region": ("object_box_or_dynamic_mask", full_mask & obj_mask),
        "background_region": ("background_minus_object_mask", bg_mask),
        "selected_region": ("feedback_selected_or_risk_region", full_mask & selected_mask),
    }
    rows = []
    for scope, (mask_type, scope_mask) in scopes.items():
        warning = []
        row = {
            "experiment": experiment,
            "split": split,
            "scope": scope,
            "view_index": idx,
            "cam_id": camera.meta.get("cam", ""),
            "frame": camera.meta.get("frame", ""),
            "frame_idx": camera.meta.get("frame_idx", ""),
            "image_name": getattr(camera, "image_name", ""),
            "depth_mode": args.depth_mode,
            "rendered_depth_type": "expected_depth=raw_depth/(acc+eps)" if args.depth_mode == "expected" else "raw_accumulated_depth",
            "mask_type": mask_type,
            "scope_pixel_count": int(scope_mask.sum()),
            "lidar_valid_pixel_count": 0,
            "da3_valid_pixel_count": 0,
            "edge_valid_pixel_count": int(scope_mask.sum()),
            "selected_pixel_count": feedback_meta.get("selected_pixel_count", "") if scope == "selected_region" else "",
            "responsible_gaussian_group_count": feedback_meta.get("responsible_gaussian_group_count", "") if scope == "selected_region" else "",
            "status": "valid" if int(scope_mask.sum()) > 0 else "not_applicable",
            "warning": "",
        }
        if int(scope_mask.sum()) == 0:
            row["warning"] = "empty_scope_mask"
            rows.append(row)
            continue

        if lidar_depth is not None and lidar_mask is not None:
            lidar_metrics, count = _safe_metrics(rendered_depth, lidar_depth, scope_mask & lidar_mask)
            row["lidar_valid_pixel_count"] = count
            if count > 0:
                row["lidar_absrel"] = lidar_metrics["absrel"]
                row["lidar_rmse"] = lidar_metrics["rmse"]
                row["lidar_mae"] = lidar_metrics["mae"]
                row["lidar_delta1"] = lidar_metrics["delta1"]
            else:
                warning.append("empty_lidar_valid_mask")
        else:
            warning.append("missing_lidar_depth")

        if da3_aligned is not None:
            da3_metrics, count = _safe_metrics(rendered_depth, da3_aligned, scope_mask & da3_valid_mask)
            row["da3_valid_pixel_count"] = count
            if count > 0:
                row["da3_absrel_aligned"] = da3_metrics["absrel"]
                row["da3_rmse_aligned"] = da3_metrics["rmse"]
                row["da3_mae_aligned"] = da3_metrics["mae"]
                row["da3_spearman"] = _spearman(rendered_depth, da3_aligned, scope_mask & da3_valid_mask)
                row["da3_order_accuracy"] = _order_accuracy(rendered_depth, da3_aligned, scope_mask & da3_valid_mask)
            else:
                warning.append("empty_da3_valid_mask")
        elif args.enable_da3:
            warning.append("da3_failed_or_missing")

        edge_rgb, edge_count = _edge_metrics(depth_edge, rgb_ref_edge, scope_mask)
        row["edge_valid_pixel_count"] = edge_count
        row["depth_edge_f1_rgb"] = edge_rgb["f1"]
        row["edge_precision_rgb"] = edge_rgb["precision"]
        row["edge_recall_rgb"] = edge_rgb["recall"]
        row["edge_chamfer_rgb"] = edge_rgb["chamfer"]
        if da3_edge is not None:
            edge_da3, _ = _edge_metrics(depth_edge, da3_edge, scope_mask & da3_valid_mask)
            row["depth_edge_f1_da3"] = edge_da3["f1"]
            row["edge_precision_da3"] = edge_da3["precision"]
            row["edge_recall_da3"] = edge_da3["recall"]
            row["edge_chamfer_da3"] = edge_da3["chamfer"]

        if scope == "object_region":
            row["object_boundary_depth_jump_consistency"] = _boundary_jump_consistency(rendered_depth, obj_mask)
        if warning:
            row["warning"] = ";".join(warning)
            if row["status"] == "valid":
                row["status"] = "partial"
        rows.append(row)
    return rows, {
        "rendered_depth": rendered_depth,
        "lidar_depth": lidar_depth,
        "lidar_mask": lidar_mask,
        "da3_aligned": da3_aligned,
        "depth_edge": depth_edge,
        "rgb_edge": rgb_ref_edge,
        "object_mask": obj_mask,
        "selected_mask": selected_mask,
    }


def evaluate_one(repo_root, config_path, output_root, args):
    streetgs_root = repo_root / "third_party" / "street_gaussian"
    sys.path.insert(0, str(streetgs_root))
    sys.path.insert(0, str(repo_root))
    os.chdir(streetgs_root)

    exp_name = Path(config_path).stem
    exp_out = output_root / exp_name
    if args.overwrite and exp_out.exists():
        shutil.rmtree(exp_out)
    materialized, payload = _materialize_config(repo_root, config_path, exp_out / "configs", args.loaded_iter)
    sys.argv = ["evaluate_geometry_consistency.py", "--config", str(materialized)]

    import numpy as np
    import torch
    from lib.datasets.dataset import Dataset
    from lib.models.scene import Scene
    from lib.models.street_gaussian_model import StreetGaussianModel
    from lib.models.street_gaussian_renderer import StreetGaussianRenderer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for checkpoint geometry rendering.")

    source_path = payload.get("source_path") or payload.get("data", {}).get("source_path") or "data/waymo/002"
    source_path = (repo_root / source_path).resolve() if not Path(source_path).is_absolute() else Path(source_path)
    feedback_meta = _make_feedback_index(repo_root, payload.get("model_path", ""))

    rows_all = []
    split_names = list(dict.fromkeys(args.splits))
    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        da3_bridge = _init_da3_bridge(args.enable_da3, payload, torch)

        split_cameras = {}
        if "test" in split_names:
            split_cameras["test"] = scene.getTestCameras()
        if "train" in split_names:
            split_cameras["train"] = scene.getTrainCameras()

        print(
            f"[GeometryEval] experiment={exp_name} loaded_iter={args.loaded_iter} "
            f"depth_mode={args.depth_mode} enable_da3={args.enable_da3}",
            flush=True,
        )
        for split, cameras in split_cameras.items():
            if args.max_views_per_split > 0:
                cameras = cameras[: args.max_views_per_split]
            print(f"[GeometryEval] split={split} views={len(cameras)}", flush=True)
            try:
                from tqdm import tqdm
                iterator = tqdm(cameras, desc=f"[GeometryEval] {exp_name} {split}", unit="view")
            except Exception:
                iterator = cameras

            for idx, camera in enumerate(iterator):
                render_pkg = renderer.render(camera, gaussians)
                raw_depth = _tensor_to_np(render_pkg["depth"])
                acc = _tensor_to_np(render_pkg["acc"])
                rendered_depth = raw_depth / np.maximum(acc, args.depth_eps) if args.depth_mode == "expected" else raw_depth
                rgb = _tensor_to_np(torch.clamp(render_pkg["rgb"], 0.0, 1.0))
                gt_rgb = _tensor_to_np(torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0))
                obj = camera.guidance.get("obj_bound")
                obj_mask = _tensor_to_np(obj.to("cuda")).astype(bool) if obj is not None else np.zeros(rendered_depth.shape, dtype=bool)

                image_name = getattr(camera, "image_name", "")
                lidar_depth, lidar_mask, lidar_warning = _load_lidar_depth(source_path, image_name, rendered_depth.shape)

                da3_depth = None
                if da3_bridge is not None:
                    try:
                        da3 = da3_bridge(camera)
                        da3_depth = _tensor_to_np(da3["relative_depth"])
                    except Exception as exc:
                        if idx == 0:
                            print(f"[GeometryEval][WARN] DA3 inference failed: {exc}", flush=True)

                selected_mask = feedback_meta.get("mask")
                rows, debug = _geometry_rows_for_view(
                    exp_name,
                    split,
                    idx,
                    camera,
                    rendered_depth,
                    acc,
                    rgb,
                    gt_rgb,
                    obj_mask,
                    lidar_depth,
                    lidar_mask,
                    da3_depth,
                    selected_mask,
                    feedback_meta,
                    args,
                )
                if lidar_warning:
                    for row in rows:
                        row["warning"] = (row["warning"] + ";" if row["warning"] else "") + lidar_warning
                for row in rows:
                    _append_csv(exp_out / "per_view_geometry_metrics.csv", row, PER_VIEW_FIELDS)
                    rows_all.append(row)

                if idx < args.max_panels_per_split:
                    gt_img = np.transpose(gt_rgb, (1, 2, 0)) if gt_rgb.shape[0] == 3 else gt_rgb
                    rd_vis = _visualize_depth(debug["rendered_depth"])
                    lidar_vis = _visualize_depth(debug["lidar_depth"], debug["lidar_mask"]) if debug["lidar_depth"] is not None else np.zeros_like(rd_vis)
                    da3_vis = _visualize_depth(debug["da3_aligned"]) if debug["da3_aligned"] is not None else np.zeros_like(rd_vis)
                    err = np.zeros_like(debug["rendered_depth"], dtype=np.float32)
                    if debug["lidar_depth"] is not None and debug["lidar_mask"] is not None:
                        valid = debug["lidar_mask"] & (debug["lidar_depth"] > 0) & (debug["rendered_depth"] > 0)
                        err[valid] = np.abs(debug["rendered_depth"][valid] - debug["lidar_depth"][valid])
                    err_vis = _visualize_depth(err)
                    edge_vis = (debug["depth_edge"].astype(np.uint8) * 255)
                    obj_vis = (debug["object_mask"].astype(np.uint8) * 255)
                    panel_path = exp_out / "visualization_panels" / split / f"{image_name}_geometry_panel.jpg"
                    _save_panel(panel_path, [
                        ("GT RGB", (np.clip(gt_img, 0, 1) * 255).astype(np.uint8)[..., ::-1]),
                        ("Rendered Depth", rd_vis),
                        ("LiDAR Sparse", lidar_vis),
                        ("DA3 Aligned", da3_vis),
                        ("Depth Error", err_vis),
                        ("Depth Edge", edge_vis),
                        ("Object Mask", obj_vis),
                    ])

                if (idx + 1) % 10 == 0 or idx + 1 == len(cameras):
                    print(f"[GeometryEval] progress {exp_name} {split}: {idx + 1}/{len(cameras)}", flush=True)

    summaries = [_summarize(rows_all, exp_name, scope, split) for scope in SCOPES for split in split_names]
    _write_csv(exp_out / "summary_geometry_metrics.csv", summaries, SUMMARY_FIELDS)
    manifest = {
        "experiment": exp_name,
        "config": str(config_path),
        "model_path": payload.get("model_path", ""),
        "loaded_iter": int(args.loaded_iter),
        "splits": split_names,
        "depth_mode": args.depth_mode,
        "rendered_depth_type": "expected_depth=raw_depth/(acc+eps)" if args.depth_mode == "expected" else "raw_accumulated_depth",
        "source_path": str(source_path),
        "enable_da3": bool(args.enable_da3),
        "notes": [
            "LiDAR metrics are held-out evaluation references only; they do not imply LiDAR training supervision.",
            "DA3 aligned metrics use per-view scale-shift alignment and measure relative visual geometry consistency, not absolute geometry accuracy.",
            "selected_region metrics are final-local diagnostics unless paired with before/after checkpoints.",
        ],
    }
    (exp_out / "geometry_eval_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return summaries


def aggregate(output_root):
    rows = []
    for path in sorted(output_root.glob("*/summary_geometry_metrics.csv")):
        rows.extend(_read_csv(path))
    _write_csv(output_root / "compare_geometry_summary.csv", rows, SUMMARY_FIELDS)
    main = [
        row for row in rows
        if row.get("scope") in {"full_image", "object_region", "background_region", "selected_region"}
        and row.get("split") == "test"
    ]
    _write_csv(output_root / "compare_geometry_summary_test.csv", main, SUMMARY_FIELDS)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-root", default="outputs/geometry_eval")
    parser.add_argument("--loaded-iter", type=int, default=30000)
    parser.add_argument("--splits", nargs="+", choices=["test", "train"], default=["test"])
    parser.add_argument("--depth-mode", choices=["expected", "raw"], default="expected")
    parser.add_argument("--depth-eps", type=float, default=1e-6)
    parser.add_argument("--min-acc", type=float, default=0.03)
    parser.add_argument("--enable-da3", action="store_true")
    parser.add_argument("--max-panels-per-split", type=int, default=12)
    parser.add_argument("--max-views-per-split", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--single-config-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = repo_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.single_config_worker and len(args.configs) > 1:
        for config in args.configs:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--configs", config,
                "--output-root", args.output_root,
                "--loaded-iter", str(args.loaded_iter),
                "--splits", *args.splits,
                "--depth-mode", args.depth_mode,
                "--depth-eps", str(args.depth_eps),
                "--min-acc", str(args.min_acc),
                "--max-panels-per-split", str(args.max_panels_per_split),
                "--max-views-per-split", str(args.max_views_per_split),
                "--single-config-worker",
            ]
            if args.enable_da3:
                cmd.append("--enable-da3")
            if args.overwrite:
                cmd.append("--overwrite")
            ret = subprocess.call(cmd, cwd=str(repo_root))
            if ret != 0:
                raise SystemExit(ret)
        rows = aggregate(output_root)
        print(json.dumps({
            "output_root": str(output_root),
            "experiment_count": len({row.get("experiment") for row in rows}),
            "compare_geometry_summary": str(output_root / "compare_geometry_summary.csv"),
            "compare_geometry_summary_test": str(output_root / "compare_geometry_summary_test.csv"),
        }, indent=2, ensure_ascii=False), flush=True)
        return

    all_summaries = []
    for config in args.configs:
        all_summaries.extend(evaluate_one(repo_root, Path(config).resolve(), output_root, args))
    rows = aggregate(output_root)
    print(json.dumps({
        "output_root": str(output_root),
        "experiment_count": len({row.get("experiment") for row in rows}),
        "compare_geometry_summary": str(output_root / "compare_geometry_summary.csv"),
        "compare_geometry_summary_test": str(output_root / "compare_geometry_summary_test.csv"),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
