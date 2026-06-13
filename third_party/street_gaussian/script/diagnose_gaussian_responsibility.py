import argparse
import json
import os
import random
import sys
import zlib

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_script_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--geometry-error-map-dir", required=True)
    parser.add_argument("--output-dir", default="output/local_smoke/gaussian_responsibility_v0")
    parser.add_argument("--views", nargs="+", default=["000002_2", "000001_2", "000000_2"])
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--min-support-pixels", type=int, default=3)
    parser.add_argument("--window-radius-scale", type=float, default=1.0)
    parser.add_argument("--min-window-radius", type=int, default=1)
    parser.add_argument("--max-window-radius", type=int, default=60)
    parser.add_argument("--high-error-quantile", type=float, default=0.90)
    parser.add_argument("--sensitivity-top-k", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--sensitivity-radius-scale", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument("--sensitivity-max-gaussians", type=int, default=20000)
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--selected-visual-views", nargs="*", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--help-responsibility", action="store_true")
    script_args, remaining = parser.parse_known_args()
    if script_args.help_responsibility:
        parser.print_help()
        sys.exit(0)
    sys.argv = [sys.argv[0]] + remaining
    return script_args


SCRIPT_ARGS = parse_script_args()

from lib.config import cfg  # noqa: E402
from lib.datasets.dataset import Dataset  # noqa: E402
from lib.models.scene import Scene  # noqa: E402
from lib.models.street_gaussian_model import StreetGaussianModel  # noqa: E402
from lib.models.street_gaussian_renderer import StreetGaussianRenderer  # noqa: E402
from lib.utils.general_utils import safe_state  # noqa: E402


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def to_numpy(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def load_error_components(root, stem):
    view_dir = os.path.join(root, stem)
    component_path = os.path.join(view_dir, "geometry_error_components.npz")
    error_path = os.path.join(view_dir, "geometry_error_map.npy")
    if not os.path.exists(component_path):
        raise FileNotFoundError(f"Missing geometry error components: {component_path}")
    if not os.path.exists(error_path):
        raise FileNotFoundError(f"Missing geometry error map: {error_path}")
    components = np.load(component_path)
    error_map = np.load(error_path).astype(np.float32)
    valid_mask = components["valid_lidar_mask"].astype(bool)
    boundary_mask = components.get("boundary_mask", np.zeros_like(valid_mask)).astype(bool)
    canny_mask = components.get("canny_mask", np.zeros_like(valid_mask)).astype(bool)
    rendered_depth_edge_mask = components.get("rendered_depth_edge_mask", np.zeros_like(valid_mask)).astype(bool)
    thin_mask = components.get("thin_structure_mask", np.zeros_like(valid_mask)).astype(bool)
    return {
        "view_dir": view_dir,
        "error_map": error_map,
        "valid_mask": valid_mask,
        "boundary_mask": boundary_mask,
        "canny_mask": canny_mask,
        "rendered_depth_edge_mask": rendered_depth_edge_mask,
        "thin_mask": thin_mask,
    }


def find_camera(cameras, stem):
    for camera in cameras:
        if camera.image_name == stem:
            return camera
    for camera in cameras:
        if str(camera.image_name).endswith(stem):
            return camera
    return None


def parse_stem(stem):
    frame, cam = stem.split("_", 1)
    return int(frame), int(cam)


def project_points(points, camera):
    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=1)
    clip = torch.matmul(points_h, camera.full_proj_transform)
    w = clip[:, 3:4]
    ndc = clip[:, :3] / (w + 1e-8)
    x = (ndc[:, 0] + 1.0) * 0.5 * float(camera.image_width)
    y = (1.0 - ndc[:, 1]) * 0.5 * float(camera.image_height)
    return torch.stack([x, y], dim=1)


def resize_map(array, shape, interpolation):
    if array.shape[:2] == shape:
        return array
    return cv2.resize(array, (shape[1], shape[0]), interpolation=interpolation)


def gaussian_window_stats(error_map, valid_mask, masks, center, radius, args, radius_scale=None):
    h, w = error_map.shape
    cx, cy = float(center[0]), float(center[1])
    if not np.isfinite(cx) or not np.isfinite(cy):
        return None
    scale = args.window_radius_scale if radius_scale is None else radius_scale
    r = int(np.ceil(max(float(args.min_window_radius), float(radius) * scale)))
    r = min(r, args.max_window_radius)
    x0, x1 = max(0, int(np.floor(cx - r))), min(w, int(np.ceil(cx + r + 1)))
    y0, y1 = max(0, int(np.floor(cy - r))), min(h, int(np.ceil(cy + r + 1)))
    if x0 >= x1 or y0 >= y1:
        return None

    xs = np.arange(x0, x1, dtype=np.float32) + 0.5
    ys = np.arange(y0, y1, dtype=np.float32) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    sigma = max(float(radius) * 0.5, 1.0)
    dist2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
    support = dist2 <= float(r * r)
    local_valid = valid_mask[y0:y1, x0:x1] & support
    support_count = int(np.count_nonzero(local_valid))
    if support_count < args.min_support_pixels:
        return None

    weights = np.exp(-0.5 * dist2 / (sigma * sigma)).astype(np.float32)
    weights = weights * local_valid.astype(np.float32)
    weight_sum = float(np.sum(weights))
    local_error = error_map[y0:y1, x0:x1]
    if weight_sum <= 1e-8:
        weighted_mean = float(np.mean(local_error[local_valid]))
    else:
        weighted_mean = float(np.sum(weights * local_error) / (weight_sum + 1e-8))
    return {
        "support_pixel_count": support_count,
        "responsibility_score": weighted_mean,
        "mean_geometry_error": float(np.mean(local_error[local_valid])),
        "max_geometry_error": float(np.max(local_error[local_valid])),
        "boundary_overlap": float(np.mean(masks["boundary"][y0:y1, x0:x1][local_valid])),
        "canny_overlap": float(np.mean(masks["canny"][y0:y1, x0:x1][local_valid])),
        "rendered_depth_edge_overlap": float(np.mean(masks["rendered_depth_edge"][y0:y1, x0:x1][local_valid])),
        "thin_structure_overlap": float(np.mean(masks["thin"][y0:y1, x0:x1][local_valid])),
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
    }


STABLE_ID_STRIDE = 10_000_000


def stable_namespace_id(model_name):
    if model_name == "background":
        return 0
    if model_name.startswith("obj_"):
        try:
            return int(model_name.split("_", 1)[1]) + 1
        except ValueError:
            return 9_000_000
    return 9_500_000 + (zlib.crc32(model_name.encode("utf-8")) % 100_000)


def build_model_metadata(pc, count):
    layer_names = np.full(count, "unknown", dtype=object)
    foreground_flags = np.zeros(count, dtype=np.int32)
    object_ids = np.full(count, -1, dtype=np.int32)
    track_ids = np.full(count, -1, dtype=np.int32)
    stable_gaussian_ids = np.full(count, -1, dtype=np.int64)
    model_local_indices = np.full(count, -1, dtype=np.int64)
    model_names = np.full(count, "unknown", dtype=object)
    for model_name, span in getattr(pc, "graph_gaussian_range", {}).items():
        start, end = span
        length = max(0, int(end) - int(start))
        local_indices = np.arange(length, dtype=np.int64)
        namespace_id = stable_namespace_id(model_name)
        layer_names[start:end] = model_name
        model_names[start:end] = model_name
        model_local_indices[start:end] = local_indices
        stable_gaussian_ids[start:end] = namespace_id * STABLE_ID_STRIDE + local_indices
        if model_name.startswith("obj_"):
            foreground_flags[start:end] = 1
            try:
                object_ids[start:end] = int(model_name.split("_", 1)[1])
                track_ids[start:end] = int(model_name.split("_", 1)[1])
            except ValueError:
                pass
    fallback = stable_gaussian_ids < 0
    if np.any(fallback):
        local_indices = np.flatnonzero(fallback).astype(np.int64)
        stable_gaussian_ids[fallback] = 9_999_999 * STABLE_ID_STRIDE + local_indices
        model_local_indices[fallback] = local_indices
    return layer_names, foreground_flags, object_ids, track_ids, stable_gaussian_ids, model_local_indices, model_names


def normalize_u8(values, valid=None):
    x = np.asarray(values, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(x)
    valid = valid & np.isfinite(x)
    out = np.zeros(x.shape, dtype=np.uint8)
    if not np.any(valid):
        return out
    lo, hi = np.percentile(x[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    out[valid] = np.clip((x[valid] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


def draw_gaussians(base, centers, radii, row_indices, color, max_items=100):
    image = base.copy()
    for idx in row_indices[:max_items]:
        x, y = centers[idx]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        radius = int(max(2, min(20, radii[idx])))
        cv2.circle(image, (int(round(x)), int(round(y))), radius, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(image, (int(round(x)), int(round(y))), 2, color, -1, lineType=cv2.LINE_AA)
    return image


def save_histogram(path, scores, top_idx, random_idx, large_idx):
    width, height = 900, 500
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    valid_scores = scores[np.isfinite(scores)]
    if valid_scores.size == 0:
        cv2.imwrite(path, canvas)
        return
    hist, edges = np.histogram(valid_scores, bins=40, range=(0.0, max(1e-6, float(np.max(valid_scores)))))
    max_hist = max(int(np.max(hist)), 1)
    left, right, top, bottom = 60, 860, 40, 420
    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 0, 0), 1)
    bin_w = (right - left) / len(hist)
    for i, count in enumerate(hist):
        x0 = int(left + i * bin_w)
        x1 = int(left + (i + 1) * bin_w)
        y = int(bottom - (bottom - top) * count / max_hist)
        cv2.rectangle(canvas, (x0, y), (x1 - 1, bottom), (180, 180, 180), -1)

    def marker(indices, color, label, y_text):
        if len(indices) == 0:
            return
        value = float(np.mean(scores[indices]))
        pos = int(left + (right - left) * value / max(float(edges[-1]), 1e-6))
        cv2.line(canvas, (pos, top), (pos, bottom), color, 2)
        cv2.putText(canvas, f"{label}: {value:.3f}", (left, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    marker(top_idx, (0, 0, 255), "topK", 455)
    marker(random_idx, (0, 140, 0), "random", 480)
    marker(large_idx, (255, 0, 0), "large-radius", 505)
    cv2.imwrite(path, canvas)


def group_stats(name, indices, arrays, high_error_threshold):
    if len(indices) == 0:
        return {
            "name": name,
            "count": 0,
            "mean_geometry_error": None,
            "boundary_overlap": None,
            "canny_overlap": None,
            "rendered_depth_edge_overlap": None,
            "thin_structure_overlap": None,
            "high_error_pixel_coverage": None,
            "average_screen_radius": None,
            "mean_border_distance": None,
        }
    mean_errors = arrays["mean_errors"][indices]
    max_errors = arrays["max_errors"][indices]
    return {
        "name": name,
        "count": int(len(indices)),
        "mean_geometry_error": float(np.mean(mean_errors)),
        "boundary_overlap": float(np.mean(arrays["boundary_overlaps"][indices])),
        "canny_overlap": float(np.mean(arrays["canny_overlaps"][indices])),
        "rendered_depth_edge_overlap": float(np.mean(arrays["rendered_depth_edge_overlaps"][indices])),
        "thin_structure_overlap": float(np.mean(arrays["thin_structure_overlaps"][indices])),
        "high_error_pixel_coverage": float(np.mean(max_errors >= high_error_threshold)),
        "average_screen_radius": float(np.mean(arrays["radii"][indices])),
        "mean_border_distance": float(np.mean(arrays["border_distances"][indices])),
    }


def load_rgb_for_stem(stem, camera, shape):
    image = to_numpy(camera.original_image)
    if image is not None:
        image = np.transpose(image[:3], (1, 2, 0))
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[:2] != shape:
            image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        return image
    return np.zeros((shape[0], shape[1], 3), dtype=np.uint8)


def compute_border_distances(centers, shape):
    h, w = shape
    x = centers[:, 0]
    y = centers[:, 1]
    distances = np.minimum.reduce([x, y, float(w - 1) - x, float(h - 1) - y])
    return np.maximum(distances, 0.0).astype(np.float32)


def border_bias_summary(indices, border_distances, radii, shape):
    if len(indices) == 0:
        return {"mean_border_distance": None, "near_border_ratio": None, "near_border_threshold": None}
    h, w = shape
    threshold = float(max(20.0, min(h, w) * 0.03))
    selected_distances = border_distances[indices]
    selected_radii = radii[indices]
    return {
        "mean_border_distance": float(np.mean(selected_distances)),
        "median_border_distance": float(np.median(selected_distances)),
        "near_border_threshold": threshold,
        "near_border_ratio": float(np.mean(selected_distances <= threshold)),
        "radius_exceeds_border_distance_ratio": float(np.mean(selected_radii >= selected_distances)),
    }


def sensitivity_test(error_map, valid_mask, masks, centers, radii, candidate_indices, args):
    results = []
    for radius_scale in args.sensitivity_radius_scale:
        rows = []
        for row_idx, gaussian_id in enumerate(candidate_indices):
            stats = gaussian_window_stats(
                error_map,
                valid_mask,
                masks,
                centers[gaussian_id],
                radii[gaussian_id],
                args,
                radius_scale=radius_scale,
            )
            if stats is None:
                continue
            stats["row_idx"] = int(row_idx)
            rows.append(stats)
        if not rows:
            for top_k in args.sensitivity_top_k:
                results.append({"radius_scale": radius_scale, "top_k": int(top_k), "count": 0})
            continue
        order = np.argsort([row["responsibility_score"] for row in rows])[::-1]
        for top_k in args.sensitivity_top_k:
            selected = [rows[i] for i in order[: min(top_k, len(order))]]
            results.append(
                {
                    "radius_scale": float(radius_scale),
                    "top_k": int(top_k),
                    "count": int(len(selected)),
                    "topK_mean_error": float(np.mean([item["mean_geometry_error"] for item in selected])),
                    "topK_boundary_overlap": float(np.mean([item["boundary_overlap"] for item in selected])),
                    "topK_canny_overlap": float(np.mean([item["canny_overlap"] for item in selected])),
                    "topK_rendered_depth_edge_overlap": float(
                        np.mean([item["rendered_depth_edge_overlap"] for item in selected])
                    ),
                    "topK_thin_overlap": float(np.mean([item["thin_structure_overlap"] for item in selected])),
                }
            )
    return results


def diagnose_view(stem, camera, pc, renderer, out_root, script_args):
    components = load_error_components(script_args.geometry_error_map_dir, stem)
    error_map = components["error_map"]
    valid_mask = components["valid_mask"]
    boundary_mask = components["boundary_mask"]
    canny_mask = components["canny_mask"]
    rendered_depth_edge_mask = components["rendered_depth_edge_mask"]
    thin_mask = components["thin_mask"]

    result = renderer.render(camera, pc)
    radii = to_numpy(result["radii"]).astype(np.float32)
    visibility = to_numpy(result["visibility_filter"]).astype(bool)
    with torch.no_grad():
        xyz = pc.get_xyz
        centers = to_numpy(project_points(xyz, camera)).astype(np.float32)
        opacity = to_numpy(pc.get_opacity).reshape(-1).astype(np.float32)
        scale = to_numpy(pc.get_scaling).astype(np.float32)

    if error_map.shape != (int(camera.image_height), int(camera.image_width)):
        error_map = resize_map(error_map, (int(camera.image_height), int(camera.image_width)), cv2.INTER_LINEAR)
        valid_mask = resize_map(valid_mask.astype(np.uint8), error_map.shape, cv2.INTER_NEAREST).astype(bool)
        boundary_mask = resize_map(boundary_mask.astype(np.uint8), error_map.shape, cv2.INTER_NEAREST).astype(bool)
        canny_mask = resize_map(canny_mask.astype(np.uint8), error_map.shape, cv2.INTER_NEAREST).astype(bool)
        rendered_depth_edge_mask = resize_map(
            rendered_depth_edge_mask.astype(np.uint8), error_map.shape, cv2.INTER_NEAREST
        ).astype(bool)
        thin_mask = resize_map(thin_mask.astype(np.uint8), error_map.shape, cv2.INTER_NEAREST).astype(bool)

    (
        layer_names,
        foreground_flags,
        object_ids,
        track_ids,
        stable_gaussian_ids_all,
        model_local_indices_all,
        model_names_all,
    ) = build_model_metadata(pc, len(radii))
    masks = {
        "boundary": boundary_mask,
        "canny": canny_mask,
        "rendered_depth_edge": rendered_depth_edge_mask,
        "thin": thin_mask,
    }

    records = []
    for gaussian_id in np.flatnonzero(visibility):
        stats = gaussian_window_stats(
            error_map,
            valid_mask,
            masks,
            centers[gaussian_id],
            radii[gaussian_id],
            script_args,
        )
        if stats is None:
            continue
        frame_id, camera_id = parse_stem(stem)
        records.append(
            {
                "gaussian_id": int(stable_gaussian_ids_all[gaussian_id]),
                "view_local_gaussian_index": int(gaussian_id),
                "model_local_index": int(model_local_indices_all[gaussian_id]),
                "frame_id": frame_id,
                "camera_id": camera_id,
                "screen_center": centers[gaussian_id].tolist(),
                "screen_space_radius": float(radii[gaussian_id]),
                "visibility": True,
                "opacity": float(opacity[gaussian_id]) if gaussian_id < len(opacity) else None,
                "scale": scale[gaussian_id].tolist() if gaussian_id < len(scale) else None,
                "foreground": int(foreground_flags[gaussian_id]),
                "object_id": int(object_ids[gaussian_id]),
                "track_id": int(track_ids[gaussian_id]),
                "layer_name": str(layer_names[gaussian_id]),
                "model_name": str(model_names_all[gaussian_id]),
                **stats,
            }
        )

    out_dir = ensure_dir(os.path.join(out_root, stem))
    if len(records) == 0:
        summary = {"stem": stem, "visible_gaussians": int(np.count_nonzero(visibility)), "scored_gaussians": 0}
        with open(os.path.join(out_dir, "responsibility_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    gaussian_ids = np.asarray([r["gaussian_id"] for r in records], dtype=np.int64)
    view_local_gaussian_indices = np.asarray([r["view_local_gaussian_index"] for r in records], dtype=np.int64)
    model_local_indices = np.asarray([r["model_local_index"] for r in records], dtype=np.int64)
    model_names = np.asarray([r["model_name"] for r in records], dtype="<U64")
    scores = np.asarray([r["responsibility_score"] for r in records], dtype=np.float32)
    support_counts = np.asarray([r["support_pixel_count"] for r in records], dtype=np.int32)
    mean_errors = np.asarray([r["mean_geometry_error"] for r in records], dtype=np.float32)
    max_errors = np.asarray([r["max_geometry_error"] for r in records], dtype=np.float32)
    boundary_overlaps = np.asarray([r["boundary_overlap"] for r in records], dtype=np.float32)
    canny_overlaps = np.asarray([r["canny_overlap"] for r in records], dtype=np.float32)
    rendered_depth_edge_overlaps = np.asarray([r["rendered_depth_edge_overlap"] for r in records], dtype=np.float32)
    thin_overlaps = np.asarray([r["thin_structure_overlap"] for r in records], dtype=np.float32)
    record_radii = np.asarray([r["screen_space_radius"] for r in records], dtype=np.float32)
    record_centers = np.asarray([r["screen_center"] for r in records], dtype=np.float32)
    border_distances = compute_border_distances(record_centers, error_map.shape)
    foreground = np.asarray([r["foreground"] for r in records], dtype=np.int32)
    object_ids = np.asarray([r["object_id"] for r in records], dtype=np.int32)
    track_ids = np.asarray([r["track_id"] for r in records], dtype=np.int32)

    order = np.argsort(scores)[::-1]
    k = min(script_args.top_k, len(order))
    topk_indices = order[:k]
    rng = np.random.default_rng(script_args.random_seed + parse_stem(stem)[0] * 10 + parse_stem(stem)[1])
    random_indices = rng.choice(len(records), size=k, replace=False) if len(records) >= k else np.arange(len(records))
    large_radius_indices = np.argsort(record_radii)[::-1][:k]

    high_error_threshold = float(np.quantile(error_map[valid_mask], script_args.high_error_quantile)) if np.any(valid_mask) else 1.0
    arrays = {
        "mean_errors": mean_errors,
        "max_errors": max_errors,
        "boundary_overlaps": boundary_overlaps,
        "canny_overlaps": canny_overlaps,
        "rendered_depth_edge_overlaps": rendered_depth_edge_overlaps,
        "thin_structure_overlaps": thin_overlaps,
        "radii": record_radii,
        "border_distances": border_distances,
    }
    baseline = {
        "topK_high_responsibility": group_stats("topK_high_responsibility", topk_indices, arrays, high_error_threshold),
        "random_visible": group_stats("random_visible", random_indices, arrays, high_error_threshold),
        "large_radius_visible": group_stats("large_radius_visible", large_radius_indices, arrays, high_error_threshold),
        "high_error_threshold": high_error_threshold,
    }
    with open(os.path.join(out_dir, "baseline_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)

    np.savez_compressed(
        os.path.join(out_dir, "gaussian_responsibility_v0.npz"),
        stable_id_schema_version=np.asarray([1], dtype=np.int32),
        stable_id_stride=np.asarray([STABLE_ID_STRIDE], dtype=np.int64),
        gaussian_ids=gaussian_ids,
        view_local_gaussian_indices=view_local_gaussian_indices,
        model_local_indices=model_local_indices,
        model_names=model_names,
        responsibility_scores=scores,
        support_pixel_counts=support_counts,
        mean_errors=mean_errors,
        max_errors=max_errors,
        boundary_overlaps=boundary_overlaps,
        canny_overlaps=canny_overlaps,
        rendered_depth_edge_overlaps=rendered_depth_edge_overlaps,
        thin_structure_overlaps=thin_overlaps,
        radii=record_radii,
        screen_centers=record_centers,
        border_distances=border_distances,
        visibility_filter=visibility,
        opacity=np.asarray([r["opacity"] if r["opacity"] is not None else np.nan for r in records], dtype=np.float32),
        scale=np.asarray([r["scale"] if r["scale"] is not None else [np.nan, np.nan, np.nan] for r in records], dtype=np.float32),
        foreground=foreground,
        object_ids=object_ids,
        track_ids=track_ids,
        topk_indices=topk_indices,
        random_indices=random_indices,
        large_radius_indices=large_radius_indices,
    )

    save_visuals = script_args.save_visuals or stem in set(script_args.selected_visual_views)
    visual_paths = {
        "topK_overlay_on_rgb": None,
        "topK_overlay_on_error_map": None,
        "topK_overlay_on_boundary_mask": None,
        "histogram": None,
    }
    if save_visuals:
        rgb = load_rgb_for_stem(stem, camera, error_map.shape)
        error_visual = cv2.applyColorMap(normalize_u8(error_map, valid_mask), cv2.COLORMAP_TURBO)
        error_visual[~valid_mask] = 0
        boundary_visual = np.zeros_like(error_visual)
        boundary_visual[..., 1] = boundary_mask.astype(np.uint8) * 255
        boundary_visual[..., 0] = canny_mask.astype(np.uint8) * 255
        boundary_visual[..., 2] = thin_mask.astype(np.uint8) * 255

        visual_paths = {
            "topK_overlay_on_rgb": os.path.join(out_dir, "topK_overlay_on_rgb.png"),
            "topK_overlay_on_error_map": os.path.join(out_dir, "topK_overlay_on_error_map.png"),
            "topK_overlay_on_boundary_mask": os.path.join(out_dir, "topK_overlay_on_boundary_mask.png"),
            "histogram": os.path.join(out_dir, "responsibility_histogram.png"),
        }
        cv2.imwrite(
            visual_paths["topK_overlay_on_rgb"],
            draw_gaussians(rgb, record_centers, record_radii, topk_indices, (0, 0, 255)),
        )
        cv2.imwrite(
            visual_paths["topK_overlay_on_error_map"],
            draw_gaussians(error_visual, record_centers, record_radii, topk_indices, (255, 255, 255)),
        )
        cv2.imwrite(
            visual_paths["topK_overlay_on_boundary_mask"],
            draw_gaussians(boundary_visual, record_centers, record_radii, topk_indices, (255, 255, 255)),
        )
        save_histogram(
            visual_paths["histogram"],
            scores,
            topk_indices,
            random_indices,
            large_radius_indices,
        )
    sensitivity_rows = np.unique(
        np.concatenate(
            [
                order[: min(len(order), script_args.sensitivity_max_gaussians)],
                large_radius_indices,
            ]
        )
    )
    candidate_local_indices = view_local_gaussian_indices[sensitivity_rows]
    sensitivity = sensitivity_test(error_map, valid_mask, masks, centers, radii, candidate_local_indices, script_args)
    with open(os.path.join(out_dir, "sensitivity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(sensitivity, f, indent=2, ensure_ascii=False)

    summary = {
        "stem": stem,
        "visible_gaussians": int(np.count_nonzero(visibility)),
        "scored_gaussians": int(len(records)),
        "top_k": int(k),
        "viewspace_points_available": result.get("viewspace_points") is not None,
        "radii_available": result.get("radii") is not None,
        "visibility_filter_available": result.get("visibility_filter") is not None,
        "responsibility_formula": "2D Gaussian weighted mean geometry_error_map within projected center/radius support over valid LiDAR pixels.",
        "projection_source": "screen centers are explicitly projected from pc.get_xyz with camera.full_proj_transform because viewspace_points is unavailable in evaluate mode.",
        "contribution_limitation": "responsibility is a screen-space support approximation, not true alpha contribution.",
        "gaussian_id_schema": "stable id = model namespace * 10000000 + model-local index; view-local renderer row is saved separately as view_local_gaussian_indices.",
        "support_window": {
            "min_support_pixels": int(script_args.min_support_pixels),
            "min_window_radius": int(script_args.min_window_radius),
            "max_window_radius": int(script_args.max_window_radius),
            "window_radius_scale": float(script_args.window_radius_scale),
        },
        "mean_responsibility": float(np.mean(scores)),
        "max_responsibility": float(np.max(scores)),
        "topK_border_bias": border_bias_summary(topk_indices, border_distances, record_radii, error_map.shape),
        "topK": baseline["topK_high_responsibility"],
        "random_visible": baseline["random_visible"],
        "large_radius_visible": baseline["large_radius_visible"],
        "top_gaussian_ids": gaussian_ids[topk_indices[:20]].tolist(),
        "paths": {
            "npz": os.path.join(out_dir, "gaussian_responsibility_v0.npz"),
            "summary": os.path.join(out_dir, "responsibility_summary.json"),
            "baseline": os.path.join(out_dir, "baseline_comparison.json"),
            "topK_overlay_on_rgb": visual_paths["topK_overlay_on_rgb"],
            "topK_overlay_on_error_map": visual_paths["topK_overlay_on_error_map"],
            "topK_overlay_on_boundary_mask": visual_paths["topK_overlay_on_boundary_mask"],
            "histogram": visual_paths["histogram"],
            "sensitivity": os.path.join(out_dir, "sensitivity_summary.json"),
        },
    }
    with open(os.path.join(out_dir, "responsibility_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    random.seed(SCRIPT_ARGS.random_seed)
    np.random.seed(SCRIPT_ARGS.random_seed)
    cfg.render.save_image = False
    cfg.render.save_video = False
    safe_state(cfg.eval.quiet)

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        cameras = scene.getTrainCameras()
        camera_by_stem = {camera.image_name: camera for camera in cameras}
        output_root = ensure_dir(SCRIPT_ARGS.output_dir)

        summaries = []
        for stem in SCRIPT_ARGS.views:
            summary_path = os.path.join(output_root, stem, "responsibility_summary.json")
            npz_path = os.path.join(output_root, stem, "gaussian_responsibility_v0.npz")
            if SCRIPT_ARGS.skip_existing and os.path.exists(summary_path) and os.path.exists(npz_path):
                print(f"Skipping existing {stem}")
                with open(summary_path, "r", encoding="utf-8") as f:
                    summaries.append(json.load(f))
                continue
            camera = camera_by_stem.get(stem) or find_camera(cameras, stem)
            if camera is None:
                print(f"Skipping {stem}: camera not found")
                continue
            print(f"Diagnosing {stem}")
            summaries.append(diagnose_view(stem, camera, gaussians, renderer, output_root, SCRIPT_ARGS))

    aggregate = {
        "views_requested": SCRIPT_ARGS.views,
        "views_analyzed": [item["stem"] for item in summaries],
        "view_count": len(summaries),
        "mean_topK_error": float(np.mean([s["topK"]["mean_geometry_error"] for s in summaries if s.get("scored_gaussians", 0) > 0]))
        if summaries
        else None,
        "mean_random_error": float(np.mean([s["random_visible"]["mean_geometry_error"] for s in summaries if s.get("scored_gaussians", 0) > 0]))
        if summaries
        else None,
        "mean_large_radius_error": float(np.mean([s["large_radius_visible"]["mean_geometry_error"] for s in summaries if s.get("scored_gaussians", 0) > 0]))
        if summaries
        else None,
    }
    with open(os.path.join(SCRIPT_ARGS.output_dir, "responsibility_v0_all_views_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"aggregate": aggregate, "frames": summaries}, f, indent=2, ensure_ascii=False)
    print(f"Analyzed {len(summaries)} views.")
    print(f"Saved outputs under: {SCRIPT_ARGS.output_dir}")


if __name__ == "__main__":
    main()
