import argparse
import json
import os

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build pixel-level geometry error maps from existing A/B/D geometry evaluations."
    )
    parser.add_argument("--group-a", required=True, help="DA3-only geometry_credibility_eval directory.")
    parser.add_argument("--group-b", required=True, help="Boundary SILog geometry_credibility_eval directory.")
    parser.add_argument("--group-d", required=True, help="Boundary + edge geometry_credibility_eval directory.")
    parser.add_argument("--comparison-dir", required=True, help="Existing geometry_group_comparison directory.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-depth", type=float, default=None)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--a-error-weight", type=float, default=1.0)
    parser.add_argument("--d-degrade-weight", type=float, default=1.0)
    parser.add_argument("--b-degrade-weight", type=float, default=0.35)
    parser.add_argument("--boundary-boost", type=float, default=0.75)
    parser.add_argument("--thin-boost", type=float, default=1.0)
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path, base_dir):
    if not path:
        return None
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidates = [
        os.path.abspath(path),
        os.path.abspath(os.path.join(os.getcwd(), path)),
        os.path.abspath(os.path.join(base_dir, path)),
        os.path.abspath(os.path.join(base_dir, "..", path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def load_eval(eval_dir):
    metrics_path = os.path.join(eval_dir, "geometry_credibility_metrics.json")
    data = load_json(metrics_path)
    frames = {}
    for frame in data.get("frames", []):
        stem = frame.get("stem")
        if stem:
            frames[stem] = frame
    return {"dir": eval_dir, "metrics_path": metrics_path, "json": data, "frames": frames}


def frame_paths(group, stem):
    frame = group["frames"][stem]
    paths = frame.get("paths", {})
    base_dir = os.path.abspath(os.path.join(group["dir"], "..", "..", ".."))
    return {name: resolve_path(path, base_dir) for name, path in paths.items()}


def load_depth(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == () and isinstance(data.item(), dict):
        data = data.item()
    if isinstance(data, dict) and "mask" in data and "value" in data:
        mask = np.asarray(data["mask"]).astype(bool)
        depth = np.zeros(mask.shape, dtype=np.float32)
        depth[mask] = np.asarray(data["value"], dtype=np.float32).reshape(-1)
        return depth
    return np.asarray(data, dtype=np.float32).squeeze()


def load_lidar(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == () and isinstance(data.item(), dict):
        data = data.item()
    if isinstance(data, dict) and "mask" in data and "value" in data:
        mask = np.asarray(data["mask"]).astype(bool)
        depth = np.zeros(mask.shape, dtype=np.float32)
        depth[mask] = np.asarray(data["value"], dtype=np.float32).reshape(-1)
        return depth, mask
    depth = np.asarray(data, dtype=np.float32).squeeze()
    return depth, np.isfinite(depth) & (depth > 0)


def resize_float(src, shape, interpolation=cv2.INTER_LINEAR):
    if src.shape[:2] == shape:
        return src
    return cv2.resize(src, (shape[1], shape[0]), interpolation=interpolation)


def resize_mask(src, shape):
    if src.shape[:2] == shape:
        return src.astype(bool)
    return cv2.resize(src.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def load_mask(path, shape):
    if not os.path.exists(path):
        return np.zeros(shape, dtype=bool)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(shape, dtype=bool)
    return resize_mask(mask > 0, shape)


def load_mask_with_source(path, shape):
    if not os.path.exists(path):
        return np.zeros(shape, dtype=bool), False
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(shape, dtype=bool), False
    return resize_mask(mask > 0, shape), True


def dilate_mask(mask, radius):
    if radius <= 0:
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def derive_canny_band(rgb_path, shape, radius=3):
    if not rgb_path or not os.path.exists(rgb_path):
        return np.zeros(shape, dtype=bool), False
    image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if image is None:
        return np.zeros(shape, dtype=bool), False
    if image.shape[:2] != shape:
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160) > 0
    return dilate_mask(edges, radius), True


def derive_depth_edge_band(depth, valid, radius=3):
    depth = np.asarray(depth, dtype=np.float32)
    valid = valid & np.isfinite(depth)
    if not np.any(valid):
        return np.zeros(depth.shape, dtype=bool), False
    safe_depth = depth.copy()
    fill = float(np.median(safe_depth[valid]))
    safe_depth[~valid] = fill
    gx = cv2.Sobel(safe_depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(safe_depth, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad[~valid] = 0.0
    positive = grad[valid & (grad > 0)]
    if positive.size == 0:
        return np.zeros(depth.shape, dtype=bool), False
    threshold = float(np.percentile(positive, 90))
    edges = valid & (grad >= max(threshold, 1e-6))
    return dilate_mask(edges, radius), True


def derive_thin_structure_band(canny_mask, rendered_edge_mask):
    if not np.any(canny_mask) and not np.any(rendered_edge_mask):
        return np.zeros(canny_mask.shape, dtype=bool), False
    combined = canny_mask & dilate_mask(rendered_edge_mask, 1)
    if not np.any(combined):
        combined = canny_mask | rendered_edge_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(combined.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    thin = combined & (~opened.astype(bool))
    if not np.any(thin):
        thin = combined
    return thin.astype(bool), True


def normalize_u8(values, valid):
    x = np.asarray(values, dtype=np.float32)
    valid = valid & np.isfinite(x)
    out = np.zeros(x.shape, dtype=np.uint8)
    if not np.any(valid):
        return out
    lo, hi = np.percentile(x[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    out[valid] = np.clip((x[valid] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


def save_error_png(path, values, valid):
    visual = cv2.applyColorMap(normalize_u8(values, valid), cv2.COLORMAP_TURBO)
    visual[~valid] = 0
    cv2.imwrite(path, visual)


def positive_part(values):
    return np.maximum(values, 0.0)


def normalized_component(values, valid):
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(values, dtype=np.float32)
    valid = valid & np.isfinite(values)
    if not np.any(valid):
        return out
    scale = float(np.percentile(values[valid], 95))
    if scale <= 1e-6:
        scale = float(np.max(values[valid]))
    if scale <= 1e-6:
        return out
    out[valid] = np.clip(values[valid] / scale, 0.0, 1.0)
    return out


def build_frame(args, groups, stem, output_root):
    paths_a = frame_paths(groups["A"], stem)
    paths_b = frame_paths(groups["B"], stem)
    paths_d = frame_paths(groups["D"], stem)

    depth_a = load_depth(paths_a["rendered_depth"])
    depth_b = resize_float(load_depth(paths_b["rendered_depth"]), depth_a.shape)
    depth_d = resize_float(load_depth(paths_d["rendered_depth"]), depth_a.shape)
    lidar, lidar_mask = load_lidar(paths_a["lidar_depth"])
    lidar = resize_float(lidar, depth_a.shape, cv2.INTER_NEAREST)
    lidar_mask = resize_mask(lidar_mask, depth_a.shape)

    min_depth = args.min_depth
    max_depth = args.max_depth
    if min_depth is None:
        min_depth = float(groups["A"]["json"].get("config", {}).get("min_depth", 1.0))
    if max_depth is None:
        max_depth = float(groups["A"]["json"].get("config", {}).get("max_depth", 80.0))

    valid_lidar_mask = (
        lidar_mask
        & np.isfinite(lidar)
        & np.isfinite(depth_a)
        & np.isfinite(depth_b)
        & np.isfinite(depth_d)
        & (lidar > min_depth)
        & (lidar < max_depth)
        & (depth_a > min_depth)
        & (depth_a < max_depth)
        & (depth_b > min_depth)
        & (depth_b < max_depth)
        & (depth_d > min_depth)
        & (depth_d < max_depth)
    )

    a_abs_error = np.abs(depth_a - lidar).astype(np.float32)
    b_abs_error = np.abs(depth_b - lidar).astype(np.float32)
    d_abs_error = np.abs(depth_d - lidar).astype(np.float32)
    b_minus_a = (b_abs_error - a_abs_error).astype(np.float32)
    d_minus_a = (d_abs_error - a_abs_error).astype(np.float32)
    d_minus_b = (d_abs_error - b_abs_error).astype(np.float32)

    frame_eval_dir = os.path.join(groups["A"]["dir"], stem)
    boundary_mask, boundary_from_file = load_mask_with_source(os.path.join(frame_eval_dir, "boundary_band_mask.png"), depth_a.shape)
    canny_mask, canny_from_file = load_mask_with_source(os.path.join(frame_eval_dir, "canny_band_mask.png"), depth_a.shape)
    rendered_edge_mask, rendered_edge_from_file = load_mask_with_source(
        os.path.join(frame_eval_dir, "rendered_depth_edge_band_mask.png"), depth_a.shape
    )
    thin_structure_mask, thin_from_file = load_mask_with_source(os.path.join(frame_eval_dir, "thin_structure_mask.png"), depth_a.shape)

    canny_derived = False
    rendered_edge_derived = False
    thin_derived = False
    if not np.any(canny_mask):
        canny_mask, canny_derived = derive_canny_band(paths_a.get("rgb"), depth_a.shape, radius=3)
    if not np.any(rendered_edge_mask):
        rendered_edge_mask, rendered_edge_derived = derive_depth_edge_band(depth_a, valid_lidar_mask, radius=3)
    if not np.any(boundary_mask):
        boundary_mask = canny_mask | rendered_edge_mask
    if not np.any(thin_structure_mask):
        thin_structure_mask, thin_derived = derive_thin_structure_band(canny_mask, rendered_edge_mask)
    mask_sources = {
        "boundary": "file" if boundary_from_file else "derived_from_canny_or_rendered_depth_edge",
        "canny": "file" if canny_from_file else ("derived_from_rgb_canny" if canny_derived else "missing"),
        "rendered_depth_edge": "file"
        if rendered_edge_from_file
        else ("derived_from_rendered_depth_gradient" if rendered_edge_derived else "missing"),
        "thin_structure": "file" if thin_from_file else ("derived_from_canny_and_depth_edge" if thin_derived else "missing"),
    }

    boundary_region = boundary_mask | canny_mask | rendered_edge_mask
    boundary_weight = np.ones(depth_a.shape, dtype=np.float32)
    boundary_weight[boundary_region] += args.boundary_boost
    thin_structure_weight = np.ones(depth_a.shape, dtype=np.float32)
    thin_structure_weight[thin_structure_mask] += args.thin_boost

    a_component = normalized_component(a_abs_error, valid_lidar_mask)
    d_degrade_component = normalized_component(positive_part(d_minus_a), valid_lidar_mask)
    b_degrade_component = normalized_component(positive_part(b_minus_a), valid_lidar_mask)

    fused = (
        args.a_error_weight * a_component
        + args.d_degrade_weight * d_degrade_component
        + args.b_degrade_weight * b_degrade_component
    )
    fused = fused * boundary_weight * thin_structure_weight
    final_geometry_error_map = np.zeros_like(fused, dtype=np.float32)
    if np.any(valid_lidar_mask):
        scale = float(np.percentile(fused[valid_lidar_mask], 98))
        if scale <= 1e-6:
            scale = float(np.max(fused[valid_lidar_mask]))
        scale = max(scale, 1e-6)
        final_geometry_error_map[valid_lidar_mask] = np.clip(fused[valid_lidar_mask] / scale, 0.0, 1.0)

    out_dir = ensure_dir(os.path.join(output_root, stem))
    np.save(os.path.join(out_dir, "geometry_error_map.npy"), final_geometry_error_map)
    save_error_png(os.path.join(out_dir, "geometry_error_map.png"), final_geometry_error_map, valid_lidar_mask)
    np.savez_compressed(
        os.path.join(out_dir, "geometry_error_components.npz"),
        valid_lidar_mask=valid_lidar_mask,
        A_abs_error=a_abs_error,
        B_abs_error=b_abs_error,
        D_abs_error=d_abs_error,
        B_minus_A=b_minus_a,
        D_minus_A=d_minus_a,
        D_minus_B=d_minus_b,
        boundary_weight=boundary_weight,
        thin_structure_weight=thin_structure_weight,
        boundary_mask=boundary_mask,
        canny_mask=canny_mask,
        rendered_depth_edge_mask=rendered_edge_mask,
        thin_structure_mask=thin_structure_mask,
        final_geometry_error_map=final_geometry_error_map,
        mask_source_keys=np.asarray(list(mask_sources.keys()), dtype="<U32"),
        mask_source_values=np.asarray(list(mask_sources.values()), dtype="<U64"),
    )

    valid_count = int(np.count_nonzero(valid_lidar_mask))
    if valid_count:
        high_threshold = float(np.percentile(final_geometry_error_map[valid_lidar_mask], 90))
        high_mask = valid_lidar_mask & (final_geometry_error_map >= high_threshold)
        boundary_high_ratio = float(np.mean(boundary_region[high_mask])) if np.any(high_mask) else 0.0
        thin_high_ratio = float(np.mean(thin_structure_mask[high_mask])) if np.any(high_mask) else 0.0
        summary = {
            "stem": stem,
            "valid_lidar_count": valid_count,
            "mean_geometry_error": float(np.mean(final_geometry_error_map[valid_lidar_mask])),
            "p90_geometry_error": high_threshold,
            "p95_geometry_error": float(np.percentile(final_geometry_error_map[valid_lidar_mask], 95)),
            "max_geometry_error": float(np.max(final_geometry_error_map[valid_lidar_mask])),
            "mean_A_abs_error": float(np.mean(a_abs_error[valid_lidar_mask])),
            "mean_D_minus_A": float(np.mean(d_minus_a[valid_lidar_mask])),
            "degraded_ratio_D_vs_A": float(np.mean(d_minus_a[valid_lidar_mask] > 0)),
            "improved_ratio_D_vs_A": float(np.mean(d_minus_a[valid_lidar_mask] < 0)),
            "high_error_boundary_ratio": boundary_high_ratio,
            "high_error_thin_structure_ratio": thin_high_ratio,
            "mask_sources": mask_sources,
            "paths": {
                "geometry_error_map_npy": os.path.join(out_dir, "geometry_error_map.npy"),
                "geometry_error_map_png": os.path.join(out_dir, "geometry_error_map.png"),
                "geometry_error_components": os.path.join(out_dir, "geometry_error_components.npz"),
            },
        }
    else:
        summary = {
            "stem": stem,
            "valid_lidar_count": 0,
            "mean_geometry_error": None,
            "p90_geometry_error": None,
            "p95_geometry_error": None,
            "max_geometry_error": None,
            "mean_A_abs_error": None,
            "mean_D_minus_A": None,
            "degraded_ratio_D_vs_A": None,
            "improved_ratio_D_vs_A": None,
            "high_error_boundary_ratio": None,
            "high_error_thin_structure_ratio": None,
            "paths": {
                "geometry_error_map_npy": os.path.join(out_dir, "geometry_error_map.npy"),
                "geometry_error_map_png": os.path.join(out_dir, "geometry_error_map.png"),
                "geometry_error_components": os.path.join(out_dir, "geometry_error_components.npz"),
            },
        }

    with open(os.path.join(out_dir, "error_map_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def aggregate_frame_summaries(frames):
    valid_frames = [f for f in frames if f["valid_lidar_count"] > 0]
    if not valid_frames:
        return {}
    total_count = sum(f["valid_lidar_count"] for f in valid_frames)
    weighted_keys = [
        "mean_geometry_error",
        "mean_A_abs_error",
        "mean_D_minus_A",
        "degraded_ratio_D_vs_A",
        "improved_ratio_D_vs_A",
        "high_error_boundary_ratio",
        "high_error_thin_structure_ratio",
    ]
    aggregate = {"valid_lidar_count": int(total_count), "frame_count": len(valid_frames)}
    for key in weighted_keys:
        aggregate[key] = float(
            sum(f[key] * f["valid_lidar_count"] for f in valid_frames) / max(total_count, 1)
        )
    aggregate["top_views_by_mean_geometry_error"] = sorted(
        valid_frames, key=lambda item: item["mean_geometry_error"], reverse=True
    )[:5]
    aggregate["top_views_by_boundary_concentration"] = sorted(
        valid_frames, key=lambda item: item["high_error_boundary_ratio"], reverse=True
    )[:5]
    aggregate["top_views_by_thin_concentration"] = sorted(
        valid_frames, key=lambda item: item["high_error_thin_structure_ratio"], reverse=True
    )[:5]
    return aggregate


def main():
    args = parse_args()
    groups = {
        "A": load_eval(args.group_a),
        "B": load_eval(args.group_b),
        "D": load_eval(args.group_d),
    }
    comparison_summary = os.path.join(args.comparison_dir, "geometry_group_comparison_summary.json")
    if not os.path.exists(comparison_summary):
        raise FileNotFoundError(f"Missing comparison summary: {comparison_summary}")
    comparison = load_json(comparison_summary)
    comparison_stems = {frame.get("stem") for frame in comparison.get("frames", []) if frame.get("stem")}
    common_stems = sorted(
        set(groups["A"]["frames"]) & set(groups["B"]["frames"]) & set(groups["D"]["frames"]) & comparison_stems
    )
    if not common_stems:
        raise RuntimeError("No matched frame/cam stems found across A/B/D eval directories and comparison summary.")

    output_root = args.output_dir or os.path.join(args.comparison_dir, "geometry_error_map")
    ensure_dir(output_root)

    frames = [build_frame(args, groups, stem, output_root) for stem in common_stems]
    summary = {
        "config": {
            "group_a": os.path.abspath(args.group_a),
            "group_b": os.path.abspath(args.group_b),
            "group_d": os.path.abspath(args.group_d),
            "comparison_dir": os.path.abspath(args.comparison_dir),
            "output_dir": os.path.abspath(output_root),
            "frame_count": len(common_stems),
            "a_error_weight": args.a_error_weight,
            "d_degrade_weight": args.d_degrade_weight,
            "b_degrade_weight": args.b_degrade_weight,
            "boundary_boost": args.boundary_boost,
            "thin_boost": args.thin_boost,
            "lidar_handling": "Waymo LiDAR dict(mask,value) is loaded with an explicit valid mask; invalid pixels are excluded from all error components.",
        },
        "aggregate": aggregate_frame_summaries(frames),
        "frames": frames,
    }
    summary_path = os.path.join(output_root, "geometry_error_map_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Built geometry error maps for {len(frames)} matched views.")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
