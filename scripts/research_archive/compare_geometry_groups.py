import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np


REGIONS = [
    "all_valid",
    "boundary_band",
    "canny_band",
    "rendered_depth_edge_band",
    "thin_structure_band",
    "stable_non_boundary",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare A/B/D LiDAR depth errors from existing geometry credibility evaluations."
    )
    parser.add_argument("--group-a", required=True, help="DA3-only geometry_credibility_eval directory.")
    parser.add_argument("--group-b", required=True, help="Boundary SILog geometry_credibility_eval directory.")
    parser.add_argument("--group-d", required=True, help="Boundary + edge geometry_credibility_eval directory.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-depth", type=float, default=None)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--diff-threshold", type=float, default=0.0)
    parser.add_argument("--top-k-crops", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=160)
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


def load_mask(path, shape):
    if not path or not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def load_rgb(path, shape):
    if not path or not os.path.exists(path):
        return None
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    if image.shape[:2] != shape:
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return image


def normalize_u8(values, valid=None, symmetric=False, vmax=None):
    x = np.asarray(values, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(x)
    valid = valid & np.isfinite(x)
    out = np.zeros(x.shape, dtype=np.uint8)
    if not np.any(valid):
        return out
    if symmetric:
        limit = vmax if vmax is not None else float(np.percentile(np.abs(x[valid]), 98))
        limit = max(limit, 1e-6)
        out[valid] = np.clip((x[valid] + limit) / (2.0 * limit) * 255.0, 0, 255).astype(np.uint8)
        return out
    lo, hi = np.percentile(x[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    out[valid] = np.clip((x[valid] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


def colorize_error(values, valid):
    return cv2.applyColorMap(normalize_u8(values, valid), cv2.COLORMAP_TURBO)


def colorize_diff(values, valid):
    gray = normalize_u8(values, valid, symmetric=True)
    x = gray.astype(np.float32) / 255.0
    blue = np.clip(2.0 * (0.5 - x), 0.0, 1.0)
    red = np.clip(2.0 * (x - 0.5), 0.0, 1.0)
    white = 1.0 - np.abs(x - 0.5) * 2.0
    image = np.stack(
        [
            (blue + white) * 255.0,
            white * 255.0,
            (red + white) * 255.0,
        ],
        axis=-1,
    )
    image[~valid] = 0
    return np.clip(image, 0, 255).astype(np.uint8)


def save_mask(path, mask):
    cv2.imwrite(path, np.asarray(mask, dtype=np.uint8) * 255)


def frame_paths(group, stem):
    frame = group["frames"][stem]
    paths = frame.get("paths", {})
    base_dir = os.path.abspath(os.path.join(group["dir"], "..", "..", ".."))
    return {name: resolve_path(path, base_dir) for name, path in paths.items()}


def region_masks(eval_dir, stem, base_valid):
    view_dir = os.path.join(eval_dir, stem)
    shape = base_valid.shape
    boundary = load_mask(os.path.join(view_dir, "boundary_band_mask.png"), shape)
    canny = load_mask(os.path.join(view_dir, "canny_band_mask.png"), shape)
    rendered_edge = load_mask(os.path.join(view_dir, "rendered_depth_edge_band_mask.png"), shape)
    thin = load_mask(os.path.join(view_dir, "thin_structure_mask.png"), shape)

    masks = {"all_valid": base_valid}
    masks["boundary_band"] = boundary if boundary is not None else np.zeros(shape, dtype=bool)
    masks["canny_band"] = canny if canny is not None else np.zeros(shape, dtype=bool)
    masks["rendered_depth_edge_band"] = rendered_edge if rendered_edge is not None else np.zeros(shape, dtype=bool)
    masks["thin_structure_band"] = thin if thin is not None else np.zeros(shape, dtype=bool)
    masks["stable_non_boundary"] = base_valid & (~masks["boundary_band"])
    return masks


def compute_region_stats(mask, valid, err_a, err_b, err_d, threshold):
    region_valid = valid & mask
    count = int(np.count_nonzero(region_valid))
    if count == 0:
        return {
            "valid_lidar_count": 0,
            "mean_error_A": None,
            "mean_error_B": None,
            "mean_error_D": None,
            "mean_B_minus_A": None,
            "mean_D_minus_A": None,
            "mean_D_minus_B": None,
            "degraded_ratio_B_vs_A": None,
            "degraded_ratio_D_vs_A": None,
            "improved_ratio_D_vs_A": None,
        }
    b_minus_a = err_b - err_a
    d_minus_a = err_d - err_a
    d_minus_b = err_d - err_b
    return {
        "valid_lidar_count": count,
        "mean_error_A": float(np.mean(err_a[region_valid])),
        "mean_error_B": float(np.mean(err_b[region_valid])),
        "mean_error_D": float(np.mean(err_d[region_valid])),
        "mean_B_minus_A": float(np.mean(b_minus_a[region_valid])),
        "mean_D_minus_A": float(np.mean(d_minus_a[region_valid])),
        "mean_D_minus_B": float(np.mean(d_minus_b[region_valid])),
        "degraded_ratio_B_vs_A": float(np.mean(b_minus_a[region_valid] > threshold)),
        "degraded_ratio_D_vs_A": float(np.mean(d_minus_a[region_valid] > threshold)),
        "improved_ratio_D_vs_A": float(np.mean(d_minus_a[region_valid] < -threshold)),
    }


def accumulate(accumulator, region, stats):
    count = stats["valid_lidar_count"]
    if count <= 0:
        return
    bucket = accumulator[region]
    bucket["valid_lidar_count"] += count
    for key in [
        "mean_error_A",
        "mean_error_B",
        "mean_error_D",
        "mean_B_minus_A",
        "mean_D_minus_A",
        "mean_D_minus_B",
        "degraded_ratio_B_vs_A",
        "degraded_ratio_D_vs_A",
        "improved_ratio_D_vs_A",
    ]:
        bucket[key] += stats[key] * count


def finalize(accumulator):
    result = {}
    for region in REGIONS:
        bucket = accumulator.get(region, {})
        count = bucket.get("valid_lidar_count", 0)
        if count <= 0:
            result[region] = compute_region_stats(
                np.zeros((1, 1), dtype=bool),
                np.zeros((1, 1), dtype=bool),
                np.zeros((1, 1), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
                0.0,
            )
            continue
        out = {"valid_lidar_count": int(count)}
        for key, value in bucket.items():
            if key != "valid_lidar_count":
                out[key] = float(value / count)
        result[region] = out
    return result


def make_crop_sheet(rgb, err_a, err_b, err_d, b_minus_a, d_minus_a, valid, center, crop_size):
    h, w = err_a.shape
    half = crop_size // 2
    y, x = center
    y0 = max(0, min(h - crop_size, y - half))
    x0 = max(0, min(w - crop_size, x - half))
    y1 = min(h, y0 + crop_size)
    x1 = min(w, x0 + crop_size)
    y0 = max(0, y1 - crop_size)
    x0 = max(0, x1 - crop_size)
    sl = np.s_[y0:y1, x0:x1]
    panels = []
    if rgb is not None:
        panels.append(rgb[sl])
    panels.extend(
        [
            colorize_error(err_a[sl], valid[sl]),
            colorize_error(err_b[sl], valid[sl]),
            colorize_error(err_d[sl], valid[sl]),
            colorize_diff(b_minus_a[sl], valid[sl]),
            colorize_diff(d_minus_a[sl], valid[sl]),
        ]
    )
    height = max(panel.shape[0] for panel in panels)
    resized = []
    for panel in panels:
        if panel.shape[0] != height:
            panel = cv2.resize(panel, (panel.shape[1], height), interpolation=cv2.INTER_LINEAR)
        resized.append(panel)
    return np.concatenate(resized, axis=1), [int(y0), int(x0), int(y1), int(x1)]


def save_top_crops(out_dir, stem, rgb, err_a, err_b, err_d, b_minus_a, d_minus_a, valid, top_k, crop_size):
    crop_dir = ensure_dir(os.path.join(out_dir, "crops"))
    score = np.maximum(b_minus_a, d_minus_a)
    score = np.where(valid & np.isfinite(score), score, -np.inf)
    if not np.any(np.isfinite(score)):
        return []
    flat_order = np.argsort(score.reshape(-1))[::-1]
    selected = []
    occupied = np.zeros(score.shape, dtype=bool)
    radius = max(8, crop_size // 3)
    h, w = score.shape
    for flat_idx in flat_order:
        if len(selected) >= top_k:
            break
        y, x = np.unravel_index(int(flat_idx), score.shape)
        if not np.isfinite(score[y, x]) or occupied[y, x]:
            continue
        sheet, bbox = make_crop_sheet(rgb, err_a, err_b, err_d, b_minus_a, d_minus_a, valid, (y, x), crop_size)
        path = os.path.join(crop_dir, f"{stem}_diff_crop_{len(selected):02d}.png")
        cv2.imwrite(path, sheet)
        selected.append({"center": [int(y), int(x)], "bbox": bbox, "score": float(score[y, x]), "path": path})
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        occupied[y0:y1, x0:x1] = True
    return selected


def compare_frame(args, groups, stem, out_root, accumulator):
    paths_a = frame_paths(groups["A"], stem)
    paths_b = frame_paths(groups["B"], stem)
    paths_d = frame_paths(groups["D"], stem)

    depth_a = load_depth(paths_a["rendered_depth"])
    depth_b = load_depth(paths_b["rendered_depth"])
    depth_d = load_depth(paths_d["rendered_depth"])
    lidar, lidar_mask = load_lidar(paths_a["lidar_depth"])
    if depth_b.shape != depth_a.shape:
        depth_b = cv2.resize(depth_b, (depth_a.shape[1], depth_a.shape[0]), interpolation=cv2.INTER_LINEAR)
    if depth_d.shape != depth_a.shape:
        depth_d = cv2.resize(depth_d, (depth_a.shape[1], depth_a.shape[0]), interpolation=cv2.INTER_LINEAR)
    if lidar.shape != depth_a.shape:
        lidar = cv2.resize(lidar, (depth_a.shape[1], depth_a.shape[0]), interpolation=cv2.INTER_NEAREST)
        lidar_mask = cv2.resize(
            lidar_mask.astype(np.uint8),
            (depth_a.shape[1], depth_a.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    min_depth = args.min_depth
    max_depth = args.max_depth
    if min_depth is None:
        min_depth = float(groups["A"]["json"].get("config", {}).get("min_depth", 1.0))
    if max_depth is None:
        max_depth = float(groups["A"]["json"].get("config", {}).get("max_depth", 80.0))

    valid = (
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
    err_a = np.abs(depth_a - lidar)
    err_b = np.abs(depth_b - lidar)
    err_d = np.abs(depth_d - lidar)
    b_minus_a = err_b - err_a
    d_minus_a = err_d - err_a
    d_minus_b = err_d - err_b

    masks = region_masks(groups["A"]["dir"], stem, valid)
    frame_stats = {}
    for region in REGIONS:
        stats = compute_region_stats(masks[region], valid, err_a, err_b, err_d, args.diff_threshold)
        frame_stats[region] = stats
        accumulate(accumulator, region, stats)

    view_dir = ensure_dir(os.path.join(out_root, stem))
    cv2.imwrite(os.path.join(view_dir, "A_abs_error.png"), colorize_error(err_a, valid))
    cv2.imwrite(os.path.join(view_dir, "B_abs_error.png"), colorize_error(err_b, valid))
    cv2.imwrite(os.path.join(view_dir, "D_abs_error.png"), colorize_error(err_d, valid))
    cv2.imwrite(os.path.join(view_dir, "B_minus_A.png"), colorize_diff(b_minus_a, valid))
    cv2.imwrite(os.path.join(view_dir, "D_minus_A.png"), colorize_diff(d_minus_a, valid))
    cv2.imwrite(os.path.join(view_dir, "D_minus_B.png"), colorize_diff(d_minus_b, valid))
    save_mask(os.path.join(view_dir, "B_degraded_vs_A_mask.png"), valid & (b_minus_a > args.diff_threshold))
    save_mask(os.path.join(view_dir, "D_degraded_vs_A_mask.png"), valid & (d_minus_a > args.diff_threshold))
    save_mask(os.path.join(view_dir, "D_improved_vs_A_mask.png"), valid & (d_minus_a < -args.diff_threshold))

    rgb = load_rgb(paths_a.get("rgb"), depth_a.shape)
    crops = save_top_crops(
        view_dir,
        stem,
        rgb,
        err_a,
        err_b,
        err_d,
        b_minus_a,
        d_minus_a,
        valid,
        args.top_k_crops,
        args.crop_size,
    )

    return {
        "stem": stem,
        "valid_lidar_count": int(np.count_nonzero(valid)),
        "regions": frame_stats,
        "paths": {
            "A_abs_error": os.path.join(view_dir, "A_abs_error.png"),
            "B_abs_error": os.path.join(view_dir, "B_abs_error.png"),
            "D_abs_error": os.path.join(view_dir, "D_abs_error.png"),
            "B_minus_A": os.path.join(view_dir, "B_minus_A.png"),
            "D_minus_A": os.path.join(view_dir, "D_minus_A.png"),
            "D_minus_B": os.path.join(view_dir, "D_minus_B.png"),
        },
        "crops": crops,
    }


def main():
    args = parse_args()
    groups = {
        "A": load_eval(args.group_a),
        "B": load_eval(args.group_b),
        "D": load_eval(args.group_d),
    }
    common_stems = sorted(set(groups["A"]["frames"]) & set(groups["B"]["frames"]) & set(groups["D"]["frames"]))
    if not common_stems:
        raise RuntimeError("No matching frame/cam stems found across A/B/D eval directories.")

    out_root = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.group_a)), "geometry_group_comparison")
    ensure_dir(out_root)

    accumulator = defaultdict(lambda: defaultdict(float))
    frame_results = []
    for stem in common_stems:
        frame_results.append(compare_frame(args, groups, stem, out_root, accumulator))

    summary = {
        "config": {
            "group_a": os.path.abspath(args.group_a),
            "group_b": os.path.abspath(args.group_b),
            "group_d": os.path.abspath(args.group_d),
            "output_dir": os.path.abspath(out_root),
            "regions": REGIONS,
            "common_frame_count": len(common_stems),
            "diff_threshold": args.diff_threshold,
            "min_depth": args.min_depth
            if args.min_depth is not None
            else groups["A"]["json"].get("config", {}).get("min_depth", 1.0),
            "max_depth": args.max_depth
            if args.max_depth is not None
            else groups["A"]["json"].get("config", {}).get("max_depth", 80.0),
            "region_masks_source": "A group geometry_credibility_eval masks, reused for same-region A/B/D comparison.",
        },
        "aggregate": finalize(accumulator),
        "frames": frame_results,
    }

    summary_path = os.path.join(out_root, "geometry_group_comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Compared {len(common_stems)} matched views.")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
