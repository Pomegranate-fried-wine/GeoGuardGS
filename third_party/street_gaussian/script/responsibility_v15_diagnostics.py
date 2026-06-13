import argparse
import json
import os

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Responsibility v1.5 diagnostics: border, thin ranking, layer conflict.")
    parser.add_argument("--v0-dir", required=True)
    parser.add_argument("--geometry-error-map-dir", required=True)
    parser.add_argument("--output-dir", default="output/local_smoke/responsibility_v1_5")
    parser.add_argument("--views", nargs="+", default=None)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--cameras", nargs="+", type=int, default=None)
    parser.add_argument("--rendered-depth-dir", default="output/local_smoke/p4_da3_only_3000_retry/train/ours_3000")
    parser.add_argument("--lidar-dir", default="data/waymo/002/lidar_depth")
    parser.add_argument("--image-dir", default="data/waymo/002/images")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--border-margin", type=float, default=None)
    parser.add_argument("--min-support-pixels", type=int, default=3)
    parser.add_argument("--min-valid-support-ratio", type=float, default=0.05)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--radius-min", type=int, default=1)
    parser.add_argument("--radius-max", type=int, default=60)
    parser.add_argument("--thin-lambda", type=float, default=2.0)
    parser.add_argument("--max-thin-candidates-per-view", type=int, default=30000)
    parser.add_argument("--max-conflict-candidates-per-view", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-visuals", action="store_true")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def inspect_v0_id_schema(v0_dir, stems):
    versions = []
    checked = 0
    for stem in stems:
        npz_path = os.path.join(v0_dir, stem, "gaussian_responsibility_v0.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        checked += 1
        if "stable_id_schema_version" in data.files:
            versions.append(int(np.asarray(data["stable_id_schema_version"]).reshape(-1)[0]))
        else:
            versions.append(0)
    unique_versions = sorted(set(versions))
    uses_stable = bool(unique_versions and min(unique_versions) >= 1)
    return {
        "checked_view_count": int(checked),
        "stable_id_schema_versions": unique_versions,
        "uses_stable_gaussian_ids": uses_stable,
        "legacy_view_local_id_inputs": int(sum(1 for version in versions if version <= 0)),
        "warning": None
        if uses_stable
        else "V0 inputs use legacy view-local gaussian_ids; v1.5 gaussian_id fields are not reliable for cross-view identity.",
    }


def select_views(args):
    if args.views:
        return args.views
    stems = []
    if args.frame_start is None or args.frame_end is None:
        raise ValueError("Provide --views or --frame-start/--frame-end.")
    cameras = args.cameras or [0, 1, 2, 3]
    for frame in range(args.frame_start, args.frame_end + 1):
        for cam in cameras:
            stems.append(f"{frame:06d}_{cam}")
    return stems


def load_components(root, stem):
    view_dir = os.path.join(root, stem)
    data = np.load(os.path.join(view_dir, "geometry_error_components.npz"))
    error_map = np.load(os.path.join(view_dir, "geometry_error_map.npy")).astype(np.float32)
    return {
        "error_map": error_map,
        "valid": data["valid_lidar_mask"].astype(bool),
        "boundary": data["boundary_mask"].astype(bool),
        "canny": data["canny_mask"].astype(bool),
        "rendered_edge": data["rendered_depth_edge_mask"].astype(bool),
        "thin": data["thin_structure_mask"].astype(bool),
    }


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


def resize_like(array, shape, interpolation):
    if array.shape[:2] == shape:
        return array
    return cv2.resize(array, (shape[1], shape[0]), interpolation=interpolation)


def support_window(center, radius, shape, args):
    h, w = shape
    cx, cy = float(center[0]), float(center[1])
    if not np.isfinite(cx) or not np.isfinite(cy):
        return None
    r = int(np.ceil(max(float(args.radius_min), float(radius) * args.radius_scale)))
    r = min(r, args.radius_max)
    x0, x1 = max(0, int(np.floor(cx - r))), min(w, int(np.ceil(cx + r + 1)))
    y0, y1 = max(0, int(np.floor(cy - r))), min(h, int(np.ceil(cy + r + 1)))
    if x0 >= x1 or y0 >= y1:
        return None
    xs = np.arange(x0, x1, dtype=np.float32) + 0.5
    ys = np.arange(y0, y1, dtype=np.float32) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    support = ((gx - cx) ** 2 + (gy - cy) ** 2) <= float(r * r)
    truncated = bool(cx - r < 0 or cy - r < 0 or cx + r >= w or cy + r >= h)
    return x0, y0, x1, y1, support, r, truncated


def support_stats(error_map, valid, masks, center, radius, args):
    win = support_window(center, radius, error_map.shape, args)
    if win is None:
        return None
    x0, y0, x1, y1, support, r, truncated = win
    valid_support = valid[y0:y1, x0:x1] & support
    support_pixels = int(np.count_nonzero(support))
    valid_pixels = int(np.count_nonzero(valid_support))
    if valid_pixels < args.min_support_pixels:
        return None
    valid_ratio = float(valid_pixels / max(support_pixels, 1))
    local_error = error_map[y0:y1, x0:x1]
    out = {
        "support_pixels": support_pixels,
        "valid_support_pixels": valid_pixels,
        "support_valid_ratio": valid_ratio,
        "window_truncated": truncated,
        "mean_error": float(np.mean(local_error[valid_support])),
        "max_error": float(np.max(local_error[valid_support])),
        "radius_used": int(r),
    }
    for name, mask in masks.items():
        local = mask[y0:y1, x0:x1] & valid_support
        out[f"{name}_support_pixels"] = int(np.count_nonzero(local))
        out[f"{name}_overlap"] = float(np.mean(mask[y0:y1, x0:x1][valid_support]))
        out[f"{name}_mean_error"] = float(np.mean(local_error[local])) if np.any(local) else None
    return out


def border_distance(centers, shape):
    h, w = shape
    x, y = centers[:, 0], centers[:, 1]
    return np.maximum(np.minimum.reduce([x, y, float(w - 1) - x, float(h - 1) - y]), 0.0).astype(np.float32)


def normalize_u8(values, valid=None):
    x = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(x) if valid is None else valid & np.isfinite(x)
    out = np.zeros(x.shape, dtype=np.uint8)
    if not np.any(valid):
        return out
    lo, hi = np.percentile(x[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    out[valid] = np.clip((x[valid] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


def draw_overlay(base, centers, radii, rows, color=(0, 0, 255)):
    image = base.copy()
    for row in rows:
        x, y = centers[row]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        r = int(max(2, min(24, radii[row])))
        cv2.circle(image, (int(round(x)), int(round(y))), r, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(image, (int(round(x)), int(round(y))), 2, color, -1, lineType=cv2.LINE_AA)
    return image


def save_histogram(path, values, title):
    canvas = np.full((460, 900, 3), 255, dtype=np.uint8)
    if len(values):
        hist, _ = np.histogram(values, bins=50)
        left, right, top, bottom = 60, 860, 40, 390
        cv2.rectangle(canvas, (left, top), (right, bottom), (0, 0, 0), 1)
        for i, count in enumerate(hist):
            x0 = int(left + (right - left) * i / len(hist))
            x1 = int(left + (right - left) * (i + 1) / len(hist))
            y = int(bottom - (bottom - top) * count / max(int(np.max(hist)), 1))
            cv2.rectangle(canvas, (x0, y), (x1 - 1, bottom), (170, 170, 170), -1)
    cv2.putText(canvas, title, (60, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.imwrite(path, canvas)


def audit_border(args, stems):
    out_dir = ensure_dir(os.path.join(args.output_dir, "border_audit"))
    camera2 = [stem for stem in stems if stem.endswith("_2")]
    all_rows = []
    near_stats = []
    for stem in camera2:
        npz_path = os.path.join(args.v0_dir, stem, "gaussian_responsibility_v0.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        comps = load_components(args.geometry_error_map_dir, stem)
        centers, radii = data["screen_centers"], data["radii"]
        scores, mean_errors = data["responsibility_scores"], data["mean_errors"]
        top_rows = data["topk_indices"][: args.top_k]
        distances = border_distance(centers, comps["error_map"].shape)
        margin = args.border_margin or max(20.0, min(comps["error_map"].shape) * 0.03)
        masks = {
            "boundary": comps["boundary"],
            "canny": comps["canny"],
            "rendered_edge": comps["rendered_edge"],
            "thin": comps["thin"],
        }
        truncated = []
        valid_ratios = []
        for row in top_rows:
            stats = support_stats(comps["error_map"], comps["valid"], masks, centers[row], radii[row], args)
            if stats is None:
                truncated.append(False)
                valid_ratios.append(0.0)
            else:
                truncated.append(stats["window_truncated"])
                valid_ratios.append(stats["support_valid_ratio"])
        top_dist = distances[top_rows]
        near = top_dist <= margin
        near_stats.append(
            {
                "stem": stem,
                "top_k": int(len(top_rows)),
                "near_border_margin": float(margin),
                "near_border_ratio": float(np.mean(near)),
                "mean_border_distance": float(np.mean(top_dist)),
                "median_border_distance": float(np.median(top_dist)),
                "window_truncated_ratio": float(np.mean(truncated)),
                "mean_support_valid_ratio": float(np.mean(valid_ratios)),
                "low_valid_support_ratio": float(np.mean(np.asarray(valid_ratios) < args.min_valid_support_ratio)),
                "near_border_mean_responsibility": float(np.mean(scores[top_rows][near])) if np.any(near) else None,
                "nonborder_mean_responsibility": float(np.mean(scores[top_rows][~near])) if np.any(~near) else None,
                "near_border_mean_geometry_error": float(np.mean(mean_errors[top_rows][near])) if np.any(near) else None,
                "nonborder_mean_geometry_error": float(np.mean(mean_errors[top_rows][~near])) if np.any(~near) else None,
            }
        )
        all_rows.extend(top_dist.tolist())
        if args.save_visuals:
            base = cv2.applyColorMap(normalize_u8(comps["error_map"], comps["valid"]), cv2.COLORMAP_TURBO)
            cv2.imwrite(os.path.join(out_dir, f"{stem}_near_border_topK_overlay.png"), draw_overlay(base, centers, radii, top_rows))

    summary = {
        "views": camera2,
        "per_view": near_stats,
        "recommendation": "consider border penalty or exclude_border_margin for camera-2 if near_border_ratio stays high in full run",
    }
    with open(os.path.join(out_dir, "border_audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "border_vs_nonborder_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"per_view": near_stats}, f, indent=2, ensure_ascii=False)
    save_histogram(os.path.join(out_dir, "border_distance_histogram.png"), np.asarray(all_rows), "top-K border distance")
    return summary


def rank_thin(args, stems):
    out_dir = ensure_dir(os.path.join(args.output_dir, "thin_structure_ranking"))
    thin_views = [stem for stem in stems if stem.endswith("_1")]
    records = []
    for stem in thin_views:
        data = np.load(os.path.join(args.v0_dir, stem, "gaussian_responsibility_v0.npz"))
        thin_overlap = data["thin_structure_overlaps"].astype(np.float32)
        candidate_rows = np.flatnonzero(thin_overlap > 0)
        if candidate_rows.size == 0:
            continue
        thin_support = np.maximum(
            np.rint(data["support_pixel_counts"][candidate_rows].astype(np.float32) * thin_overlap[candidate_rows]),
            1,
        ).astype(np.int32)
        # V0.1 stores per-support mean error; use it as a scale-aware proxy for thin mean error here.
        thin_mean_error = data["mean_errors"][candidate_rows].astype(np.float32)
        thin_score = thin_mean_error * (1.0 + args.thin_lambda * thin_overlap[candidate_rows])
        order = np.argsort(thin_score)[::-1][: args.max_thin_candidates_per_view]
        for local_idx in order:
            row = int(candidate_rows[local_idx])
            records.append(
                (
                    stem,
                    int(data["gaussian_ids"][row]),
                    int(row),
                    float(thin_score[local_idx]),
                    int(thin_support[local_idx]),
                    float(thin_overlap[row]),
                    float(thin_mean_error[local_idx]),
                    float(data["responsibility_scores"][row]),
                    float(data["radii"][row]),
                )
            )
    dtype = [
        ("stem", "U16"),
        ("gaussian_id", "i8"),
        ("row", "i8"),
        ("thin_responsibility_score", "f4"),
        ("thin_support_pixels", "i4"),
        ("thin_overlap", "f4"),
        ("thin_mean_error", "f4"),
        ("global_responsibility_score", "f4"),
        ("radius", "f4"),
    ]
    arr = np.asarray(records, dtype=dtype)
    order = np.argsort(arr["thin_responsibility_score"])[::-1] if len(arr) else np.array([], dtype=np.int64)
    top = arr[order[: args.top_k]]
    np.savez_compressed(os.path.join(out_dir, "thin_responsibility_topK.npz"), topK=top, all_records=arr)

    baseline = {}
    for stem in thin_views:
        data = np.load(os.path.join(args.v0_dir, stem, "gaussian_responsibility_v0.npz"))
        baseline[stem] = {
            "global_topK_thin_overlap": float(np.mean(data["thin_structure_overlaps"][data["topk_indices"][: args.top_k]])),
            "random_thin_overlap": float(np.mean(data["thin_structure_overlaps"][data["random_indices"][: args.top_k]])),
            "large_radius_thin_overlap": float(np.mean(data["thin_structure_overlaps"][data["large_radius_indices"][: args.top_k]])),
        }
    summary = {
        "views": thin_views,
        "candidate_count": int(len(arr)),
        "topK_mean_thin_score": float(np.mean(top["thin_responsibility_score"])) if len(top) else None,
        "topK_mean_thin_overlap": float(np.mean(top["thin_overlap"])) if len(top) else None,
        "topK_mean_thin_error": float(np.mean(top["thin_mean_error"])) if len(top) else None,
        "top_gaussians": [
            {
                "stem": str(item["stem"]),
                "gaussian_id": int(item["gaussian_id"]),
                "thin_score": float(item["thin_responsibility_score"]),
                "thin_overlap": float(item["thin_overlap"]),
                "thin_mean_error": float(item["thin_mean_error"]),
            }
            for item in top[:20]
        ],
    }
    with open(os.path.join(out_dir, "thin_responsibility_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "thin_vs_global_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    if args.save_visuals and len(top):
        stem = str(top[0]["stem"])
        data = np.load(os.path.join(args.v0_dir, stem, "gaussian_responsibility_v0.npz"))
        comps = load_components(args.geometry_error_map_dir, stem)
        rows = [int(x["row"]) for x in top if str(x["stem"]) == stem]
        base = cv2.applyColorMap(normalize_u8(comps["error_map"], comps["valid"]), cv2.COLORMAP_TURBO)
        cv2.imwrite(os.path.join(out_dir, "thin_topK_overlay_on_error_map.png"), draw_overlay(base, data["screen_centers"], data["radii"], rows))
        rgb = cv2.imread(os.path.join(args.image_dir, f"{stem}.png"), cv2.IMREAD_COLOR)
        if rgb is not None:
            cv2.imwrite(os.path.join(out_dir, "thin_topK_overlay_on_rgb.png"), draw_overlay(rgb, data["screen_centers"], data["radii"], rows))
    return summary


def diagnose_layer_conflict(args, stems):
    out_dir = ensure_dir(os.path.join(args.output_dir, "layer_conflict"))
    views = [stem for stem in stems if stem.endswith("_1")]
    rows = []
    for stem in views:
        data = np.load(os.path.join(args.v0_dir, stem, "gaussian_responsibility_v0.npz"))
        comps = load_components(args.geometry_error_map_dir, stem)
        rendered = np.load(os.path.join(args.rendered_depth_dir, f"{stem}_depth.npy")).astype(np.float32)
        rendered = resize_like(rendered, comps["error_map"].shape, cv2.INTER_LINEAR)
        lidar, lidar_mask = load_lidar(os.path.join(args.lidar_dir, f"{stem}.npy"))
        lidar = resize_like(lidar, comps["error_map"].shape, cv2.INTER_NEAREST)
        lidar_mask = resize_like(lidar_mask.astype(np.uint8), comps["error_map"].shape, cv2.INTER_NEAREST).astype(bool)
        gx = cv2.Sobel(rendered, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(rendered, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        masks = {"boundary": comps["boundary"], "canny": comps["canny"], "rendered_edge": comps["rendered_edge"], "thin": comps["thin"]}
        top_rows = data["topk_indices"][: args.max_conflict_candidates_per_view]
        for row in top_rows:
            win = support_window(data["screen_centers"][row], data["radii"][row], comps["error_map"].shape, args)
            if win is None:
                continue
            x0, y0, x1, y1, support, _, _ = win
            valid_support = comps["valid"][y0:y1, x0:x1] & support
            if np.count_nonzero(valid_support) < args.min_support_pixels:
                continue
            lidar_valid = lidar_mask[y0:y1, x0:x1] & support
            lidar_values = lidar[y0:y1, x0:x1][lidar_valid]
            depth_values = rendered[y0:y1, x0:x1][support]
            grad_values = grad[y0:y1, x0:x1][support]
            rows.append(
                {
                    "stem": stem,
                    "gaussian_id": int(data["gaussian_ids"][row]),
                    "responsibility_score": float(data["responsibility_scores"][row]),
                    "local_rendered_depth_variance": float(np.var(depth_values)),
                    "local_rendered_depth_gradient_mean": float(np.mean(grad_values)),
                    "local_lidar_depth_variance": float(np.var(lidar_values)) if lidar_values.size > 1 else None,
                    "boundary_overlap": float(data["boundary_overlaps"][row]),
                    "canny_overlap": float(data["canny_overlaps"][row]),
                    "rendered_depth_edge_overlap": float(data["rendered_depth_edge_overlaps"][row]),
                    "thin_overlap": float(data["thin_structure_overlaps"][row]),
                    "screen_radius": float(data["radii"][row]),
                    "support_pixel_count": int(np.count_nonzero(valid_support)),
                }
            )
    if rows:
        rv = np.asarray([r["local_rendered_depth_variance"] for r in rows], dtype=np.float32)
        rg = np.asarray([r["local_rendered_depth_gradient_mean"] for r in rows], dtype=np.float32)
        rvn = rv / max(float(np.percentile(rv, 95)), 1e-6)
        rgn = rg / max(float(np.percentile(rg, 95)), 1e-6)
        for idx, r in enumerate(rows):
            r["layer_conflict_score"] = float(np.clip(rvn[idx], 0, 1) + np.clip(rgn[idx], 0, 1) + r["boundary_overlap"])
    dtype = [(k, "O") for k in rows[0].keys()] if rows else [("empty", "i4")]
    np.savez_compressed(os.path.join(out_dir, "layer_conflict_v0.npz"), records=np.asarray([tuple(r.values()) for r in rows], dtype=dtype) if rows else np.asarray([], dtype=dtype))
    top = sorted(rows, key=lambda item: item.get("layer_conflict_score", 0), reverse=True)[:20]
    summary = {
        "views": views,
        "candidate_count": len(rows),
        "mean_layer_conflict_score": float(np.mean([r["layer_conflict_score"] for r in rows])) if rows else None,
        "top_conflict_gaussians": top,
    }
    with open(os.path.join(out_dir, "layer_conflict_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if args.save_visuals and top:
        stem = top[0]["stem"]
        data = np.load(os.path.join(args.v0_dir, stem, "gaussian_responsibility_v0.npz"))
        comps = load_components(args.geometry_error_map_dir, stem)
        gid_to_row = {int(g): i for i, g in enumerate(data["gaussian_ids"])}
        overlay_rows = [gid_to_row[r["gaussian_id"]] for r in top if r["stem"] == stem and r["gaussian_id"] in gid_to_row]
        base = cv2.applyColorMap(normalize_u8(comps["error_map"], comps["valid"]), cv2.COLORMAP_TURBO)
        cv2.imwrite(os.path.join(out_dir, "high_responsibility_high_conflict_overlay.png"), draw_overlay(base, data["screen_centers"], data["radii"], overlay_rows))
    return summary


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    stems = select_views(args)
    id_schema = inspect_v0_id_schema(args.v0_dir, stems)
    results = {
        "gaussian_id_schema": id_schema,
        "border": audit_border(args, stems),
        "thin": rank_thin(args, stems),
        "layer_conflict": diagnose_layer_conflict(args, stems),
    }
    with open(os.path.join(args.output_dir, "responsibility_v1_5_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved v1.5 diagnostics: {args.output_dir}")


if __name__ == "__main__":
    main()
