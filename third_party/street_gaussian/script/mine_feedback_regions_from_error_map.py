import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-error-map-dir", default="output/local_formal/geometry_error_map")
    parser.add_argument("--output-dir", default="output/local_feedback/feedback_region_mining_A5000")
    parser.add_argument("--views", nargs="+", default=None)
    parser.add_argument("--max-regions", type=int, default=30)
    parser.add_argument("--per-view-max-regions", type=int, default=3)
    parser.add_argument("--high-error-quantile", type=float, default=0.9)
    parser.add_argument("--min-valid-pixels", type=int, default=20)
    parser.add_argument("--min-area", type=int, default=16)
    parser.add_argument("--max-area-ratio", type=float, default=0.08)
    parser.add_argument("--dilate-radius", type=int, default=3)
    parser.add_argument("--fallback-patch-radius", type=int, default=24)
    parser.add_argument("--fallback-candidates-per-view", type=int, default=8)
    return parser.parse_args()


def classify_region(boundary_ratio, edge_ratio, thin_ratio):
    if thin_ratio >= 0.2:
        return "thin_high_error_region"
    if boundary_ratio >= 0.3 or edge_ratio >= 0.3:
        return "boundary_edge_high_error_region"
    return "general_lidar_high_error_region"


def load_components(path):
    data = np.load(path)
    final_error = np.asarray(data["final_geometry_error_map"], dtype=np.float32)
    valid = np.asarray(data["valid_lidar_mask"]).astype(bool)
    a_abs = np.asarray(data["A_abs_error"], dtype=np.float32) if "A_abs_error" in data.files else final_error
    masks = {
        "boundary": np.asarray(data["boundary_mask"]).astype(bool) if "boundary_mask" in data.files else np.zeros_like(valid),
        "canny": np.asarray(data["canny_mask"]).astype(bool) if "canny_mask" in data.files else np.zeros_like(valid),
        "rendered_depth_edge": np.asarray(data["rendered_depth_edge_mask"]).astype(bool)
        if "rendered_depth_edge_mask" in data.files
        else np.zeros_like(valid),
        "thin": np.asarray(data["thin_structure_mask"]).astype(bool) if "thin_structure_mask" in data.files else np.zeros_like(valid),
    }
    return final_error, a_abs, valid, masks


def mine_view(view_dir, args):
    stem = view_dir.name
    npz_path = view_dir / "geometry_error_components.npz"
    if not npz_path.exists():
        return []
    final_error, a_abs, valid, masks = load_components(npz_path)
    usable = valid & np.isfinite(final_error) & np.isfinite(a_abs) & (a_abs > 0)
    if not np.any(usable):
        return []

    threshold = float(np.quantile(final_error[usable], args.high_error_quantile))
    high = usable & (final_error >= threshold)
    if args.dilate_radius > 0:
        k = 2 * args.dilate_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        high = cv2.dilate(high.astype(np.uint8), kernel, iterations=1).astype(bool) & usable

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(high.astype(np.uint8), 8)
    h, w = high.shape
    max_area = float(h * w * args.max_area_ratio)
    rows = []
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if area < args.min_area or area > max_area:
            continue
        component = labels == label
        valid_count = int(np.count_nonzero(component & valid))
        if valid_count < args.min_valid_pixels:
            continue
        err_vals = final_error[component & valid]
        depth_vals = a_abs[component & valid]
        if err_vals.size == 0:
            continue
        boundary_ratio = float(np.mean(masks["boundary"][component]))
        canny_ratio = float(np.mean(masks["canny"][component]))
        edge_ratio = float(np.mean(masks["rendered_depth_edge"][component]))
        thin_ratio = float(np.mean(masks["thin"][component]))
        mean_error = float(np.mean(err_vals))
        mean_depth_error = float(np.mean(depth_vals))
        score = float(valid_count * mean_error * np.log1p(mean_depth_error))
        region_type = classify_region(boundary_ratio, max(canny_ratio, edge_ratio), thin_ratio)
        rows.append(
            {
                "view_id": stem,
                "region_id": f"auto{label}",
                "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                "region_type": region_type,
                "review_score": score,
                "valid_lidar_high_error_pixels": valid_count,
                "mean_geometry_error": mean_error,
                "mean_depth_error": mean_depth_error,
                "boundary_ratio": boundary_ratio,
                "canny_ratio": canny_ratio,
                "rendered_depth_edge_ratio": edge_ratio,
                "thin_ratio": thin_ratio,
                "area": int(area),
            }
        )
    rows.sort(key=lambda item: item["review_score"], reverse=True)
    if len(rows) >= args.per_view_max_regions:
        return rows[: args.per_view_max_regions]

    covered = np.zeros_like(valid, dtype=bool)
    for row in rows:
        x0, y0, x1, y1 = row["bbox"]
        covered[y0:y1, x0:x1] = True

    ys, xs = np.where(usable & ~covered)
    if xs.size == 0:
        return rows[: args.per_view_max_regions]
    order = np.argsort(final_error[ys, xs])[::-1]
    radius = int(args.fallback_patch_radius)
    fallback_added = 0
    for idx in order:
        if fallback_added >= args.fallback_candidates_per_view or len(rows) >= args.per_view_max_regions:
            break
        cx, cy = int(xs[idx]), int(ys[idx])
        if covered[cy, cx]:
            continue
        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)
        component = np.zeros_like(valid, dtype=bool)
        component[y0:y1, x0:x1] = True
        component &= usable
        valid_count = int(np.count_nonzero(component))
        if valid_count < args.min_valid_pixels:
            continue
        err_vals = final_error[component]
        depth_vals = a_abs[component]
        boundary_ratio = float(np.mean(masks["boundary"][component]))
        canny_ratio = float(np.mean(masks["canny"][component]))
        edge_ratio = float(np.mean(masks["rendered_depth_edge"][component]))
        thin_ratio = float(np.mean(masks["thin"][component]))
        mean_error = float(np.mean(err_vals))
        mean_depth_error = float(np.mean(depth_vals))
        score = float(valid_count * mean_error * np.log1p(mean_depth_error))
        region_type = classify_region(boundary_ratio, max(canny_ratio, edge_ratio), thin_ratio)
        rows.append(
            {
                "view_id": stem,
                "region_id": f"patch{fallback_added}_{cx}_{cy}",
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                "region_type": region_type,
                "review_score": score,
                "valid_lidar_high_error_pixels": valid_count,
                "mean_geometry_error": mean_error,
                "mean_depth_error": mean_depth_error,
                "boundary_ratio": boundary_ratio,
                "canny_ratio": canny_ratio,
                "rendered_depth_edge_ratio": edge_ratio,
                "thin_ratio": thin_ratio,
                "area": int(np.count_nonzero(component)),
            }
        )
        covered[y0:y1, x0:x1] = True
        fallback_added += 1

    rows.sort(key=lambda item: item["review_score"], reverse=True)
    return rows[: args.per_view_max_regions]


def main():
    args = parse_args()
    geom_root = Path(args.geometry_error_map_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    view_dirs = [p for p in geom_root.iterdir() if p.is_dir()]
    if args.views:
        allowed = set(args.views)
        view_dirs = [p for p in view_dirs if p.name in allowed]
    view_dirs.sort(key=lambda p: p.name)

    rows = []
    per_view_counts = {}
    for view_dir in view_dirs:
        mined = mine_view(view_dir, args)
        per_view_counts[view_dir.name] = len(mined)
        rows.extend(mined)
    rows.sort(key=lambda item: item["review_score"], reverse=True)
    rows = rows[: args.max_regions]

    csv_path = out_dir / "feedback_regions_75view.csv"
    fields = [
        "view_id",
        "region_id",
        "bbox",
        "region_type",
        "review_score",
        "valid_lidar_high_error_pixels",
        "mean_geometry_error",
        "mean_depth_error",
        "boundary_ratio",
        "canny_ratio",
        "rendered_depth_edge_ratio",
        "thin_ratio",
        "area",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["bbox"] = json.dumps(row["bbox"])
            writer.writerow(row)

    summary = {
        "config": vars(args),
        "geometry_error_map_dir": os.path.abspath(args.geometry_error_map_dir),
        "processed_views": len(view_dirs),
        "selected_regions": len(rows),
        "per_view_mined_region_count": per_view_counts,
        "top_regions": rows[:20],
        "csv_path": os.path.abspath(csv_path),
    }
    summary_path = out_dir / "feedback_region_mining_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved mined feedback regions: {csv_path}")
    print(f"Selected regions: {len(rows)} from {len(view_dirs)} views")


if __name__ == "__main__":
    main()
