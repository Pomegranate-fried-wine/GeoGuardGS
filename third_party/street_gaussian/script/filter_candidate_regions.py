import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Filter region candidates and run clustering sensitivity analysis.")
    parser.add_argument("--region-dir", default="output/local_formal/structure_candidates_v0_A/region_candidates_v0")
    parser.add_argument("--candidate-npz", default="output/local_formal/structure_candidates_v0_A/structure_candidates_v0.npz")
    parser.add_argument("--geometry-error-map-dir", default="output/local_formal/geometry_error_map")
    parser.add_argument("--v0-dir", default="output/local_formal/gaussian_responsibility_v0_1_A")
    parser.add_argument("--render-dir", default="output/local_formal/p15_allcam_A_da3_only_5000/train/ours_5000")
    parser.add_argument("--image-dir", default="data/waymo/002/images")
    parser.add_argument("--output-dir", default="output/local_formal/structure_candidates_v0_A/region_candidates_v0_filtered")
    parser.add_argument("--radii", nargs="+", type=int, default=[8, 12, 16, 24])
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=14)
    parser.add_argument("--cameras", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-border-ratio", type=float, default=0.5)
    parser.add_argument("--max-low-support-ratio", type=float, default=0.5)
    parser.add_argument("--max-area-ratio", type=float, default=0.2)
    parser.add_argument("--min-candidates", type=int, default=3)
    parser.add_argument("--high-depth-error", type=float, default=5.0)
    parser.add_argument("--high-geometry-error", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--save-visuals", action="store_true", default=True)
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def to_float(value, default=0.0):
    if value in (None, "", "None", "null"):
        return default
    try:
        return float(value)
    except Exception:
        return default


def to_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_regions(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["bbox_list"] = json.loads(row["bbox"])
        x0, y0, x1, y1 = row["bbox_list"]
        row["bbox_area"] = max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)
        row["image_area"] = 1066 * 1600
        row["bbox_area_ratio"] = row["bbox_area"] / row["image_area"]
        row["candidate_count_int"] = to_int(row["candidate_count"])
        row["mean_responsibility_float"] = to_float(row["mean_responsibility"])
        row["mean_geometry_error_float"] = to_float(row["mean_geometry_error"])
        row["mean_depth_error_float"] = to_float(row["mean_depth_error"])
        row["border_suspect_ratio_float"] = to_float(row["border_suspect_ratio"])
        row["low_support_uncertain_ratio_float"] = to_float(row["low_support_uncertain_ratio"])
        row["shrink_candidate_count_int"] = to_int(row["shrink_candidate_count"])
        row["main_tags_list"] = [x for x in row["main_tags"].split("|") if x]
    return rows


def filter_region(row, args):
    if row["region_type"] in {"uncertain_border_region", "low_support_region"}:
        return False, "excluded_region_type"
    if row["border_suspect_ratio_float"] > args.max_border_ratio:
        return False, "border_ratio"
    if row["low_support_uncertain_ratio_float"] > args.max_low_support_ratio:
        return False, "low_support_ratio"
    if row["bbox_area_ratio"] > args.max_area_ratio:
        return False, "area_ratio"
    if row["candidate_count_int"] < args.min_candidates:
        return False, "too_few_candidates"
    keep = (
        row["region_type"] in {"boundary_region", "general_high_error_region"}
        or row["shrink_candidate_count_int"] > 0
        or row["mean_geometry_error_float"] >= args.high_geometry_error
        or row["mean_depth_error_float"] >= args.high_depth_error
    )
    return (keep, "kept" if keep else "not_target")


def score_region(row):
    penalty = max(0.0, 1.0 - row["border_suspect_ratio_float"]) * max(0.0, 1.0 - row["low_support_uncertain_ratio_float"])
    return (
        row["mean_responsibility_float"]
        * max(row["mean_geometry_error_float"], 1e-6)
        * math.log1p(max(row["candidate_count_int"], 0))
        * penalty
    )


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


def resize_to(image, shape):
    if image is None:
        return None
    h, w = shape[:2]
    if image.shape[:2] == (h, w):
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def draw_region(base, row, color=(0, 255, 255)):
    out = base.copy()
    x0, y0, x1, y1 = row["bbox_list"]
    cv2.rectangle(out, (x0, y0), (x1, y1), color, 3)
    label = f"{row['view_id']} R{row['region_id']} {row['region_type']} score={row['review_score']:.4g}"
    cv2.putText(out, label, (x0, max(25, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def load_components(root, stem):
    path = os.path.join(root, stem, "geometry_error_components.npz")
    error_path = os.path.join(root, stem, "geometry_error_map.npy")
    if not os.path.exists(path) or not os.path.exists(error_path):
        return None
    data = np.load(path)
    return {
        "error": np.load(error_path).astype(np.float32),
        "valid": data["valid_lidar_mask"].astype(bool),
    }


def save_review_overlays(rows, args):
    out_dir = ensure_dir(os.path.join(args.output_dir, "selected_review_overlays"))
    for rank, row in enumerate(rows[: args.top_k], start=1):
        stem = row["view_id"]
        comps = load_components(args.geometry_error_map_dir, stem)
        if comps is None:
            continue
        shape = comps["error"].shape
        original = cv2.imread(os.path.join(args.image_dir, f"{stem}.png"), cv2.IMREAD_COLOR)
        rendered = cv2.imread(os.path.join(args.render_dir, f"{stem}_rgb.png"), cv2.IMREAD_COLOR)
        original = resize_to(original, shape) if original is not None else np.zeros((*shape, 3), dtype=np.uint8)
        rendered = resize_to(rendered, shape) if rendered is not None else np.zeros_like(original)
        rgb_err = cv2.applyColorMap(normalize_u8(np.mean(np.abs(original.astype(np.float32) - rendered.astype(np.float32)), axis=2)), cv2.COLORMAP_TURBO)
        geom = cv2.applyColorMap(normalize_u8(comps["error"], comps["valid"]), cv2.COLORMAP_TURBO)
        prefix = f"rank{rank:03d}_{stem}_region{row['region_id']}_{row['region_type']}"
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_original_rgb_overlay.png"), draw_region(original, row))
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_rendered_rgb_overlay.png"), draw_region(rendered, row))
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_rgb_error_overlay.png"), draw_region(rgb_err, row))
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_geometry_error_map_overlay.png"), draw_region(geom, row))
    return out_dir


def run_cluster_for_radius(radius, args):
    out_dir = os.path.join(args.output_dir, "sensitivity_runs", f"radius_{radius}")
    summary_path = os.path.join(out_dir, "region_candidates_summary.json")
    if args.skip_existing and os.path.exists(summary_path):
        return load_json(summary_path)
    cmd = [
        sys.executable,
        "script/cluster_candidate_regions.py",
        "--candidate-npz",
        args.candidate_npz,
        "--geometry-error-map-dir",
        args.geometry_error_map_dir,
        "--render-dir",
        args.render_dir,
        "--image-dir",
        args.image_dir,
        "--frame-start",
        str(args.frame_start),
        "--frame-end",
        str(args.frame_end),
        "--cameras",
        *[str(c) for c in args.cameras],
        "--output-dir",
        out_dir,
        "--cluster-radius",
        str(radius),
        "--min-candidates-per-region",
        str(args.min_candidates),
    ]
    if args.v0_dir:
        cmd.extend(["--v0-dir", args.v0_dir])
    subprocess.run(cmd, check=True)
    return load_json(summary_path)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    region_csv = os.path.join(args.region_dir, "region_candidates.csv")
    region_summary_path = os.path.join(args.region_dir, "region_candidates_summary.json")
    regions = load_regions(region_csv)
    original_summary = load_json(region_summary_path)

    kept = []
    excluded = Counter()
    for row in regions:
        ok, reason = filter_region(row, args)
        if ok:
            row["review_score"] = score_region(row)
            kept.append(row)
        else:
            excluded[reason] += 1
    kept = sorted(kept, key=lambda r: (r["review_score"], r["mean_depth_error_float"], r["candidate_count_int"]), reverse=True)

    fieldnames = list(regions[0].keys()) + ["review_score"] if regions else ["review_score"]
    drop_internal = {"bbox_list", "main_tags_list"}
    fieldnames = [f for f in fieldnames if f not in drop_internal]
    csv_rows = []
    for row in kept:
        out = {k: v for k, v in row.items() if k in fieldnames}
        csv_rows.append(out)
    filtered_csv = os.path.join(args.output_dir, "filtered_region_candidates.csv")
    write_csv(filtered_csv, csv_rows, fieldnames)

    top_review = kept[: args.top_k]
    with open(os.path.join(args.output_dir, "top_review_regions.json"), "w", encoding="utf-8") as f:
        json.dump(top_review, f, indent=2, ensure_ascii=False)
    overlay_dir = save_review_overlays(top_review, args) if args.save_visuals else None

    sensitivity = []
    if not args.skip_sensitivity:
        for radius in args.radii:
            summary = run_cluster_for_radius(radius, args)
            total = int(summary.get("total_regions", 0))
            type_counts = summary.get("region_type_counts", {})
            top = summary.get("top_regions", [])[:10]
            sensitivity.append(
                {
                    "cluster_radius": radius,
                    "total_regions": total,
                    "region_type_counts": type_counts,
                    "uncertain_border_region_ratio": float(type_counts.get("uncertain_border_region", 0) / max(total, 1)),
                    "boundary_region_count": int(type_counts.get("boundary_region", 0)),
                    "general_high_error_region_count": int(type_counts.get("general_high_error_region", 0)),
                    "top_region_keys": [f"{r['view_id']}:{r['region_id']}:{r['region_type']}" for r in top],
                    "has_nonborder_high_error_region": bool(
                        type_counts.get("boundary_region", 0) or type_counts.get("general_high_error_region", 0)
                    ),
                }
            )
    sens_json = os.path.join(args.output_dir, "sensitivity_summary.json")
    with open(sens_json, "w", encoding="utf-8") as f:
        json.dump(sensitivity, f, indent=2, ensure_ascii=False)
    sens_csv = os.path.join(args.output_dir, "sensitivity_summary.csv")
    with open(sens_csv, "w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "cluster_radius",
            "total_regions",
            "uncertain_border_region_ratio",
            "boundary_region_count",
            "general_high_error_region_count",
            "has_nonborder_high_error_region",
            "region_type_counts",
            "top_region_keys",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sensitivity:
            out = dict(row)
            out["region_type_counts"] = json.dumps(out["region_type_counts"], ensure_ascii=False)
            out["top_region_keys"] = "|".join(out["top_region_keys"])
            writer.writerow(out)

    config = {
        "input_region_dir": os.path.abspath(args.region_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "filters": {
            "exclude_region_types": ["uncertain_border_region", "low_support_region"],
            "max_border_ratio": args.max_border_ratio,
            "max_low_support_ratio": args.max_low_support_ratio,
            "max_area_ratio": args.max_area_ratio,
            "min_candidates": args.min_candidates,
            "high_depth_error": args.high_depth_error,
            "high_geometry_error": args.high_geometry_error,
            "score": "mean_responsibility * max(mean_geometry_error, 1e-6) * log(1 + candidate_count) * border/support penalties",
        },
        "sensitivity_radii": args.radii,
    }
    with open(os.path.join(args.output_dir, "region_filter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    filtered_summary = {
        "input_summary": {
            "total_regions": original_summary.get("total_regions"),
            "region_type_counts": original_summary.get("region_type_counts"),
            "views_processed": original_summary.get("views_processed"),
            "gaussian_id_schema": original_summary.get("gaussian_id_schema"),
        },
        "filtered_region_count": len(kept),
        "excluded_counts": dict(excluded),
        "filtered_region_type_counts": dict(Counter(r["region_type"] for r in kept)),
        "top_review_regions": top_review,
        "kept_000002_0": any(r["view_id"] == "000002_0" for r in kept),
        "selected_review_overlays": overlay_dir,
        "filtered_csv": os.path.abspath(filtered_csv),
        "sensitivity_summary": os.path.abspath(sens_json),
        "recommendation": (
            "Only a very small set of non-border/non-low-support review regions remains. Review overlays before any shrink-only experiment."
            if kept
            else "No safe review region remains; fix border/support rules before refinement."
        ),
    }
    with open(os.path.join(args.output_dir, "filtered_region_summary.json"), "w", encoding="utf-8") as f:
        json.dump(filtered_summary, f, indent=2, ensure_ascii=False)
    print(f"Filtered regions: {len(kept)}")
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
