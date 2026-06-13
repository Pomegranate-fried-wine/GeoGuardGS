import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


DEFAULT_INCLUDE_TAGS = [
    "high_confidence_candidate",
    "thin_structure_responsible",
    "layer_conflict_high",
    "stable_boundary_edge_conflict",
    "shrink_candidate",
    "surface_align_candidate",
    "opacity_decay_candidate",
    "border_suspect",
    "low_support_uncertain",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster per-view Gaussian candidates into local candidate regions.")
    parser.add_argument("--candidate-npz", default="output/local_formal/structure_candidates_v0_A/structure_candidates_v0.npz")
    parser.add_argument("--high-confidence-json", default="output/local_formal/structure_candidates_v0_A/high_confidence_candidates.json")
    parser.add_argument("--uncertain-json", default="output/local_formal/structure_candidates_v0_A/uncertain_candidates.json")
    parser.add_argument("--geometry-error-map-dir", default="output/local_formal/geometry_error_map")
    parser.add_argument("--v0-dir", default="output/local_formal/gaussian_responsibility_v0_1_A")
    parser.add_argument("--image-dir", default="data/waymo/002/images")
    parser.add_argument("--render-dir", default="output/local_formal/p15_allcam_A_da3_only_5000/train/ours_5000")
    parser.add_argument("--output-dir", default="output/local_formal/structure_candidates_v0_A/region_candidates_v0")
    parser.add_argument("--include-tags", nargs="+", default=DEFAULT_INCLUDE_TAGS)
    parser.add_argument("--views", nargs="+", default=None)
    parser.add_argument("--selected-views", nargs="+", default=None)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--cameras", nargs="+", type=int, default=None)
    parser.add_argument("--max-views", type=int, default=None)
    parser.add_argument("--cluster-radius", type=int, default=24)
    parser.add_argument("--min-candidates-per-region", type=int, default=3)
    parser.add_argument("--max-points-per-view", type=int, default=60000)
    parser.add_argument("--top-regions", type=int, default=50)
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--visual-top-views", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_views(args):
    if args.views:
        stems = args.views
    elif args.frame_start is not None and args.frame_end is not None:
        cameras = args.cameras or [0, 1, 2, 3, 4]
        stems = [f"{frame:06d}_{cam}" for frame in range(args.frame_start, args.frame_end + 1) for cam in cameras]
    else:
        stems = sorted(p.name for p in Path(args.v0_dir).iterdir() if (p / "gaussian_responsibility_v0.npz").exists())
    if args.selected_views:
        selected = set(args.selected_views)
        stems = [stem for stem in stems if stem in selected]
    if args.max_views is not None:
        stems = stems[: args.max_views]
    return stems


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
    h, w = shape[:2]
    if image is None:
        return None
    if image.shape[:2] == (h, w):
        return image
    interp = cv2.INTER_NEAREST if image.ndim == 2 else cv2.INTER_AREA
    return cv2.resize(image, (w, h), interpolation=interp)


def load_components(root, stem):
    view_dir = os.path.join(root, stem)
    comp_path = os.path.join(view_dir, "geometry_error_components.npz")
    error_path = os.path.join(view_dir, "geometry_error_map.npy")
    if not os.path.exists(comp_path) or not os.path.exists(error_path):
        return None
    comp = np.load(comp_path)
    data = {
        "error": np.load(error_path).astype(np.float32),
        "valid": comp["valid_lidar_mask"].astype(bool),
        "boundary": comp["boundary_mask"].astype(bool),
        "thin": comp["thin_structure_mask"].astype(bool),
        "rendered_edge": comp["rendered_depth_edge_mask"].astype(bool),
    }
    if "A_abs_error" in comp.files:
        data["depth_error"] = comp["A_abs_error"].astype(np.float32)
    return data


def load_rgb_sources(args, stem, shape):
    original = cv2.imread(os.path.join(args.image_dir, f"{stem}.png"), cv2.IMREAD_COLOR)
    rendered = cv2.imread(os.path.join(args.render_dir, f"{stem}_rgb.png"), cv2.IMREAD_COLOR)
    original = resize_to(original, shape) if original is not None else np.zeros((*shape, 3), dtype=np.uint8)
    rendered = resize_to(rendered, shape) if rendered is not None else None
    rgb_error = None
    if rendered is not None:
        diff = np.mean(np.abs(original.astype(np.float32) - rendered.astype(np.float32)), axis=2)
        rgb_error = cv2.applyColorMap(normalize_u8(diff), cv2.COLORMAP_TURBO)
    return original, rendered, rgb_error


def load_candidate_arrays(path):
    data = np.load(path, allow_pickle=True)
    tag_names = [str(x) for x in data["tag_names"].tolist()]
    tag_index = {name: idx for idx, name in enumerate(tag_names)}
    return data, tag_names, tag_index


def inspect_candidate_schema(candidate_data):
    if "uses_stable_gaussian_ids" not in candidate_data.files:
        return {
            "uses_stable_gaussian_ids": None,
            "warning": "Candidate npz does not record Gaussian id schema; region clusters may inherit legacy view-local ids.",
        }
    uses_stable = bool(np.asarray(candidate_data["uses_stable_gaussian_ids"]).reshape(-1)[0])
    return {
        "uses_stable_gaussian_ids": uses_stable,
        "warning": None
        if uses_stable
        else "Candidate npz was generated from legacy view-local ids; region clusters are not reliable for cross-view Gaussian identity.",
    }


def inspect_v0_schema(v0_dir, views):
    versions = []
    checked = 0
    for stem in views:
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
        else "V0 inputs use legacy view-local gaussian_ids; per-view positions can still be drawn, but region-to-global-Gaussian attribution is unreliable.",
    }


def per_view_candidate_rows(v0_data, candidate_data, gid_to_idx, include_mask, args):
    gids = v0_data["gaussian_ids"].astype(np.int64)
    global_rows = np.array([gid_to_idx.get(int(g), -1) for g in gids], dtype=np.int64)
    keep = (global_rows >= 0) & include_mask[np.maximum(global_rows, 0)]
    rows = np.flatnonzero(keep)
    if rows.size > args.max_points_per_view:
        scores = candidate_data["mean_responsibility"][global_rows[rows]]
        order = np.argsort(scores)[::-1][: args.max_points_per_view]
        rows = rows[order]
    return rows, global_rows[rows]


def cluster_points(centers, shape, radius):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for x, y in centers:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(mask, (int(round(x)), int(round(y))), radius, 1, -1)
    num, labels = cv2.connectedComponents(mask, connectivity=8)
    point_labels = []
    for x, y in centers:
        xi = int(np.clip(round(float(x)), 0, w - 1))
        yi = int(np.clip(round(float(y)), 0, h - 1))
        point_labels.append(int(labels[yi, xi]))
    return np.asarray(point_labels, dtype=np.int32), labels, num


def infer_region_type(stats):
    if stats["border_suspect_ratio"] >= 0.5:
        return "uncertain_border_region"
    if stats["low_support_uncertain_ratio"] >= 0.5:
        return "low_support_region"
    if stats["thin_candidate_count"] > 0 and stats["thin_candidate_count"] >= max(1, stats["candidate_count"] * 0.2):
        return "thin_region"
    if stats["layer_conflict_count"] > 0:
        return "layer_conflict_region"
    if "stable_boundary_edge_conflict" in stats["main_tags"] or stats["mean_boundary_overlap"] >= 0.5:
        return "boundary_region"
    return "general_high_error_region"


def region_stats(stem, region_id, point_rows, global_rows, v0_data, cdata, tag_names, tag_matrix, comps, rgb_error):
    centers = v0_data["screen_centers"][point_rows].astype(np.float32)
    xs, ys = centers[:, 0], centers[:, 1]
    h, w = comps["error"].shape
    x0, x1 = int(np.clip(np.floor(np.nanmin(xs)), 0, w - 1)), int(np.clip(np.ceil(np.nanmax(xs)), 0, w - 1))
    y0, y1 = int(np.clip(np.floor(np.nanmin(ys)), 0, h - 1)), int(np.clip(np.ceil(np.nanmax(ys)), 0, h - 1))
    tag_counts = Counter()
    for grow in global_rows:
        active = tag_matrix[grow].astype(bool)
        for idx, active_flag in enumerate(active):
            if active_flag:
                tag_counts[tag_names[idx]] += 1
    main_tags = [name for name, _ in tag_counts.most_common(5)]
    candidate_count = int(len(point_rows))
    center_int = np.round(centers).astype(np.int32)
    valid_center = (
        (center_int[:, 0] >= 0)
        & (center_int[:, 0] < w)
        & (center_int[:, 1] >= 0)
        & (center_int[:, 1] < h)
    )
    sampled_error = comps["error"][center_int[valid_center, 1], center_int[valid_center, 0]] if np.any(valid_center) else np.asarray([])
    mean_rgb_error = None
    if rgb_error is not None and np.any(valid_center):
        gray = cv2.cvtColor(rgb_error, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mean_rgb_error = float(np.mean(gray[center_int[valid_center, 1], center_int[valid_center, 0]]))
    mean_depth_error = None
    if "depth_error" in comps and np.any(valid_center):
        mean_depth_error = float(np.mean(comps["depth_error"][center_int[valid_center, 1], center_int[valid_center, 0]]))
    stats = {
        "view_id": stem,
        "region_id": int(region_id),
        "candidate_count": candidate_count,
        "bbox": [x0, y0, x1, y1],
        "center_x": float(np.nanmean(xs)),
        "center_y": float(np.nanmean(ys)),
        "mean_responsibility": float(np.mean(cdata["mean_responsibility"][global_rows])),
        "mean_geometry_error": float(np.mean(sampled_error)) if sampled_error.size else float(np.mean(v0_data["mean_errors"][point_rows])),
        "mean_rgb_error": mean_rgb_error,
        "mean_depth_error": mean_depth_error,
        "mean_boundary_overlap": float(np.mean(cdata["mean_boundary_overlap"][global_rows])),
        "mean_rendered_depth_edge_overlap": float(np.mean(cdata["mean_rendered_depth_edge_overlap"][global_rows])),
        "mean_thin_structure_overlap": float(np.mean(cdata["mean_thin_structure_overlap"][global_rows])),
        "main_tags": main_tags,
        "border_suspect_ratio": float(np.mean(tag_matrix[global_rows, tag_names.index("border_suspect")])) if "border_suspect" in tag_names else 0.0,
        "low_support_uncertain_ratio": float(np.mean(tag_matrix[global_rows, tag_names.index("low_support_uncertain")])) if "low_support_uncertain" in tag_names else 0.0,
        "thin_candidate_count": int(np.count_nonzero(tag_matrix[global_rows, tag_names.index("thin_structure_responsible")])) if "thin_structure_responsible" in tag_names else 0,
        "shrink_candidate_count": int(np.count_nonzero(tag_matrix[global_rows, tag_names.index("shrink_candidate")])) if "shrink_candidate" in tag_names else 0,
        "surface_align_candidate_count": int(np.count_nonzero(tag_matrix[global_rows, tag_names.index("surface_align_candidate")])) if "surface_align_candidate" in tag_names else 0,
        "layer_conflict_count": int(np.count_nonzero(tag_matrix[global_rows, tag_names.index("layer_conflict_high")])) if "layer_conflict_high" in tag_names else 0,
    }
    stats["region_type"] = infer_region_type(stats)
    return stats


def draw_regions(base, regions, color_by_type=True):
    colors = {
        "boundary_region": (0, 255, 255),
        "thin_region": (0, 255, 0),
        "layer_conflict_region": (0, 0, 255),
        "uncertain_border_region": (255, 0, 0),
        "low_support_region": (255, 128, 0),
        "general_high_error_region": (255, 255, 255),
    }
    out = base.copy()
    for region in regions:
        color = colors.get(region["region_type"], (255, 255, 255)) if color_by_type else (0, 255, 255)
        x0, y0, x1, y1 = region["bbox"]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            out,
            f"R{region['region_id']} n={region['candidate_count']} {region['region_type']}",
            (x0, max(18, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def save_visuals(args, stem, regions, comps):
    if not regions:
        return {}
    out_dir = ensure_dir(os.path.join(args.output_dir, "selected_view_overlays"))
    shape = comps["error"].shape
    original, rendered, rgb_error = load_rgb_sources(args, stem, shape)
    geom = cv2.applyColorMap(normalize_u8(comps["error"], comps["valid"]), cv2.COLORMAP_TURBO)
    paths = {}
    for name, image in [
        ("original_rgb_overlay", original),
        ("rendered_rgb_overlay", rendered if rendered is not None else np.zeros_like(original)),
        ("rgb_error_overlay", rgb_error if rgb_error is not None else np.zeros_like(original)),
        ("geometry_error_map_overlay", geom),
    ]:
        path = os.path.join(out_dir, f"{stem}_{name}.png")
        cv2.imwrite(path, draw_regions(image, regions))
        paths[name] = path
    return paths


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    summary_path = os.path.join(args.output_dir, "region_candidates_summary.json")
    if args.skip_existing and os.path.exists(summary_path):
        print(f"Skip existing: {summary_path}")
        return

    cdata, tag_names, tag_index = load_candidate_arrays(args.candidate_npz)
    tag_matrix = cdata["tag_matrix"].astype(bool)
    include_tags = [tag for tag in args.include_tags if tag in tag_index]
    include_mask = np.zeros(len(cdata["gaussian_ids"]), dtype=bool)
    for tag in include_tags:
        include_mask |= tag_matrix[:, tag_index[tag]]
    gid_to_idx = {int(g): i for i, g in enumerate(cdata["gaussian_ids"])}
    views = select_views(args)
    candidate_schema = inspect_candidate_schema(cdata)
    v0_schema = inspect_v0_schema(args.v0_dir, views)

    high_confidence = load_json(args.high_confidence_json)
    uncertain = load_json(args.uncertain_json)
    optional_inputs = {
        "high_confidence_json_found": high_confidence is not None,
        "uncertain_json_found": uncertain is not None,
        "render_dir_found": os.path.isdir(args.render_dir),
    }

    rows = []
    per_view_counts = {}
    missing_views = []
    visual_paths = {}
    for stem in views:
        npz_path = os.path.join(args.v0_dir, stem, "gaussian_responsibility_v0.npz")
        comps = load_components(args.geometry_error_map_dir, stem)
        if not os.path.exists(npz_path) or comps is None:
            missing_views.append(stem)
            continue
        v0 = np.load(npz_path)
        point_rows, global_rows = per_view_candidate_rows(v0, cdata, gid_to_idx, include_mask, args)
        if len(point_rows) == 0:
            per_view_counts[stem] = 0
            continue
        centers = v0["screen_centers"][point_rows].astype(np.float32)
        labels, _, _ = cluster_points(centers, comps["error"].shape, args.cluster_radius)
        _, _, rgb_error = load_rgb_sources(args, stem, comps["error"].shape)
        view_regions = []
        region_id = 0
        for label in sorted(set(labels.tolist())):
            if label <= 0:
                continue
            local = np.flatnonzero(labels == label)
            if len(local) < args.min_candidates_per_region:
                continue
            stats = region_stats(
                stem,
                region_id,
                point_rows[local],
                global_rows[local],
                v0,
                cdata,
                tag_names,
                tag_matrix,
                comps,
                rgb_error,
            )
            view_regions.append(stats)
            rows.append(stats)
            region_id += 1
        per_view_counts[stem] = len(view_regions)
        if args.save_visuals and view_regions:
            visual_paths[stem] = save_visuals(args, stem, view_regions, comps)

    csv_path = os.path.join(args.output_dir, "region_candidates.csv")
    fieldnames = [
        "view_id",
        "region_id",
        "candidate_count",
        "bbox",
        "center_x",
        "center_y",
        "mean_responsibility",
        "mean_geometry_error",
        "mean_rgb_error",
        "mean_depth_error",
        "mean_boundary_overlap",
        "mean_rendered_depth_edge_overlap",
        "mean_thin_structure_overlap",
        "main_tags",
        "border_suspect_ratio",
        "low_support_uncertain_ratio",
        "thin_candidate_count",
        "shrink_candidate_count",
        "surface_align_candidate_count",
        "layer_conflict_count",
        "region_type",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["bbox"] = json.dumps(out["bbox"])
            out["main_tags"] = "|".join(out["main_tags"])
            writer.writerow(out)

    region_type_counts = Counter(row["region_type"] for row in rows)
    top_regions = sorted(rows, key=lambda r: (r["mean_responsibility"], r["candidate_count"]), reverse=True)[: args.top_regions]
    summary = {
        "config": {
            "candidate_npz": os.path.abspath(args.candidate_npz),
            "geometry_error_map_dir": os.path.abspath(args.geometry_error_map_dir),
            "v0_dir": os.path.abspath(args.v0_dir),
            "output_dir": os.path.abspath(args.output_dir),
            "include_tags": include_tags,
            "cluster_radius": args.cluster_radius,
            "min_candidates_per_region": args.min_candidates_per_region,
            "max_points_per_view": args.max_points_per_view,
            "save_visuals": args.save_visuals,
        },
        "optional_inputs": optional_inputs,
        "gaussian_id_schema": {
            "candidate": candidate_schema,
            "v0": v0_schema,
        },
        "views_requested": len(views),
        "views_processed": int(sum(1 for v in per_view_counts.values() if v >= 0)),
        "views_with_regions": int(sum(1 for v in per_view_counts.values() if v > 0)),
        "missing_views": missing_views,
        "total_regions": int(len(rows)),
        "region_type_counts": dict(region_type_counts),
        "per_view_region_counts": per_view_counts,
        "top_regions": top_regions,
        "visual_paths": visual_paths,
        "csv_path": os.path.abspath(csv_path),
        "notes": "Diagnostic clustering only. No Gaussian parameters were modified.",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(args.output_dir, "region_candidates_arrays.npz"),
        view_ids=np.asarray([r["view_id"] for r in rows]),
        region_ids=np.asarray([r["region_id"] for r in rows], dtype=np.int32),
        candidate_counts=np.asarray([r["candidate_count"] for r in rows], dtype=np.int32),
        bboxes=np.asarray([r["bbox"] for r in rows], dtype=np.int32) if rows else np.zeros((0, 4), dtype=np.int32),
        centers=np.asarray([[r["center_x"], r["center_y"]] for r in rows], dtype=np.float32)
        if rows
        else np.zeros((0, 2), dtype=np.float32),
        mean_responsibility=np.asarray([r["mean_responsibility"] for r in rows], dtype=np.float32),
        mean_geometry_error=np.asarray([r["mean_geometry_error"] for r in rows], dtype=np.float32),
        region_types=np.asarray([r["region_type"] for r in rows]),
    )
    print(f"Processed {len(per_view_counts)} views, generated {len(rows)} regions.")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
