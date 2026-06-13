import argparse
import ast
import csv
import json
import os
from collections import Counter

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit A10000 post-training region chain and generate review pack for filtered regions."
    )
    parser.add_argument("--model-dir", default="output/local_formal/p15_allcam_A_da3_only_10000")
    parser.add_argument("--render-dir", default="output/local_formal/p15_allcam_A_da3_only_10000/train/ours_10000")
    parser.add_argument("--eval-dir", default="output/local_formal/p15_allcam_A_da3_only_10000/geometry_credibility_eval")
    parser.add_argument("--geometry-error-map-dir", default="output/local_formal/geometry_error_map_A10000")
    parser.add_argument("--v0-dir", default="output/local_formal/gaussian_responsibility_v0_1_A10000")
    parser.add_argument("--v1-dir", default="output/local_formal/gaussian_responsibility_v1_A10000")
    parser.add_argument("--v15-dir", default="output/local_formal/responsibility_v1_5_A10000_compact")
    parser.add_argument("--candidate-dir", default="output/local_formal/structure_candidates_v0_A10000")
    parser.add_argument("--output-dir", default="output/local_formal/audit_A10000_mainline_regions")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--crop-pad", type=int, default=48)
    parser.add_argument("--pilot-count", type=int, default=3)
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def to_int(value, default=0):
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except Exception:
        return default


def parse_bbox(value):
    if isinstance(value, (list, tuple)):
        vals = value
    else:
        vals = ast.literal_eval(str(value))
    return [int(round(float(v))) for v in vals]


def normalize_u8(array, mask=None, percentile=98.0):
    arr = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= mask.astype(bool)
    if valid.any():
        vmax = float(np.percentile(arr[valid], percentile))
    else:
        vmax = float(np.nanmax(arr)) if arr.size else 1.0
    vmax = max(vmax, 1e-6)
    out = np.clip(arr / vmax, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def color_map(array, mask=None):
    gray = normalize_u8(array, mask=mask)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def load_image(path):
    if not os.path.exists(path):
        return None
    return cv2.imread(path, cv2.IMREAD_COLOR)


def crop_with_pad(image, bbox, pad):
    if image is None:
        return None, None
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    return image[y1 : y2 + 1, x1 : x2 + 1].copy(), (x1, y1, x2, y2)


def draw_region(image, bbox, crop_box=None, label=None):
    if image is None:
        return None
    out = image.copy()
    x1, y1, x2, y2 = bbox
    if crop_box is not None:
        cx1, cy1, _, _ = crop_box
        x1, x2 = x1 - cx1, x2 - cx1
        y1, y2 = y1 - cy1, y2 - cy1
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.circle(out, ((x1 + x2) // 2, (y1 + y2) // 2), 5, (0, 0, 255), -1)
    if label:
        cv2.putText(out, label, (max(0, x1), max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return out


def stack_panel(images, labels):
    valid = [(img, lab) for img, lab in zip(images, labels) if img is not None]
    if not valid:
        return None
    h = max(img.shape[0] for img, _ in valid)
    w = max(img.shape[1] for img, _ in valid)
    tiles = []
    for img, lab in valid:
        canvas = np.zeros((h + 24, w, 3), dtype=np.uint8)
        resized = img
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        cv2.putText(canvas, lab, (6, h + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(canvas)
    rows = []
    for i in range(0, len(tiles), 3):
        row_tiles = tiles[i : i + 3]
        while len(row_tiles) < 3:
            row_tiles.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows)


def audit_masks(args):
    rows = []
    geom_root = args.geometry_error_map_dir
    for stem in sorted(os.listdir(geom_root)):
        view_dir = os.path.join(geom_root, stem)
        comp_path = os.path.join(view_dir, "geometry_error_components.npz")
        err_path = os.path.join(view_dir, "geometry_error_map.npy")
        if not os.path.isdir(view_dir) or not os.path.exists(comp_path) or not os.path.exists(err_path):
            continue
        comp = np.load(comp_path)
        err = np.load(err_path).astype(np.float32)
        valid = comp["valid_lidar_mask"].astype(bool) if "valid_lidar_mask" in comp.files else np.zeros(err.shape, bool)
        a_err = comp["A_abs_error"].astype(np.float32) if "A_abs_error" in comp.files else np.zeros_like(err)
        final = comp["final_geometry_error_map"].astype(np.float32) if "final_geometry_error_map" in comp.files else err
        boundary = comp["boundary_mask"].astype(bool) if "boundary_mask" in comp.files else np.zeros(err.shape, bool)
        canny = comp["canny_mask"].astype(bool) if "canny_mask" in comp.files else np.zeros(err.shape, bool)
        rendered_edge = comp["rendered_depth_edge_mask"].astype(bool) if "rendered_depth_edge_mask" in comp.files else np.zeros(err.shape, bool)
        thin = comp["thin_structure_mask"].astype(bool) if "thin_structure_mask" in comp.files else np.zeros(err.shape, bool)
        stable = valid & ~(boundary | canny | rendered_edge | thin)
        high = final >= np.quantile(final[valid], 0.9) if valid.any() else np.zeros(err.shape, bool)
        diff_terms_present = all(k in comp.files for k in ["B_minus_A", "D_minus_A", "D_minus_B"])
        final_equals_a = False
        if valid.any() and np.nanmax(a_err[valid]) > 0:
            a_norm = a_err / max(float(np.nanpercentile(a_err[valid], 98.0)), 1e-6)
            a_norm = np.clip(a_norm, 0.0, 1.0)
            final_equals_a = bool(np.nanmean(np.abs(final[valid] - a_norm[valid])) < 1e-3)
        rows.append(
            {
                "view_id": stem,
                "shape": f"{err.shape[0]}x{err.shape[1]}",
                "valid_lidar_count": int(valid.sum()),
                "boundary_count": int(boundary.sum()),
                "canny_count": int(canny.sum()),
                "rendered_depth_edge_count": int(rendered_edge.sum()),
                "thin_structure_count": int(thin.sum()),
                "stable_non_boundary_count": int(stable.sum()),
                "high_error_count": int(high.sum()),
                "high_error_boundary_ratio": float((high & boundary).sum() / max(high.sum(), 1)),
                "high_error_canny_ratio": float((high & canny).sum() / max(high.sum(), 1)),
                "high_error_rendered_edge_ratio": float((high & rendered_edge).sum() / max(high.sum(), 1)),
                "high_error_thin_ratio": float((high & thin).sum() / max(high.sum(), 1)),
                "mean_A_abs_error_valid": float(np.nanmean(a_err[valid])) if valid.any() else None,
                "mean_geometry_error_valid": float(np.nanmean(final[valid])) if valid.any() else None,
                "diff_terms_present": diff_terms_present,
                "final_map_equals_normalized_A_abs_error": final_equals_a,
                "component_keys": "|".join(comp.files),
            }
        )
    return rows


def summarize_region_rows(region_rows):
    total = len(region_rows)
    type_counts = Counter(r.get("region_type", "") for r in region_rows)
    uncertain_border = sum(1 for r in region_rows if r.get("region_type") == "uncertain_border_region")
    low_support = sum(1 for r in region_rows if r.get("region_type") == "low_support_region")
    return {
        "total_regions": total,
        "region_type_counts": dict(type_counts),
        "uncertain_border_ratio": uncertain_border / max(total, 1),
        "low_support_ratio": low_support / max(total, 1),
    }


def region_score(row):
    return (
        to_float(row.get("mean_responsibility"))
        * max(to_float(row.get("mean_geometry_error")), 1e-6)
        * np.log1p(max(to_int(row.get("candidate_count")), 1))
        * (1.0 - min(to_float(row.get("border_suspect_ratio")), 1.0) * 0.5)
        * (1.0 - min(to_float(row.get("low_support_uncertain_ratio")), 1.0) * 0.5)
    )


def make_review_pack(args, filtered_rows):
    out_dir = ensure_dir(os.path.join(args.output_dir, "top_region_review_pack"))
    review_rows = []
    ranked = sorted(filtered_rows, key=region_score, reverse=True)[: args.top_k]
    for rank, row in enumerate(ranked, 1):
        stem = row["view_id"]
        bbox = parse_bbox(row["bbox"])
        prefix = f"rank{rank:03d}_{stem}_region{row['region_id']}_{row['region_type']}"

        original = load_image(os.path.join(args.render_dir, f"{stem}_gt.png"))
        rendered = load_image(os.path.join(args.render_dir, f"{stem}_rgb.png"))
        depth = None
        depth_path = os.path.join(args.render_dir, f"{stem}_depth.npy")
        if os.path.exists(depth_path):
            depth = color_map(np.load(depth_path))
        geom = None
        a_abs = None
        comp_path = os.path.join(args.geometry_error_map_dir, stem, "geometry_error_components.npz")
        err_path = os.path.join(args.geometry_error_map_dir, stem, "geometry_error_map.npy")
        if os.path.exists(err_path):
            geom = color_map(np.load(err_path))
        if os.path.exists(comp_path):
            comp = np.load(comp_path)
            if "A_abs_error" in comp.files:
                valid = comp["valid_lidar_mask"].astype(bool) if "valid_lidar_mask" in comp.files else None
                a_abs = color_map(comp["A_abs_error"], mask=valid)

        rgb_error = None
        if original is not None and rendered is not None and original.shape == rendered.shape:
            diff = np.mean(np.abs(original.astype(np.float32) - rendered.astype(np.float32)), axis=2)
            rgb_error = color_map(diff)

        crops = []
        crop_box = None
        for img in [original, rendered, rgb_error, depth, geom, a_abs]:
            crop, cb = crop_with_pad(img, bbox, args.crop_pad) if img is not None else (None, None)
            if crop_box is None and cb is not None:
                crop_box = cb
            crops.append(crop)
        overlays = [draw_region(c, bbox, crop_box, f"{stem} r{row['region_id']}") if c is not None else None for c in crops]
        panel = stack_panel(
            overlays,
            ["original RGB", "rendered RGB", "RGB error", "rendered depth", "geometry error map", "A abs LiDAR error"],
        )
        panel_path = os.path.join(out_dir, f"{prefix}_panel.png")
        if panel is not None:
            cv2.imwrite(panel_path, panel)
        review = dict(row)
        review.update(
            {
                "rank": rank,
                "review_score": float(region_score(row)),
                "panel_path": os.path.abspath(panel_path) if panel is not None else "",
                "recommended_action_family": suggest_family(row),
            }
        )
        review_rows.append(review)
    return out_dir, review_rows


def suggest_family(row):
    region_type = row.get("region_type")
    tags = row.get("main_tags", "")
    if region_type == "thin_region" or to_int(row.get("surface_align_candidate_count")) > 0 or "thin_structure_responsible" in tags:
        return "surface-align pilot"
    if region_type == "layer_conflict_region" or to_int(row.get("layer_conflict_count")) > 0 or "layer_conflict_high" in tags:
        return "diagnosis only"
    if region_type == "boundary_region" or to_int(row.get("shrink_candidate_count")) > 0:
        return "shrink-only pilot"
    return "manual review"


def recommend_pilots(review_rows, count):
    recommendations = []
    for row in review_rows:
        family = row["recommended_action_family"]
        if family not in {"shrink-only pilot", "surface-align pilot"}:
            continue
        if to_float(row.get("border_suspect_ratio")) > 0.5 or to_float(row.get("low_support_uncertain_ratio")) > 0.5:
            continue
        reasons = [
            f"region_type={row.get('region_type')}",
            f"mean_depth_error={to_float(row.get('mean_depth_error')):.3f}",
            f"mean_geometry_error={to_float(row.get('mean_geometry_error')):.4f}",
            f"border_suspect_ratio={to_float(row.get('border_suspect_ratio')):.3f}",
            f"low_support_uncertain_ratio={to_float(row.get('low_support_uncertain_ratio')):.3f}",
        ]
        recommendations.append(
            {
                "rank": row["rank"],
                "view_id": row["view_id"],
                "region_id": row["region_id"],
                "bbox": row["bbox"],
                "candidate_count": to_int(row.get("candidate_count")),
                "recommended_pilot": family,
                "review_score": row["review_score"],
                "panel_path": row["panel_path"],
                "reasons": reasons,
            }
        )
        if len(recommendations) >= count:
            break
    return recommendations


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    mask_rows = audit_masks(args)
    mask_csv = os.path.join(args.output_dir, "a10000_mask_audit_by_view.csv")
    if mask_rows:
        write_csv(mask_csv, mask_rows, list(mask_rows[0].keys()))

    filtered_csv = os.path.join(args.candidate_dir, "region_candidates_v0_filtered", "filtered_region_candidates.csv")
    all_region_csv = os.path.join(args.candidate_dir, "region_candidates_v0", "region_candidates.csv")
    filtered_summary_path = os.path.join(args.candidate_dir, "region_candidates_v0_filtered", "filtered_region_summary.json")
    filter_config_path = os.path.join(args.candidate_dir, "region_candidates_v0_filtered", "region_filter_config.json")
    region_summary_path = os.path.join(args.candidate_dir, "region_candidates_v0", "region_candidates_summary.json")

    filtered_rows = read_csv(filtered_csv)
    all_region_rows = read_csv(all_region_csv)
    filtered_summary = load_json(filtered_summary_path, {})
    filter_config = load_json(filter_config_path, {})
    region_summary = load_json(region_summary_path, {})

    review_dir, review_rows = make_review_pack(args, filtered_rows)
    review_csv = os.path.join(args.output_dir, "a10000_top_region_review_summary.csv")
    if review_rows:
        fields = list(review_rows[0].keys())
        write_csv(review_csv, review_rows, fields)

    pilot_recs = recommend_pilots(review_rows, args.pilot_count)
    write_json(os.path.join(args.output_dir, "a10000_pilot_region_recommendations.json"), {"recommendations": pilot_recs})

    mask_summary = {
        "view_count": len(mask_rows),
        "total_valid_lidar_count": int(sum(r["valid_lidar_count"] for r in mask_rows)),
        "views_with_boundary_mask": int(sum(r["boundary_count"] > 0 for r in mask_rows)),
        "views_with_canny_mask": int(sum(r["canny_count"] > 0 for r in mask_rows)),
        "views_with_rendered_depth_edge_mask": int(sum(r["rendered_depth_edge_count"] > 0 for r in mask_rows)),
        "views_with_thin_structure_mask": int(sum(r["thin_structure_count"] > 0 for r in mask_rows)),
        "mean_high_error_boundary_ratio": float(np.mean([r["high_error_boundary_ratio"] for r in mask_rows])) if mask_rows else None,
        "mean_high_error_canny_ratio": float(np.mean([r["high_error_canny_ratio"] for r in mask_rows])) if mask_rows else None,
        "mean_high_error_rendered_edge_ratio": float(np.mean([r["high_error_rendered_edge_ratio"] for r in mask_rows])) if mask_rows else None,
        "mean_high_error_thin_ratio": float(np.mean([r["high_error_thin_ratio"] for r in mask_rows])) if mask_rows else None,
        "diff_terms_present_in_all_views": bool(mask_rows and all(r["diff_terms_present"] for r in mask_rows)),
        "final_map_equals_A_abs_error_in_all_views": bool(mask_rows and all(r["final_map_equals_normalized_A_abs_error"] for r in mask_rows)),
        "input_paths": {
            "geometry_error_map_dir": os.path.abspath(args.geometry_error_map_dir),
            "render_dir": os.path.abspath(args.render_dir),
        },
    }
    write_json(os.path.join(args.output_dir, "a10000_mask_audit_summary.json"), mask_summary)

    region_audit = {
        "all_regions": summarize_region_rows(all_region_rows),
        "filtered_regions": {
            "filtered_region_count": len(filtered_rows),
            "region_type_counts": dict(Counter(r.get("region_type", "") for r in filtered_rows)),
            "uncertain_border_ratio": sum(r.get("region_type") == "uncertain_border_region" for r in filtered_rows) / max(len(filtered_rows), 1),
            "low_support_ratio": sum(r.get("region_type") == "low_support_region" for r in filtered_rows) / max(len(filtered_rows), 1),
        },
        "source_summaries": {
            "region_candidates_summary": region_summary,
            "filtered_region_summary": filtered_summary,
            "region_filter_config": filter_config,
        },
        "review_pack_dir": os.path.abspath(review_dir),
        "top_review_csv": os.path.abspath(review_csv),
        "pilot_recommendations_json": os.path.abspath(os.path.join(args.output_dir, "a10000_pilot_region_recommendations.json")),
        "self_consistency": {
            "v0_dir": os.path.abspath(args.v0_dir),
            "v1_dir": os.path.abspath(args.v1_dir),
            "v15_dir": os.path.abspath(args.v15_dir),
            "candidate_dir": os.path.abspath(args.candidate_dir),
            "region_summary_uses_A10000_geometry_error_map": "geometry_error_map_A10000" in json.dumps(region_summary),
            "filter_config_uses_A10000_region_dir": "structure_candidates_v0_A10000" in json.dumps(filter_config),
            "final_geometry_error_map_contains_group_diff_terms": mask_summary["diff_terms_present_in_all_views"],
            "responsibility_chain_is_pure_A10000_lidar_error": False,
        },
    }
    write_json(os.path.join(args.output_dir, "a10000_region_audit_summary.json"), region_audit)

    notes = []
    if mask_summary["diff_terms_present_in_all_views"]:
        notes.append(
            "- `geometry_error_components.npz` contains `B_minus_A`, `D_minus_A`, and `D_minus_B`; "
            "`final_geometry_error_map` is not guaranteed to be pure A10000 LiDAR abs error."
        )
    if not mask_summary["final_map_equals_A_abs_error_in_all_views"]:
        notes.append(
            "- `final_geometry_error_map` does not match normalized `A_abs_error` across all views. "
            "For a strict A10000 mainline, build a self-error map from `A_abs_error` only before structure pilots."
        )
    if mask_summary["views_with_boundary_mask"] == 0 or mask_summary["views_with_thin_structure_mask"] == 0:
        notes.append("- Some mask families are empty across all audited views; inspect mask propagation before using overlap fields.")
    if notes:
        with open(os.path.join(args.output_dir, "fix_notes.md"), "w", encoding="utf-8") as f:
            f.write("# A10000 Audit Notes\n\n")
            f.write("\n".join(notes))
            f.write("\n")

    print(f"Audited {len(mask_rows)} geometry-error views.")
    print(f"Filtered regions reviewed: {len(filtered_rows)}")
    print(f"Pilot recommendations: {len(pilot_recs)}")
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
