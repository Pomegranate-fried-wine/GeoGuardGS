import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np


CATEGORIES = [
    ("high_confidence", "high_confidence_candidate", 20),
    ("surface_align", "surface_align_candidate", 30),
    ("shrink", "shrink_candidate", 30),
    ("layer_conflict_low_support", None, 20),
    ("border_suspect", "border_suspect", 20),
    ("opacity_decay", "opacity_decay_candidate", 20),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate manual audit visuals for structure candidate tags.")
    parser.add_argument("--candidate-dir", default="output/local_formal/structure_candidates_v0_A")
    parser.add_argument("--geometry-error-map-dir", default="output/local_formal/geometry_error_map")
    parser.add_argument("--v0-dir", default="output/local_formal/gaussian_responsibility_v0_1_A")
    parser.add_argument("--v1-dir", default="output/local_formal/gaussian_responsibility_v1_A")
    parser.add_argument("--v15-dir", default="output/local_formal/responsibility_v1_5_A_compact")
    parser.add_argument("--image-dir", default="data/waymo/002/images")
    parser.add_argument("--render-dir", default="output/local_formal/p15_allcam_A_da3_only_5000/train/ours_5000")
    parser.add_argument(
        "--output-dir",
        default="output/local_formal/structure_candidates_v0_A/manual_check_visuals_render_compare",
    )
    parser.add_argument("--circle-radius-min", type=int, default=5)
    parser.add_argument("--circle-radius-max", type=int, default=36)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_u8(arr, valid=None):
    arr = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(arr) if valid is None else (valid & np.isfinite(arr))
    out = np.zeros(arr.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(arr[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        out[valid] = np.clip((arr[valid] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


def tag_lookup(tag_names):
    return {str(name): i for i, name in enumerate(tag_names.tolist())}


def select_candidates(candidate_npz, tag_idx):
    tags = candidate_npz["tag_matrix"].astype(bool)
    mean_resp = candidate_npz["mean_responsibility"]
    high_views = candidate_npz["high_error_view_counts"]
    radius = candidate_npz["mean_screen_radius"]
    selected = {}
    for category, tag_name, limit in CATEGORIES:
        if category == "layer_conflict_low_support":
            mask = tags[:, tag_idx["layer_conflict_high"]] & tags[:, tag_idx["low_support_uncertain"]]
        else:
            mask = tags[:, tag_idx[tag_name]]
        idxs = np.flatnonzero(mask)
        if idxs.size:
            order = np.lexsort((radius[idxs], mean_resp[idxs], high_views[idxs]))[::-1]
            idxs = idxs[order]
        selected[category] = idxs[:limit].astype(np.int64)
    return selected


def find_best_views(v0_dir, selected_ids):
    selected_ids = set(int(x) for x in selected_ids)
    best = {}
    for view_dir in sorted(Path(v0_dir).iterdir()):
        npz_path = view_dir / "gaussian_responsibility_v0.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        gids = data["gaussian_ids"].astype(np.int64)
        mask = np.isin(gids, list(selected_ids))
        if not np.any(mask):
            continue
        rows = np.flatnonzero(mask)
        for row in rows:
            gid = int(gids[row])
            score = float(data["responsibility_scores"][row])
            if gid not in best or score > best[gid]["responsibility_score"]:
                best[gid] = {
                    "view_id": view_dir.name,
                    "row": int(row),
                    "responsibility_score": score,
                    "center": data["screen_centers"][row].astype(np.float32),
                    "radius": float(data["radii"][row]),
                    "support_pixel_count": int(data["support_pixel_counts"][row]),
                    "boundary_overlap": float(data["boundary_overlaps"][row]),
                    "rendered_depth_edge_overlap": float(data["rendered_depth_edge_overlaps"][row]),
                    "thin_overlap": float(data["thin_structure_overlaps"][row]),
                    "mean_error": float(data["mean_errors"][row]),
                }
    return best


def load_components(root, stem):
    comp = np.load(os.path.join(root, stem, "geometry_error_components.npz"))
    error = np.load(os.path.join(root, stem, "geometry_error_map.npy")).astype(np.float32)
    data = {
        "error": error,
        "valid": comp["valid_lidar_mask"].astype(bool),
        "boundary": comp["boundary_mask"].astype(bool),
        "rendered_edge": comp["rendered_depth_edge_mask"].astype(bool),
        "thin": comp["thin_structure_mask"].astype(bool),
    }
    if "A_abs_error" in comp.files:
        data["A_abs_error"] = comp["A_abs_error"].astype(np.float32)
    return data


def support_valid_ratio(center, radius, valid, radius_min=1, radius_max=60):
    h, w = valid.shape
    cx, cy = float(center[0]), float(center[1])
    r = int(np.ceil(max(radius_min, min(radius_max, radius))))
    x0, x1 = max(0, int(np.floor(cx - r))), min(w, int(np.ceil(cx + r + 1)))
    y0, y1 = max(0, int(np.floor(cy - r))), min(h, int(np.ceil(cy + r + 1)))
    if x0 >= x1 or y0 >= y1:
        return None
    xs = np.arange(x0, x1, dtype=np.float32) + 0.5
    ys = np.arange(y0, y1, dtype=np.float32) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    support = ((gx - cx) ** 2 + (gy - cy) ** 2) <= float(r * r)
    if not np.any(support):
        return None
    return float(np.count_nonzero(valid[y0:y1, x0:x1] & support) / max(np.count_nonzero(support), 1))


def color_mask(mask, color):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = color
    return out


def draw_candidate(base, center, radius, args, color=(0, 255, 255)):
    out = base.copy()
    x, y = float(center[0]), float(center[1])
    r = int(max(args.circle_radius_min, min(args.circle_radius_max, radius)))
    cv2.circle(out, (int(round(x)), int(round(y))), r, color, 2, lineType=cv2.LINE_AA)
    cv2.circle(out, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    return out


def resize_to(image, shape):
    h, w = shape[:2]
    if image.shape[:2] == (h, w):
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def make_contact_sheet(images):
    h, w = images[0].shape[:2]
    thumb_w = 520
    thumb_h = int(round(h * thumb_w / max(w, 1)))
    resized = [cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA) for img in images]
    blank = np.zeros_like(resized[0])
    while len(resized) % 3:
        resized.append(blank)
    rows = []
    for idx in range(0, len(resized), 3):
        rows.append(np.hstack(resized[idx : idx + 3]))
    return np.vstack(rows)


def add_label(image, label):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 580), 34), (0, 0, 0), -1)
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def load_rendered_rgb(render_dir, stem, shape):
    candidates = [
        os.path.join(render_dir, f"{stem}_rgb.png"),
        os.path.join(render_dir, f"{stem}.png"),
    ]
    for path in candidates:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is not None:
            return resize_to(image, shape), path
    return None, None


def load_rendered_depth(render_dir, stem, shape):
    candidates = [
        os.path.join(render_dir, f"{stem}_depth.npy"),
        os.path.join(render_dir, f"{stem}_raw_depth.npy"),
    ]
    for path in candidates:
        if os.path.exists(path):
            depth = np.load(path).astype(np.float32)
            depth = resize_to(depth, shape)
            return cv2.applyColorMap(normalize_u8(depth, np.isfinite(depth) & (depth > 0)), cv2.COLORMAP_JET), path
    png_path = os.path.join(render_dir, f"{stem}_depth.png")
    image = cv2.imread(png_path, cv2.IMREAD_COLOR)
    if image is not None:
        return resize_to(image, shape), png_path
    return None, None


def rgb_abs_error(original, rendered):
    if rendered is None:
        return np.zeros_like(original)
    diff = np.mean(np.abs(original.astype(np.float32) - rendered.astype(np.float32)), axis=2)
    return cv2.applyColorMap(normalize_u8(diff), cv2.COLORMAP_TURBO)


def depth_error_or_lidar(comps):
    if "A_abs_error" in comps:
        err = comps["A_abs_error"].astype(np.float32)
        valid = comps["valid"]
        vis = cv2.applyColorMap(normalize_u8(err, valid), cv2.COLORMAP_TURBO)
        lidar_points = valid.astype(bool)
        if np.any(lidar_points):
            vis[lidar_points] = np.clip(
                vis[lidar_points].astype(np.float32) * 0.55 + np.array([255, 255, 255], dtype=np.float32) * 0.45,
                0,
                255,
            ).astype(np.uint8)
        return vis
    return color_mask(comps["valid"], (255, 255, 255))


def build_visuals(args, category, rec, view, out_dir):
    stem = view["view_id"]
    comps = load_components(args.geometry_error_map_dir, stem)
    rgb = cv2.imread(os.path.join(args.image_dir, f"{stem}.png"), cv2.IMREAD_COLOR)
    if rgb is None:
        rgb = np.zeros((*comps["error"].shape, 3), dtype=np.uint8)
    rgb = resize_to(rgb, comps["error"].shape)
    rendered_rgb, rendered_rgb_source = load_rendered_rgb(args.render_dir, stem, comps["error"].shape)
    rendered_depth, rendered_depth_source = load_rendered_depth(args.render_dir, stem, comps["error"].shape)
    if rendered_rgb is None:
        rendered_rgb = np.zeros_like(rgb)
    if rendered_depth is None:
        rendered_depth = np.zeros_like(rgb)

    error_vis = cv2.applyColorMap(normalize_u8(comps["error"], comps["valid"]), cv2.COLORMAP_TURBO)
    edge_base = cv2.addWeighted(
        rendered_rgb,
        0.55,
        np.maximum(color_mask(comps["boundary"], (0, 255, 255)), color_mask(comps["rendered_edge"], (255, 0, 255))),
        0.45,
        0,
    )
    thin_base = cv2.addWeighted(rendered_rgb, 0.6, color_mask(comps["thin"], (0, 255, 0)), 0.4, 0)
    rgb_error = rgb_abs_error(rgb, rendered_rgb)
    depth_error = depth_error_or_lidar(comps)

    center, radius = view["center"], view["radius"]
    original_rgb = add_label(rgb, "Original RGB")
    rendered_overlay = add_label(draw_candidate(rendered_rgb, center, radius, args), "Rendered RGB + selected Gaussian")
    rgb_error_overlay = add_label(draw_candidate(rgb_error, center, radius, args), "RGB absolute error")
    error_overlay = draw_candidate(error_vis, center, radius, args)
    edge_overlay = draw_candidate(edge_base, center, radius, args)
    thin_overlay = draw_candidate(thin_base, center, radius, args)
    depth_overlay = draw_candidate(rendered_depth, center, radius, args)
    depth_error_overlay = draw_candidate(depth_error, center, radius, args)
    contact = make_contact_sheet(
        [
            original_rgb,
            rendered_overlay,
            rgb_error_overlay,
            add_label(error_overlay, "Geometry error map"),
            add_label(depth_overlay, "Rendered depth"),
            add_label(depth_error_overlay, "Depth error / LiDAR valid points"),
            add_label(edge_overlay, "Boundary + rendered-depth-edge overlay"),
            add_label(thin_overlay, "Thin-structure overlay"),
        ]
    )

    prefix = f"{category}_gid{rec['gaussian_id']}_{stem}"
    paths = {
        "original_rgb_path": os.path.join(out_dir, f"{prefix}_original_rgb.png"),
        "rendered_rgb_overlay_path": os.path.join(out_dir, f"{prefix}_rendered_rgb_overlay.png"),
        "rgb_error_path": os.path.join(out_dir, f"{prefix}_rgb_abs_error.png"),
        "error_map_overlay_path": os.path.join(out_dir, f"{prefix}_error_overlay.png"),
        "rendered_depth_path": os.path.join(out_dir, f"{prefix}_rendered_depth.png"),
        "depth_error_path": os.path.join(out_dir, f"{prefix}_depth_error_lidar.png"),
        "edge_overlay_path": os.path.join(out_dir, f"{prefix}_edge_overlay.png"),
        "thin_overlay_path": os.path.join(out_dir, f"{prefix}_thin_overlay.png"),
        "contact_sheet_path": os.path.join(out_dir, f"{prefix}_contact_sheet.png"),
        "metadata_path": os.path.join(out_dir, f"{prefix}_metadata.json"),
    }
    cv2.imwrite(paths["original_rgb_path"], original_rgb)
    cv2.imwrite(paths["rendered_rgb_overlay_path"], rendered_overlay)
    cv2.imwrite(paths["rgb_error_path"], rgb_error_overlay)
    cv2.imwrite(paths["error_map_overlay_path"], error_overlay)
    cv2.imwrite(paths["rendered_depth_path"], depth_overlay)
    cv2.imwrite(paths["depth_error_path"], depth_error_overlay)
    cv2.imwrite(paths["edge_overlay_path"], edge_overlay)
    cv2.imwrite(paths["thin_overlay_path"], thin_overlay)
    cv2.imwrite(paths["contact_sheet_path"], contact)
    paths["rendered_rgb_source"] = rendered_rgb_source
    paths["rendered_depth_source"] = rendered_depth_source
    return paths, support_valid_ratio(center, radius, comps["valid"])


def as_record(candidate_npz, tag_names, tag_mat, idx):
    tags = [name for j, name in enumerate(tag_names) if tag_mat[idx, j]]
    return {
        "gaussian_id": int(candidate_npz["gaussian_ids"][idx]),
        "tags": tags,
        "mean_responsibility": float(candidate_npz["mean_responsibility"][idx]),
        "visible_view_count": int(candidate_npz["visible_view_counts"][idx]),
        "high_error_view_count": int(candidate_npz["high_error_view_counts"][idx]),
        "mean_support_pixel_count": float(candidate_npz["mean_support_pixel_count"][idx]),
        "boundary_overlap": float(candidate_npz["mean_boundary_overlap"][idx]),
        "rendered_depth_edge_overlap": float(candidate_npz["mean_rendered_depth_edge_overlap"][idx]),
        "thin_overlap": float(candidate_npz["mean_thin_structure_overlap"][idx]),
        "layer_conflict_score": float(candidate_npz["layer_conflict_score"][idx]),
        "border_suspect": bool("border_suspect" in tags),
        "low_support_uncertain": bool("low_support_uncertain" in tags),
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    candidate_npz = np.load(os.path.join(args.candidate_dir, "structure_candidates_v0.npz"), allow_pickle=True)
    summary = load_json(os.path.join(args.candidate_dir, "structure_candidates_summary.json"))
    tag_names = [str(x) for x in candidate_npz["tag_names"].tolist()]
    tag_idx = tag_lookup(candidate_npz["tag_names"])
    tag_mat = candidate_npz["tag_matrix"].astype(bool)

    selected = select_candidates(candidate_npz, tag_idx)
    selected_ids = set()
    for idxs in selected.values():
        for idx in idxs:
            selected_ids.add(int(candidate_npz["gaussian_ids"][idx]))
    best_views = find_best_views(args.v0_dir, selected_ids)

    audit_rows = []
    counts = {}
    missing = []
    candidate_id = 0
    for category, idxs in selected.items():
        out_dir = ensure_dir(os.path.join(args.output_dir, category))
        counts[category] = 0
        for idx in idxs:
            rec = as_record(candidate_npz, tag_names, tag_mat, int(idx))
            view = best_views.get(rec["gaussian_id"])
            if view is None:
                missing.append({"category": category, "gaussian_id": rec["gaussian_id"], "reason": "not found in v0 view files"})
                continue
            paths, valid_ratio = build_visuals(args, category, rec, view, out_dir)
            metadata = {
                "candidate_id": candidate_id,
                "gaussian_id": rec["gaussian_id"],
                "candidate_type": category,
                "tags": rec["tags"],
                "view_id": view["view_id"],
                "mean_responsibility": rec["mean_responsibility"],
                "visible_view_count": rec["visible_view_count"],
                "high_error_view_count": rec["high_error_view_count"],
                "support_pixel_count": view["support_pixel_count"],
                "support_valid_ratio": valid_ratio,
                "boundary_overlap": view["boundary_overlap"],
                "rendered_depth_edge_overlap": view["rendered_depth_edge_overlap"],
                "thin_overlap": view["thin_overlap"],
                "layer_conflict_score": rec["layer_conflict_score"],
                "border_suspect": rec["border_suspect"],
                "low_support_uncertain": rec["low_support_uncertain"],
                "contact_sheet_path": paths["contact_sheet_path"],
                "rendered_rgb_source": paths["rendered_rgb_source"],
                "rendered_depth_source": paths["rendered_depth_source"],
            }
            with open(paths["metadata_path"], "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            audit_rows.append(
                {
                    "candidate_id": candidate_id,
                    "gaussian_id": rec["gaussian_id"],
                    "category": category,
                    "tags": "|".join(rec["tags"]),
                    "view_id": view["view_id"],
                    "original_rgb_path": paths["original_rgb_path"],
                    "rendered_rgb_overlay_path": paths["rendered_rgb_overlay_path"],
                    "rgb_error_path": paths["rgb_error_path"],
                    "error_map_overlay_path": paths["error_map_overlay_path"],
                    "rendered_depth_path": paths["rendered_depth_path"],
                    "depth_error_path": paths["depth_error_path"],
                    "edge_overlay_path": paths["edge_overlay_path"],
                    "thin_overlay_path": paths["thin_overlay_path"],
                    "contact_sheet_path": paths["contact_sheet_path"],
                    "metadata_path": paths["metadata_path"],
                    "mean_responsibility": rec["mean_responsibility"],
                    "support_pixel_count": view["support_pixel_count"],
                    "support_valid_ratio": valid_ratio,
                    "border_suspect": rec["border_suspect"],
                    "low_support_uncertain": rec["low_support_uncertain"],
                    "manual_is_reasonable": "",
                    "manual_error_type": "",
                    "manual_notes": "",
                }
            )
            counts[category] += 1
            candidate_id += 1

    csv_path = os.path.join(args.output_dir, "audit_index.csv")
    fieldnames = [
        "candidate_id",
        "gaussian_id",
        "category",
        "tags",
        "view_id",
        "original_rgb_path",
        "rendered_rgb_overlay_path",
        "rgb_error_path",
        "error_map_overlay_path",
        "rendered_depth_path",
        "depth_error_path",
        "edge_overlay_path",
        "thin_overlay_path",
        "contact_sheet_path",
        "metadata_path",
        "mean_responsibility",
        "support_pixel_count",
        "support_valid_ratio",
        "border_suspect",
        "low_support_uncertain",
        "manual_is_reasonable",
        "manual_error_type",
        "manual_notes",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    audit_summary = {
        "candidate_dir": os.path.abspath(args.candidate_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "source_candidate_summary": {
            "normalized_verification_passed": summary.get("normalized_verification_passed"),
            "tag_counts": summary.get("tag_counts", {}),
        },
        "generated_counts": counts,
        "total_visualized_candidates": len(audit_rows),
        "audit_index_csv": os.path.abspath(csv_path),
        "missing_candidates": missing,
        "notes": "Manual fields in audit_index.csv are intentionally blank for human review. No Gaussian parameters were modified.",
    }
    with open(os.path.join(args.output_dir, "audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved manual-check visuals: {args.output_dir}")
    print(json.dumps({"generated_counts": counts, "missing": len(missing)}, indent=2))


if __name__ == "__main__":
    main()
