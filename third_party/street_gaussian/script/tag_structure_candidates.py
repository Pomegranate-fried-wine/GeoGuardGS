import argparse
import json
import os
from collections import Counter, defaultdict

import cv2
import numpy as np


TAG_NAMES = [
    "multi_view_persistent",
    "global_responsibility_high",
    "thin_structure_responsible",
    "layer_conflict_high",
    "stable_boundary_edge_conflict",
    "border_suspect",
    "low_support_uncertain",
    "high_confidence_candidate",
    "split_candidate",
    "shrink_candidate",
    "surface_align_candidate",
    "opacity_decay_candidate",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="V2-pre candidate-only Gaussian structure tagging with confidence guards."
    )
    parser.add_argument("--v1-dir", default="output/local_formal/gaussian_responsibility_v1_A")
    parser.add_argument("--v15-dir", default="output/local_formal/responsibility_v1_5_A_compact")
    parser.add_argument("--v0-dir", default="output/local_formal/gaussian_responsibility_v0_1_A")
    parser.add_argument("--geometry-error-map-dir", default="output/local_formal/geometry_error_map")
    parser.add_argument("--image-dir", default="data/waymo/002/images")
    parser.add_argument("--normalized-verification", default="output/local_formal_norm_eval/normalized_post_training_verification_summary.json")
    parser.add_argument("--output-dir", default="output/local_formal/structure_candidates_v0_A")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--global-percentile", type=float, default=90.0)
    parser.add_argument("--thin-percentile", type=float, default=90.0)
    parser.add_argument("--layer-percentile", type=float, default=90.0)
    parser.add_argument("--boundary-edge-percentile", type=float, default=90.0)
    parser.add_argument("--radius-percentile", type=float, default=85.0)
    parser.add_argument("--min-visible-views", type=int, default=3)
    parser.add_argument("--min-high-error-views", type=int, default=3)
    parser.add_argument("--high-error-view-ratio", type=float, default=0.35)
    parser.add_argument("--support-pixel-threshold", type=int, default=10)
    parser.add_argument("--thin-support-threshold", type=int, default=10)
    parser.add_argument("--layer-support-threshold", type=int, default=10)
    parser.add_argument("--border-margin", type=float, default=32.0)
    parser.add_argument("--severe-border-margin", type=float, default=8.0)
    parser.add_argument("--border-risk-cameras", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--border-risk-views", nargs="*", default=["000009_2", "000010_2", "000012_2", "000014_2"])
    parser.add_argument("--max-json-records", type=int, default=100)
    parser.add_argument("--selected-candidate-ids", nargs="*", type=int, default=[])
    parser.add_argument("--selected-views", nargs="*", default=[])
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--overlay-per-type", type=int, default=3)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def percentile(values, pct):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, pct)) if values.size else float("inf")


def parse_view_camera(stem):
    try:
        return int(str(stem).split("_")[-1])
    except Exception:
        return -1


def load_v1(v1_dir):
    path = os.path.join(v1_dir, "gaussian_responsibility_global.npz")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_v1_schema(v1_dir):
    summary_path = os.path.join(v1_dir, "global_responsibility_summary.json")
    schema = {
        "uses_stable_gaussian_ids": None,
        "stable_id_schema_versions": [],
        "legacy_view_local_id_inputs": None,
        "warning": "global_responsibility_summary.json not found; id schema could not be verified.",
    }
    if os.path.exists(summary_path):
        summary = load_json(summary_path)
        schema.update(summary.get("gaussian_id_schema", {}))
        schema["warning"] = summary.get("id_schema_warning")
    npz_path = os.path.join(v1_dir, "gaussian_responsibility_global.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        if "uses_stable_gaussian_ids" in data.files:
            schema["uses_stable_gaussian_ids"] = bool(np.asarray(data["uses_stable_gaussian_ids"]).reshape(-1)[0])
        if "stable_id_schema_versions" in data.files:
            schema["stable_id_schema_versions"] = [int(x) for x in data["stable_id_schema_versions"].tolist()]
        if "legacy_view_local_id_inputs" in data.files:
            schema["legacy_view_local_id_inputs"] = int(np.asarray(data["legacy_view_local_id_inputs"]).reshape(-1)[0])
    if schema.get("uses_stable_gaussian_ids") is False and not schema.get("warning"):
        schema["warning"] = "V1 uses legacy view-local ids; candidate tags are not reliable for cross-view Gaussian identity."
    if schema.get("uses_stable_gaussian_ids") is None and not schema.get("warning"):
        schema["warning"] = "V1 output does not record Gaussian id schema; treat it as legacy until V0/V1 are regenerated with stable ids."
    return schema


def aggregate_v0_support(v0_dir, gaussian_ids, args):
    gid_to_idx = {int(g): i for i, g in enumerate(gaussian_ids)}
    n = len(gaussian_ids)
    support_sum = np.zeros(n, dtype=np.float64)
    support_count = np.zeros(n, dtype=np.int32)
    min_border = np.full(n, np.inf, dtype=np.float32)
    border_hit_count = np.zeros(n, dtype=np.int32)
    severe_border_hit_count = np.zeros(n, dtype=np.int32)
    risk_view_hit_count = np.zeros(n, dtype=np.int32)
    camera_counter = defaultdict(Counter)
    view_counter = defaultdict(list)
    top_rows_by_view = {}

    for stem in sorted(os.listdir(v0_dir)):
        npz_path = os.path.join(v0_dir, stem, "gaussian_responsibility_v0.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        gids = data["gaussian_ids"].astype(np.int64)
        rows = np.array([gid_to_idx.get(int(g), -1) for g in gids], dtype=np.int64)
        valid = rows >= 0
        if not np.any(valid):
            continue
        rows = rows[valid]
        support = data["support_pixel_counts"].astype(np.float32)[valid]
        border = data["border_distances"].astype(np.float32)[valid] if "border_distances" in data else np.full_like(support, np.inf)
        np.add.at(support_sum, rows, support)
        np.add.at(support_count, rows, 1)
        np.minimum.at(min_border, rows, border)
        np.add.at(border_hit_count, rows, (border < args.border_margin).astype(np.int32))
        np.add.at(severe_border_hit_count, rows, (border < args.severe_border_margin).astype(np.int32))
        cam = parse_view_camera(stem)
        risk_view = cam in set(args.border_risk_cameras) or stem in set(args.border_risk_views)
        if risk_view:
            np.add.at(risk_view_hit_count, rows, 1)
        for row, gid in zip(rows, gids[valid]):
            camera_counter[int(gid)][cam] += 1
            if len(view_counter[int(gid)]) < 20:
                view_counter[int(gid)].append(stem)
        if "topk_indices" in data:
            top_rows_by_view[stem] = [int(x) for x in data["gaussian_ids"][data["topk_indices"][: args.top_k]].tolist()]

    mean_support = support_sum / np.maximum(support_count, 1)
    return {
        "mean_support_pixel_count": mean_support.astype(np.float32),
        "support_observation_count": support_count,
        "min_border_distance": min_border,
        "border_hit_count": border_hit_count,
        "severe_border_hit_count": severe_border_hit_count,
        "risk_view_hit_count": risk_view_hit_count,
        "camera_counter": camera_counter,
        "view_counter": view_counter,
        "top_rows_by_view": top_rows_by_view,
    }


def aggregate_thin(v15_dir, gaussian_ids):
    gid_to_idx = {int(g): i for i, g in enumerate(gaussian_ids)}
    n = len(gaussian_ids)
    max_score = np.zeros(n, dtype=np.float32)
    max_overlap = np.zeros(n, dtype=np.float32)
    max_error = np.zeros(n, dtype=np.float32)
    max_support = np.zeros(n, dtype=np.int32)
    view_count = np.zeros(n, dtype=np.int32)
    examples = defaultdict(list)
    path = os.path.join(v15_dir, "thin_structure_ranking", "thin_responsibility_topK.npz")
    if not os.path.exists(path):
        return max_score, max_overlap, max_error, max_support, view_count, examples
    data = np.load(path, allow_pickle=True)
    records = data["all_records"] if "all_records" in data.files else data["topK"]
    seen = set()
    for rec in records:
        gid = int(rec["gaussian_id"])
        idx = gid_to_idx.get(gid)
        if idx is None:
            continue
        score = float(rec["thin_responsibility_score"])
        overlap = float(rec["thin_overlap"])
        err = float(rec["thin_mean_error"])
        support = int(rec["thin_support_pixels"])
        max_score[idx] = max(max_score[idx], score)
        max_overlap[idx] = max(max_overlap[idx], overlap)
        max_error[idx] = max(max_error[idx], err)
        max_support[idx] = max(max_support[idx], support)
        key = (gid, str(rec["stem"]))
        if key not in seen:
            view_count[idx] += 1
            seen.add(key)
        if len(examples[gid]) < 5:
            examples[gid].append(
                {
                    "stem": str(rec["stem"]),
                    "thin_score": score,
                    "thin_overlap": overlap,
                    "thin_mean_error": err,
                    "thin_support_pixels": support,
                }
            )
    return max_score, max_overlap, max_error, max_support, view_count, examples


def aggregate_layer(v15_dir, gaussian_ids):
    gid_to_idx = {int(g): i for i, g in enumerate(gaussian_ids)}
    n = len(gaussian_ids)
    max_score = np.zeros(n, dtype=np.float32)
    max_depth_var = np.zeros(n, dtype=np.float32)
    max_depth_grad = np.zeros(n, dtype=np.float32)
    max_support = np.zeros(n, dtype=np.int32)
    view_count = np.zeros(n, dtype=np.int32)
    examples = defaultdict(list)
    path = os.path.join(v15_dir, "layer_conflict", "layer_conflict_v0.npz")
    if not os.path.exists(path):
        return max_score, max_depth_var, max_depth_grad, max_support, view_count, examples
    records = np.load(path, allow_pickle=True)["records"]
    seen = set()
    for rec in records:
        gid = int(rec["gaussian_id"])
        idx = gid_to_idx.get(gid)
        if idx is None:
            continue
        score = float(rec["layer_conflict_score"])
        depth_var = float(rec["local_rendered_depth_variance"])
        depth_grad = float(rec["local_rendered_depth_gradient_mean"])
        support = int(rec["support_pixel_count"])
        max_score[idx] = max(max_score[idx], score)
        max_depth_var[idx] = max(max_depth_var[idx], depth_var)
        max_depth_grad[idx] = max(max_depth_grad[idx], depth_grad)
        max_support[idx] = max(max_support[idx], support)
        key = (gid, str(rec["stem"]))
        if key not in seen:
            view_count[idx] += 1
            seen.add(key)
        if len(examples[gid]) < 5:
            examples[gid].append(
                {
                    "stem": str(rec["stem"]),
                    "layer_conflict_score": score,
                    "depth_variance": depth_var,
                    "depth_gradient": depth_grad,
                    "support_pixel_count": support,
                    "boundary_overlap": float(rec["boundary_overlap"]),
                    "rendered_depth_edge_overlap": float(rec["rendered_depth_edge_overlap"]),
                }
            )
    return max_score, max_depth_var, max_depth_grad, max_support, view_count, examples


def compact_record(i, arrays, tags, thin_examples, layer_examples, support_info):
    gid = int(arrays["gaussian_ids"][i])
    tag_list = [name for name in TAG_NAMES if bool(tags[name][i])]
    return {
        "gaussian_id": gid,
        "tags": tag_list,
        "mean_responsibility": float(arrays["mean_responsibility"][i]),
        "max_responsibility": float(arrays["max_responsibility"][i]),
        "visible_view_count": int(arrays["visible_view_counts"][i]),
        "high_error_view_count": int(arrays["high_error_view_counts"][i]),
        "mean_boundary_overlap": float(arrays["mean_boundary_overlap"][i]),
        "mean_canny_overlap": float(arrays["mean_canny_overlap"][i]),
        "mean_rendered_depth_edge_overlap": float(arrays["mean_rendered_depth_edge_overlap"][i]),
        "mean_thin_structure_overlap": float(arrays["mean_thin_structure_overlap"][i]),
        "mean_screen_radius": float(arrays["mean_screen_radius"][i]),
        "max_screen_radius": float(arrays["max_screen_radius"][i]),
        "mean_support_pixel_count": float(support_info["mean_support_pixel_count"][i]),
        "min_border_distance": float(support_info["min_border_distance"][i]) if np.isfinite(support_info["min_border_distance"][i]) else None,
        "involved_views_sample": support_info["view_counter"].get(gid, [])[:10],
        "thin_examples": thin_examples.get(gid, [])[:3],
        "layer_conflict_examples": layer_examples.get(gid, [])[:3],
    }


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_error_map(path):
    arr = np.load(path).astype(np.float32)
    valid = np.isfinite(arr) & (arr > 0)
    out = np.zeros(arr.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(arr[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        out[valid] = np.clip((arr[valid] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(out, cv2.COLORMAP_TURBO)


def save_selected_overlays(args, arrays, tags, support_info):
    overlay_dir = ensure_dir(os.path.join(args.output_dir, "selected_candidate_overlay"))
    type_to_indices = {}
    priority = np.lexsort((arrays["mean_responsibility"], arrays["high_error_view_counts"]))[::-1]
    for tag in [
        "high_confidence_candidate",
        "thin_structure_responsible",
        "layer_conflict_high",
        "stable_boundary_edge_conflict",
        "border_suspect",
        "low_support_uncertain",
        "split_candidate",
        "shrink_candidate",
        "surface_align_candidate",
    ]:
        selected = [int(i) for i in priority if tags[tag][i]][: args.overlay_per_type]
        type_to_indices[tag] = selected

    selected_ids = set(args.selected_candidate_ids)
    for tag, indices in type_to_indices.items():
        for i in indices:
            selected_ids.add(int(arrays["gaussian_ids"][i]))
    if not selected_ids:
        return overlay_dir

    wanted_views = set(args.selected_views)
    if not wanted_views:
        for gid in selected_ids:
            for stem in support_info["view_counter"].get(int(gid), [])[:2]:
                wanted_views.add(stem)
    wanted_views = sorted(wanted_views)
    selected_by_id = {int(arrays["gaussian_ids"][i]): i for indices in type_to_indices.values() for i in indices}

    for stem in wanted_views:
        npz_path = os.path.join(args.v0_dir, stem, "gaussian_responsibility_v0.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        base_path = os.path.join(args.geometry_error_map_dir, stem, "geometry_error_map.npy")
        if os.path.exists(base_path):
            base = normalize_error_map(base_path)
        else:
            base = np.zeros((1066, 1600, 3), dtype=np.uint8)
        gids = data["gaussian_ids"].astype(np.int64)
        mask = np.isin(gids, list(selected_ids))
        for gid, center, radius in zip(gids[mask], data["screen_centers"][mask], data["radii"][mask]):
            x, y = center
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            r = int(max(2, min(24, float(radius))))
            color = (0, 255, 255) if tags["high_confidence_candidate"][selected_by_id.get(int(gid), 0)] else (0, 0, 255)
            cv2.circle(base, (int(round(x)), int(round(y))), r, color, 1, lineType=cv2.LINE_AA)
            cv2.circle(base, (int(round(x)), int(round(y))), 2, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(base, str(int(gid)), (int(round(x)) + 3, int(round(y)) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        cv2.imwrite(os.path.join(overlay_dir, f"{stem}_selected_candidates_on_error_map.png"), base)
    return overlay_dir


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    if args.skip_existing and os.path.exists(os.path.join(args.output_dir, "structure_candidates_summary.json")):
        print(f"Skip existing: {args.output_dir}")
        return

    arrays = load_v1(args.v1_dir)
    v1_schema = load_v1_schema(args.v1_dir)
    gaussian_ids = arrays["gaussian_ids"].astype(np.int64)
    n = len(gaussian_ids)
    norm_summary = load_json(args.normalized_verification) if os.path.exists(args.normalized_verification) else {}
    normalized_passed = bool(norm_summary.get("decision", {}).get("old_p15_can_be_used_for_candidate_only_tagging", False))

    support_info = aggregate_v0_support(args.v0_dir, gaussian_ids, args)
    thin_score, thin_overlap, thin_error, thin_support, thin_view_count, thin_examples = aggregate_thin(args.v15_dir, gaussian_ids)
    layer_score, layer_depth_var, layer_depth_grad, layer_support, layer_view_count, layer_examples = aggregate_layer(args.v15_dir, gaussian_ids)

    thresholds = {
        "global_responsibility_p": percentile(arrays["mean_responsibility"], args.global_percentile),
        "thin_overlap_p": percentile(np.maximum(arrays["mean_thin_structure_overlap"], thin_overlap), args.thin_percentile),
        "thin_error_p": percentile(thin_error[thin_error > 0], args.thin_percentile),
        "layer_conflict_p": percentile(layer_score[layer_score > 0], args.layer_percentile),
        "boundary_overlap_p": percentile(arrays["mean_boundary_overlap"], args.boundary_edge_percentile),
        "rendered_edge_overlap_p": percentile(arrays["mean_rendered_depth_edge_overlap"], args.boundary_edge_percentile),
        "screen_radius_p": percentile(arrays["mean_screen_radius"], args.radius_percentile),
    }
    combined_thin_overlap = np.maximum(arrays["mean_thin_structure_overlap"], thin_overlap)
    overlap_availability = {
        "boundary": bool(n and np.nanmax(arrays["mean_boundary_overlap"]) > 0),
        "rendered_depth_edge": bool(n and np.nanmax(arrays["mean_rendered_depth_edge_overlap"]) > 0),
        "thin": bool(n and np.nanmax(combined_thin_overlap) > 0),
        "thin_error": bool(n and np.nanmax(thin_error) > 0),
        "layer_conflict": bool(n and np.nanmax(layer_score) > 0),
    }
    boundary_edge_overlap_available = overlap_availability["boundary"] and overlap_availability["rendered_depth_edge"]
    thin_available = overlap_availability["thin"] and overlap_availability["thin_error"]
    layer_available = overlap_availability["layer_conflict"]

    visible = arrays["visible_view_counts"].astype(np.float32)
    high = arrays["high_error_view_counts"].astype(np.float32)
    high_ratio = high / np.maximum(visible, 1.0)
    mean_support = support_info["mean_support_pixel_count"]

    tags = {name: np.zeros(n, dtype=bool) for name in TAG_NAMES}
    tags["multi_view_persistent"] = (high >= args.min_high_error_views) | (
        (visible >= args.min_visible_views) & (high_ratio >= args.high_error_view_ratio)
    )
    tags["global_responsibility_high"] = arrays["mean_responsibility"] >= thresholds["global_responsibility_p"]
    tags["border_suspect"] = (
        (support_info["min_border_distance"] < args.border_margin)
        | (support_info["border_hit_count"] > 0)
        | (support_info["risk_view_hit_count"] > 0)
    )
    severe_border = (support_info["min_border_distance"] < args.severe_border_margin) | (support_info["severe_border_hit_count"] > 0)
    tags["low_support_uncertain"] = (
        (mean_support < args.support_pixel_threshold)
        | ((layer_score > 0) & (layer_support < args.layer_support_threshold))
        | ((thin_score > 0) & (thin_support < args.thin_support_threshold))
    )
    tags["thin_structure_responsible"] = (
        thin_available
        & (combined_thin_overlap >= thresholds["thin_overlap_p"])
        & (thin_error >= thresholds["thin_error_p"])
        & (thin_support >= args.thin_support_threshold)
        & (~severe_border)
    )
    tags["layer_conflict_high"] = layer_available & (layer_score >= thresholds["layer_conflict_p"])
    tags["stable_boundary_edge_conflict"] = (
        boundary_edge_overlap_available
        & (arrays["mean_boundary_overlap"] >= thresholds["boundary_overlap_p"])
        & (arrays["mean_rendered_depth_edge_overlap"] >= thresholds["rendered_edge_overlap_p"])
        & tags["multi_view_persistent"]
        & (mean_support >= args.support_pixel_threshold)
        & (~severe_border)
    )
    explanatory = tags["thin_structure_responsible"] | tags["layer_conflict_high"] | tags["stable_boundary_edge_conflict"]
    tags["high_confidence_candidate"] = (
        tags["multi_view_persistent"]
        & tags["global_responsibility_high"]
        & (~tags["low_support_uncertain"])
        & (~severe_border)
        & explanatory
    )
    large_radius = arrays["mean_screen_radius"] >= thresholds["screen_radius_p"]
    tags["split_candidate"] = tags["high_confidence_candidate"] & tags["layer_conflict_high"] & large_radius
    tags["shrink_candidate"] = (
        tags["high_confidence_candidate"]
        & large_radius
        & (
            tags["stable_boundary_edge_conflict"]
            | (
                boundary_edge_overlap_available
                & (arrays["mean_boundary_overlap"] >= thresholds["boundary_overlap_p"])
                & (arrays["mean_rendered_depth_edge_overlap"] >= thresholds["rendered_edge_overlap_p"])
            )
        )
    )
    tags["surface_align_candidate"] = (
        tags["high_confidence_candidate"]
        & tags["thin_structure_responsible"]
        & (thin_support >= args.thin_support_threshold)
    )
    tags["opacity_decay_candidate"] = (
        tags["global_responsibility_high"]
        & (~tags["high_confidence_candidate"])
        & (tags["low_support_uncertain"] | tags["border_suspect"] | (~tags["multi_view_persistent"]) | (~explanatory))
    )

    any_candidate = np.zeros(n, dtype=bool)
    for name in TAG_NAMES:
        any_candidate |= tags[name]
    candidate_indices = np.flatnonzero(any_candidate)

    tag_matrix = np.vstack([tags[name] for name in TAG_NAMES]).T.astype(np.uint8)
    np.savez_compressed(
        os.path.join(args.output_dir, "structure_candidates_v0.npz"),
        gaussian_ids=gaussian_ids,
        uses_stable_gaussian_ids=np.asarray([bool(v1_schema.get("uses_stable_gaussian_ids"))], dtype=np.bool_)
        if v1_schema.get("uses_stable_gaussian_ids") is not None
        else np.asarray([False], dtype=np.bool_),
        candidate_indices=candidate_indices.astype(np.int64),
        tag_names=np.asarray(TAG_NAMES),
        tag_matrix=tag_matrix,
        mean_responsibility=arrays["mean_responsibility"],
        max_responsibility=arrays["max_responsibility"],
        visible_view_counts=arrays["visible_view_counts"],
        high_error_view_counts=arrays["high_error_view_counts"],
        mean_boundary_overlap=arrays["mean_boundary_overlap"],
        mean_canny_overlap=arrays["mean_canny_overlap"],
        mean_rendered_depth_edge_overlap=arrays["mean_rendered_depth_edge_overlap"],
        mean_thin_structure_overlap=arrays["mean_thin_structure_overlap"],
        mean_screen_radius=arrays["mean_screen_radius"],
        max_screen_radius=arrays["max_screen_radius"],
        mean_support_pixel_count=mean_support,
        min_border_distance=support_info["min_border_distance"],
        thin_score=thin_score,
        thin_overlap=thin_overlap,
        thin_mean_error=thin_error,
        thin_support_pixels=thin_support,
        thin_view_count=thin_view_count,
        layer_conflict_score=layer_score,
        layer_depth_variance=layer_depth_var,
        layer_depth_gradient=layer_depth_grad,
        layer_support_pixels=layer_support,
        layer_view_count=layer_view_count,
    )

    selected = {}
    order = np.lexsort((arrays["mean_responsibility"], arrays["high_error_view_counts"]))[::-1]
    for name in TAG_NAMES:
        idxs = [int(i) for i in order if tags[name][i]][: args.max_json_records]
        selected[name] = [compact_record(i, arrays, tags, thin_examples, layer_examples, support_info) for i in idxs]

    high_conf = selected["high_confidence_candidate"]
    uncertain_mask = tags["low_support_uncertain"] | tags["border_suspect"]
    uncertain_idxs = [int(i) for i in order if uncertain_mask[i]][: args.max_json_records]
    uncertain = [compact_record(i, arrays, tags, thin_examples, layer_examples, support_info) for i in uncertain_idxs]

    save_json(os.path.join(args.output_dir, "high_confidence_candidates.json"), high_conf)
    save_json(os.path.join(args.output_dir, "uncertain_candidates.json"), uncertain)

    by_type = {name: {"count": int(np.count_nonzero(tags[name])), "examples": selected[name][:10]} for name in TAG_NAMES}
    save_json(os.path.join(args.output_dir, "candidate_by_type_summary.json"), by_type)

    top_by_view = {}
    for stem, gids in support_info["top_rows_by_view"].items():
        records = []
        for gid in gids[: args.top_k]:
            idx_arr = np.where(gaussian_ids == gid)[0]
            if idx_arr.size:
                records.append(compact_record(int(idx_arr[0]), arrays, tags, thin_examples, layer_examples, support_info))
        top_by_view[stem] = records[:20]
    save_json(os.path.join(args.output_dir, "candidate_topK_by_view.json"), top_by_view)

    camera_distribution = {}
    for idx in candidate_indices:
        gid = int(gaussian_ids[idx])
        for cam, count in support_info["camera_counter"].get(gid, {}).items():
            camera_distribution[str(cam)] = camera_distribution.get(str(cam), 0) + int(count)

    top_views = Counter()
    for idx in candidate_indices:
        gid = int(gaussian_ids[idx])
        for stem in support_info["view_counter"].get(gid, []):
            top_views[stem] += 1

    overlay_dir = None
    if args.save_overlays:
        overlay_dir = save_selected_overlays(args, arrays, tags, support_info)
    else:
        ensure_dir(os.path.join(args.output_dir, "selected_candidate_overlay"))
        overlay_dir = os.path.join(args.output_dir, "selected_candidate_overlay")

    summary = {
        "config": {
            "v1_dir": os.path.abspath(args.v1_dir),
            "v15_dir": os.path.abspath(args.v15_dir),
            "v0_dir": os.path.abspath(args.v0_dir),
            "geometry_error_map_dir": os.path.abspath(args.geometry_error_map_dir),
            "normalized_verification": os.path.abspath(args.normalized_verification),
            "output_dir": os.path.abspath(args.output_dir),
            "thresholds": thresholds,
            "overlap_availability": overlap_availability,
            "overlap_guard_note": "Overlap-dependent tags are disabled when the corresponding mask/overlap family is all zero or unavailable.",
            "gaussian_id_schema": v1_schema,
            "parameters": vars(args),
        },
        "id_schema_warning": v1_schema.get("warning"),
        "normalized_verification_passed": normalized_passed,
        "total_gaussian_count": int(n),
        "total_candidate_count": int(len(candidate_indices)),
        "tag_counts": {name: int(np.count_nonzero(tags[name])) for name in TAG_NAMES},
        "border_suspect_ratio": float(np.mean(tags["border_suspect"])),
        "low_support_uncertain_ratio": float(np.mean(tags["low_support_uncertain"])),
        "camera_wise_candidate_distribution": camera_distribution,
        "top_candidate_views": [{"stem": stem, "count": int(count)} for stem, count in top_views.most_common(20)],
        "selected_candidate_ids": [int(gaussian_ids[i]) for i in order if any_candidate[i]][:50],
        "selected_overlay_dir": overlay_dir,
        "decision_note": "candidate-only labels only; no Gaussian parameters were modified.",
        "next_step_recommendation": (
            "Do not enter split/shrink/prune yet; first review high_confidence candidates and border/low-support guards."
            if int(np.count_nonzero(tags["high_confidence_candidate"])) == 0
            else "Candidate-only tagging is ready for manual review. Low-risk structure operations should still be a separate explicit experiment."
        ),
    }
    save_json(os.path.join(args.output_dir, "structure_candidates_summary.json"), summary)
    print(f"Saved candidate tags for {n} Gaussians.")
    print(f"Candidates with any tag: {len(candidate_indices)}")
    print(f"High-confidence candidates: {summary['tag_counts']['high_confidence_candidate']}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
