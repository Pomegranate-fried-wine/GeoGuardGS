import argparse
import json
import os

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate single-view Gaussian responsibility into multi-view v1.")
    parser.add_argument("--v0-dir", required=True)
    parser.add_argument("--output-dir", default="output/local_smoke/gaussian_responsibility_v1")
    parser.add_argument("--geometry-error-map-dir", default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--high-view-quantile", type=float, default=0.90)
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_views(v0_dir, high_view_quantile):
    view_data = {}
    schema_versions = []
    arrays = {
        "gaussian_ids": [],
        "scores": [],
        "high_flags": [],
        "boundary": [],
        "canny": [],
        "rendered_edge": [],
        "thin": [],
        "radii": [],
        "foreground": [],
        "object_ids": [],
        "track_ids": [],
    }
    for stem in sorted(os.listdir(v0_dir)):
        view_dir = os.path.join(v0_dir, stem)
        npz_path = os.path.join(view_dir, "gaussian_responsibility_v0.npz")
        if not os.path.isdir(view_dir) or not os.path.exists(npz_path):
            continue
        data = np.load(npz_path, allow_pickle=True)
        if "stable_id_schema_version" in data:
            schema_versions.append(int(np.asarray(data["stable_id_schema_version"]).reshape(-1)[0]))
        else:
            schema_versions.append(0)
        scores = data["responsibility_scores"].astype(np.float32)
        high_threshold = float(np.quantile(scores, high_view_quantile)) if scores.size else 1.0
        high_flags = scores >= high_threshold
        canny = data["canny_overlaps"].astype(np.float32) if "canny_overlaps" in data else np.zeros_like(scores)
        rendered_edge = (
            data["rendered_depth_edge_overlaps"].astype(np.float32)
            if "rendered_depth_edge_overlaps" in data
            else np.zeros_like(scores)
        )
        foreground = data["foreground"].astype(np.int32) if "foreground" in data else np.zeros_like(scores, dtype=np.int32)
        object_ids = data["object_ids"].astype(np.int32) if "object_ids" in data else np.full_like(scores, -1, dtype=np.int32)
        track_ids = data["track_ids"].astype(np.int32) if "track_ids" in data else np.full_like(scores, -1, dtype=np.int32)

        gids = data["gaussian_ids"].astype(np.int64)
        radii = data["radii"].astype(np.float32)
        view_data[stem] = {
            "gaussian_ids": gids,
            "scores": scores,
            "screen_centers": data["screen_centers"].astype(np.float32),
            "radii": radii,
        }
        arrays["gaussian_ids"].append(gids)
        arrays["scores"].append(scores)
        arrays["high_flags"].append(high_flags.astype(np.float32))
        arrays["boundary"].append(data["boundary_overlaps"].astype(np.float32))
        arrays["canny"].append(canny)
        arrays["rendered_edge"].append(rendered_edge)
        arrays["thin"].append(data["thin_structure_overlaps"].astype(np.float32))
        arrays["radii"].append(radii)
        arrays["foreground"].append(foreground)
        arrays["object_ids"].append(object_ids)
        arrays["track_ids"].append(track_ids)

    if not arrays["gaussian_ids"]:
        raise RuntimeError("No V0 responsibility npz files found.")
    flat = {key: np.concatenate(value) for key, value in arrays.items()}
    schema_unique = sorted(set(schema_versions))
    schema = {
        "stable_id_schema_versions": schema_unique,
        "uses_stable_gaussian_ids": bool(schema_unique and min(schema_unique) >= 1),
        "legacy_view_local_id_inputs": int(sum(1 for version in schema_versions if version <= 0)),
        "view_count": int(len(schema_versions)),
    }
    return view_data, flat, schema


def aggregate_arrays(flat):
    gids = flat["gaussian_ids"]
    unique_gids, inverse = np.unique(gids, return_inverse=True)
    count = np.bincount(inverse).astype(np.float32)

    def mean(name):
        return np.bincount(inverse, weights=flat[name]) / np.maximum(count, 1.0)

    def max_by(name):
        out = np.full(unique_gids.shape, -np.inf, dtype=np.float32)
        np.maximum.at(out, inverse, flat[name])
        return out

    score_sum = np.bincount(inverse, weights=flat["scores"])
    score_sq_sum = np.bincount(inverse, weights=flat["scores"] * flat["scores"])
    mean_resp = score_sum / np.maximum(count, 1.0)
    variance = score_sq_sum / np.maximum(count, 1.0) - mean_resp * mean_resp

    first_row = np.full(unique_gids.shape, -1, dtype=np.int64)
    np.minimum.at(first_row, inverse, np.arange(len(gids), dtype=np.int64))
    return {
        "gaussian_ids": unique_gids,
        "visible_view_counts": count.astype(np.int32),
        "high_error_view_counts": np.bincount(inverse, weights=flat["high_flags"]).astype(np.int32),
        "mean_responsibility": mean_resp.astype(np.float32),
        "max_responsibility": max_by("scores"),
        "responsibility_variance": np.maximum(variance, 0.0).astype(np.float32),
        "mean_boundary_overlap": mean("boundary").astype(np.float32),
        "mean_canny_overlap": mean("canny").astype(np.float32),
        "mean_rendered_depth_edge_overlap": mean("rendered_edge").astype(np.float32),
        "mean_thin_structure_overlap": mean("thin").astype(np.float32),
        "mean_screen_radius": mean("radii").astype(np.float32),
        "max_screen_radius": max_by("radii"),
        "foreground": flat["foreground"][first_row].astype(np.int32),
        "object_ids": flat["object_ids"][first_row].astype(np.int32),
        "track_ids": flat["track_ids"][first_row].astype(np.int32),
    }


def group_summary(name, indices, agg):
    if len(indices) == 0:
        return {"name": name, "count": 0}
    return {
        "name": name,
        "count": int(len(indices)),
        "mean_responsibility": float(np.mean(agg["mean_responsibility"][indices])),
        "max_responsibility": float(np.mean(agg["max_responsibility"][indices])),
        "visible_view_count": float(np.mean(agg["visible_view_counts"][indices])),
        "high_error_view_count": float(np.mean(agg["high_error_view_counts"][indices])),
        "boundary_overlap": float(np.mean(agg["mean_boundary_overlap"][indices])),
        "canny_overlap": float(np.mean(agg["mean_canny_overlap"][indices])),
        "rendered_depth_edge_overlap": float(np.mean(agg["mean_rendered_depth_edge_overlap"][indices])),
        "thin_structure_overlap": float(np.mean(agg["mean_thin_structure_overlap"][indices])),
        "radius": float(np.mean(agg["mean_screen_radius"][indices])),
    }


def top_involved_views(gaussian_id, view_data):
    views = []
    for stem, data in view_data.items():
        if np.any(data["gaussian_ids"] == gaussian_id):
            views.append(stem)
    return views


def save_histogram(path, values):
    canvas = np.full((460, 900, 3), 255, dtype=np.uint8)
    hist, _ = np.histogram(values, bins=50, range=(0.0, max(float(np.max(values)), 1e-6)))
    left, right, top, bottom = 60, 860, 40, 390
    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 0, 0), 1)
    max_hist = max(int(np.max(hist)), 1)
    bin_w = (right - left) / len(hist)
    for idx, count in enumerate(hist):
        x0 = int(left + idx * bin_w)
        x1 = int(left + (idx + 1) * bin_w)
        y = int(bottom - (bottom - top) * count / max_hist)
        cv2.rectangle(canvas, (x0, y), (x1 - 1, bottom), (160, 160, 160), -1)
    cv2.putText(canvas, "Global responsibility histogram", (60, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(path, canvas)


def draw_global_overlays(args, view_data, top_ids):
    overlay_dir = ensure_dir(os.path.join(args.output_dir, "topK_global_overlay_by_view"))
    top_id_set = set(int(x) for x in top_ids)
    for stem, data in view_data.items():
        base = None
        if args.geometry_error_map_dir:
            base = cv2.imread(os.path.join(args.geometry_error_map_dir, stem, "geometry_error_map.png"), cv2.IMREAD_COLOR)
        if base is None:
            base = np.zeros((1066, 1600, 3), dtype=np.uint8)
        mask = np.isin(data["gaussian_ids"], list(top_id_set))
        for center, radius in zip(data["screen_centers"][mask], data["radii"][mask]):
            x, y = center
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            r = int(max(2, min(24, radius)))
            cv2.circle(base, (int(round(x)), int(round(y))), r, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.circle(base, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.imwrite(os.path.join(overlay_dir, f"{stem}_global_topK_overlay.png"), base)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    view_data, flat, id_schema = load_views(args.v0_dir, args.high_view_quantile)
    agg = aggregate_arrays(flat)

    order = np.lexsort(
        (
            agg["max_responsibility"],
            agg["mean_responsibility"],
            agg["visible_view_counts"],
            agg["high_error_view_counts"],
        )
    )[::-1]
    k = min(args.top_k, len(order))
    top_idx = order[:k]
    rng = np.random.default_rng(args.random_seed)
    random_idx = rng.choice(len(order), size=k, replace=False) if len(order) >= k else np.arange(len(order))
    large_idx = np.argsort(agg["mean_screen_radius"])[::-1][:k]
    single_candidates = np.where(agg["visible_view_counts"] == 1)[0]
    high_single_idx = (
        single_candidates[np.argsort(agg["max_responsibility"][single_candidates])[::-1][:k]]
        if single_candidates.size
        else np.array([], dtype=np.int64)
    )

    np.savez_compressed(
        os.path.join(args.output_dir, "gaussian_responsibility_global.npz"),
        **agg,
        stable_id_schema_versions=np.asarray(id_schema["stable_id_schema_versions"], dtype=np.int32),
        uses_stable_gaussian_ids=np.asarray([id_schema["uses_stable_gaussian_ids"]], dtype=np.bool_),
        legacy_view_local_id_inputs=np.asarray([id_schema["legacy_view_local_id_inputs"]], dtype=np.int32),
        is_dynamic=(agg["track_ids"] >= 0),
        topk_global_indices=top_idx,
        random_indices=random_idx,
        large_radius_indices=large_idx,
        high_single_view_indices=high_single_idx,
    )

    top_ids = agg["gaussian_ids"][top_idx]
    with open(os.path.join(args.output_dir, "topK_global_responsibility.txt"), "w", encoding="utf-8") as f:
        for rank, idx in enumerate(top_idx, start=1):
            gid = int(agg["gaussian_ids"][idx])
            views = top_involved_views(gid, view_data)
            f.write(
                f"{rank:04d} gaussian_id={gid} mean={agg['mean_responsibility'][idx]:.6f} "
                f"max={agg['max_responsibility'][idx]:.6f} visible={int(agg['visible_view_counts'][idx])} "
                f"high_views={int(agg['high_error_view_counts'][idx])} "
                f"boundary={agg['mean_boundary_overlap'][idx]:.4f} "
                f"canny={agg['mean_canny_overlap'][idx]:.4f} "
                f"edge={agg['mean_rendered_depth_edge_overlap'][idx]:.4f} "
                f"thin={agg['mean_thin_structure_overlap'][idx]:.4f} views={','.join(views)}\n"
            )

    baseline = {
        "topK_global_responsibility": group_summary("topK_global_responsibility", top_idx, agg),
        "random_visible_gaussians": group_summary("random_visible_gaussians", random_idx, agg),
        "large_radius_gaussians": group_summary("large_radius_gaussians", large_idx, agg),
        "high_single_view_only_gaussians": group_summary("high_single_view_only_gaussians", high_single_idx, agg),
    }
    view_consistency = {
        "views": sorted(view_data.keys()),
        "view_count": len(view_data),
        "global_gaussian_count": int(len(agg["gaussian_ids"])),
        "multi_view_gaussian_count": int(np.count_nonzero(agg["visible_view_counts"] > 1)),
        "persistent_high_error_count": int(np.count_nonzero(agg["high_error_view_counts"] > 1)),
        "max_visible_view_count": int(np.max(agg["visible_view_counts"])),
        "max_high_error_view_count": int(np.max(agg["high_error_view_counts"])),
    }
    top_preview = []
    for idx in top_idx[:20]:
        gid = int(agg["gaussian_ids"][idx])
        top_preview.append(
            {
                "gaussian_id": gid,
                "visible_view_count": int(agg["visible_view_counts"][idx]),
                "high_error_view_count": int(agg["high_error_view_counts"][idx]),
                "mean_responsibility": float(agg["mean_responsibility"][idx]),
                "max_responsibility": float(agg["max_responsibility"][idx]),
                "mean_boundary_overlap": float(agg["mean_boundary_overlap"][idx]),
                "mean_canny_overlap": float(agg["mean_canny_overlap"][idx]),
                "mean_rendered_depth_edge_overlap": float(agg["mean_rendered_depth_edge_overlap"][idx]),
                "mean_thin_structure_overlap": float(agg["mean_thin_structure_overlap"][idx]),
                "mean_screen_radius": float(agg["mean_screen_radius"][idx]),
                "involved_views": top_involved_views(gid, view_data),
            }
        )

    summary = {
        "config": {
            "v0_dir": os.path.abspath(args.v0_dir),
            "output_dir": os.path.abspath(args.output_dir),
            "geometry_error_map_dir": os.path.abspath(args.geometry_error_map_dir)
            if args.geometry_error_map_dir
            else None,
            "top_k": args.top_k,
            "high_view_quantile": args.high_view_quantile,
        },
        "gaussian_id_schema": id_schema,
        "id_schema_warning": None
        if id_schema["uses_stable_gaussian_ids"]
        else "Input V0 files use legacy view-local gaussian_ids; multi-view aggregation can merge unrelated renderer rows. Regenerate V0 with stable_id_schema_version>=1 before using candidates.",
        "baseline_comparison": baseline,
        "view_consistency": view_consistency,
        "top_global_gaussians": top_preview,
    }
    with open(os.path.join(args.output_dir, "global_responsibility_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "view_consistency_summary.json"), "w", encoding="utf-8") as f:
        json.dump(view_consistency, f, indent=2, ensure_ascii=False)

    save_histogram(os.path.join(args.output_dir, "responsibility_global_histogram.png"), agg["mean_responsibility"])
    draw_global_overlays(args, view_data, top_ids)
    print(f"Aggregated {len(agg['gaussian_ids'])} Gaussians from {len(view_data)} views.")
    print(f"Saved: {os.path.join(args.output_dir, 'global_responsibility_summary.json')}")


if __name__ == "__main__":
    main()
