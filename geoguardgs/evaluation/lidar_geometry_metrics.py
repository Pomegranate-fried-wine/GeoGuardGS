import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np


METRIC_KEYS = ["AbsRel", "RMSE", "MAE", "delta_lt_1_25"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate rendered depth against sparse LiDAR with geometry credibility regions."
    )
    parser.add_argument("--pred-dir", default="output/Waymo/002_baseline/train/ours_7000")
    parser.add_argument("--lidar-dir", default="data/waymo/002/lidar_depth")
    parser.add_argument("--image-dir", default="data/waymo/002/images")
    parser.add_argument("--object-boundary-dir", default=None)
    parser.add_argument("--prior-depth-dir", default=None)
    parser.add_argument("--acc-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--frames", nargs="*", type=int, default=None)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=20)
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--cam-ids", nargs="+", type=int, default=None)
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--min-acc", type=float, default=0.05)
    parser.add_argument("--boundary-band-radius", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--depth-edge-percentile", type=float, default=90.0)
    parser.add_argument("--thin-structure-percentile", type=float, default=92.0)
    parser.add_argument("--save-region-visuals", action="store_true")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--num-crops", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Reserved for checkpoint-driven rendering. Use render.py first, then pass --pred-dir for this evaluator.",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def to_float_map(array):
    array = np.asarray(array)
    if array.ndim == 3:
        array = np.squeeze(array)
    return array.astype(np.float32)


def resize_like(src, shape, interpolation):
    if src is None:
        return None
    if src.shape[:2] == shape:
        return src
    return cv2.resize(src, (shape[1], shape[0]), interpolation=interpolation)


def load_lidar(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == () and isinstance(data.item(), dict):
        data = data.item()
    if isinstance(data, dict) and "mask" in data and "value" in data:
        mask = np.asarray(data["mask"]).astype(bool)
        depth = np.zeros(mask.shape, dtype=np.float32)
        depth[mask] = np.asarray(data["value"], dtype=np.float32).reshape(-1)
        return depth, mask
    depth = to_float_map(data)
    return depth, np.isfinite(depth) & (depth > 0)


def load_optional_map(directory, stem, suffixes):
    if not directory:
        return None, None
    for suffix in suffixes:
        path = os.path.join(directory, f"{stem}{suffix}")
        if os.path.exists(path):
            if suffix.endswith(".npy"):
                return to_float_map(np.load(path, allow_pickle=True)), path
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image is not None:
                return to_float_map(image), path
    return None, None


def load_rgb(image_dir, stem):
    if not image_dir:
        return None, None
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(image_dir, f"{stem}{ext}")
        if os.path.exists(path):
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is not None:
                return image, path
    return None, None


def find_rendered_depth(pred_dir, stem):
    candidates = [
        f"{stem}_depth.npy",
        f"{stem}_rendered_depth.npy",
        f"{stem}.npy",
        os.path.join(stem, "rendered_depth.npy"),
        os.path.join(stem, "depth.npy"),
    ]
    for candidate in candidates:
        path = os.path.join(pred_dir, candidate)
        if os.path.exists(path):
            return path
    return None


def dilate_mask(mask, radius):
    mask = np.asarray(mask).astype(bool)
    if radius <= 0:
        return mask
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def normalize_u8(values, valid=None):
    x = np.asarray(values, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(x)
    valid = valid & np.isfinite(x)
    out = np.zeros(x.shape, dtype=np.uint8)
    if not np.any(valid):
        return out
    lo, hi = np.percentile(x[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    out[valid] = np.clip((x[valid] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


def colorize(values, valid=None, cmap=cv2.COLORMAP_TURBO):
    return cv2.applyColorMap(normalize_u8(values, valid), cmap)


def edge_from_depth(depth, valid, percentile):
    valid = valid & np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros(depth.shape, dtype=bool)
    filled = depth.copy()
    median = float(np.median(filled[valid]))
    filled[~valid] = median
    gx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    threshold = np.percentile(grad[valid], percentile)
    return (grad >= threshold) & valid


def confidence_flag(count):
    if count >= 50:
        return "normal"
    if count > 0:
        return "low_confidence"
    return "empty"


def compute_metrics(rendered, lidar, base_valid, region_mask):
    region_mask = np.asarray(region_mask).astype(bool)
    valid = base_valid & region_mask
    count = int(np.count_nonzero(valid))
    result = {
        "pixel_count": int(np.count_nonzero(region_mask)),
        "valid_lidar_count": count,
        "confidence": confidence_flag(count),
    }
    if count == 0:
        for key in METRIC_KEYS:
            result[key] = None
        return result

    pred = rendered[valid].astype(np.float64)
    gt = lidar[valid].astype(np.float64)
    err = np.abs(pred - gt)
    ratio = np.maximum(pred / gt, gt / pred)
    result.update(
        {
            "AbsRel": float(np.mean(err / gt)),
            "RMSE": float(np.sqrt(np.mean((pred - gt) ** 2))),
            "MAE": float(np.mean(err)),
            "delta_lt_1_25": float(np.mean(ratio < 1.25)),
        }
    )
    return result


def unavailable_region(reason):
    result = {
        "available": False,
        "reason": reason,
        "pixel_count": 0,
        "valid_lidar_count": 0,
        "confidence": "empty",
    }
    for key in METRIC_KEYS:
        result[key] = None
    return result


def save_mask(path, mask):
    cv2.imwrite(path, (np.asarray(mask).astype(np.uint8) * 255))


def save_region_error_map(path, abs_error, base_valid, region_mask):
    region_valid = base_valid & region_mask
    visual = colorize(abs_error, region_valid)
    visual[~region_valid] = 0
    cv2.imwrite(path, visual)


def crop_bounds(center_y, center_x, height, width, crop_size):
    half = crop_size // 2
    y0 = max(0, min(height - crop_size, center_y - half))
    x0 = max(0, min(width - crop_size, center_x - half))
    y1 = min(height, y0 + crop_size)
    x1 = min(width, x0 + crop_size)
    return y0, y1, x0, x1


def save_high_error_crops(frame_dir, stem, rgb, rendered, lidar, abs_error, base_valid, edge_mask, args):
    crop_dir = ensure_dir(os.path.join(frame_dir, "crops"))
    score = abs_error.copy()
    score[~base_valid] = 0
    if not np.any(score > 0):
        return []

    height, width = score.shape
    crop_size = min(args.crop_size, height, width)
    selected = []
    suppressed = np.zeros_like(base_valid, dtype=bool)

    for idx in range(args.num_crops):
        candidate = score.copy()
        candidate[suppressed] = 0
        flat = int(np.argmax(candidate))
        if candidate.flat[flat] <= 0:
            break
        y, x = np.unravel_index(flat, candidate.shape)
        y0, y1, x0, x1 = crop_bounds(int(y), int(x), height, width, crop_size)
        suppressed[y0:y1, x0:x1] = True

        panels = []
        if rgb is not None:
            panels.append(rgb[y0:y1, x0:x1])
        panels.append(colorize(rendered[y0:y1, x0:x1], np.isfinite(rendered[y0:y1, x0:x1])))
        panels.append(colorize(lidar[y0:y1, x0:x1], lidar[y0:y1, x0:x1] > 0))
        panels.append(colorize(abs_error[y0:y1, x0:x1], base_valid[y0:y1, x0:x1]))
        panels.append(cv2.cvtColor((edge_mask[y0:y1, x0:x1].astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR))
        sheet = np.concatenate(panels, axis=1)
        out_path = os.path.join(crop_dir, f"{stem}_crop_{idx:02d}.png")
        cv2.imwrite(out_path, sheet)
        selected.append({"center": [int(y), int(x)], "path": out_path})
    return selected


def aggregate_region_metrics(region_values):
    rendered = np.concatenate([x[0] for x in region_values]) if region_values else np.array([])
    lidar = np.concatenate([x[1] for x in region_values]) if region_values else np.array([])
    pixel_count = int(sum(x[2] for x in region_values))
    count = int(lidar.size)
    result = {
        "available": bool(region_values),
        "pixel_count": pixel_count,
        "valid_lidar_count": count,
        "confidence": confidence_flag(count),
    }
    if count == 0:
        for key in METRIC_KEYS:
            result[key] = None
        return result
    err = np.abs(rendered - lidar)
    ratio = np.maximum(rendered / lidar, lidar / rendered)
    result.update(
        {
            "AbsRel": float(np.mean(err / lidar)),
            "RMSE": float(np.sqrt(np.mean((rendered - lidar) ** 2))),
            "MAE": float(np.mean(err)),
            "delta_lt_1_25": float(np.mean(ratio < 1.25)),
        }
    )
    return result


def add_region(region_accumulator, name, rendered, lidar, base_valid, mask):
    valid = base_valid & mask
    region_accumulator[name].append(
        (rendered[valid].astype(np.float64), lidar[valid].astype(np.float64), int(np.count_nonzero(mask)))
    )


def evaluate_frame(args, stem, frame_dir, region_accumulator):
    pred_path = find_rendered_depth(args.pred_dir, stem)
    lidar_path = os.path.join(args.lidar_dir, f"{stem}.npy")
    if pred_path is None or not os.path.exists(lidar_path):
        return None

    rendered = to_float_map(np.load(pred_path, allow_pickle=True))
    lidar, lidar_mask = load_lidar(lidar_path)
    if rendered.shape != lidar.shape:
        rendered = resize_like(rendered, lidar.shape, cv2.INTER_LINEAR)

    rgb, rgb_path = load_rgb(args.image_dir, stem)
    if rgb is not None:
        rgb = resize_like(rgb, lidar.shape, cv2.INTER_LINEAR)

    acc_dir = args.acc_dir or args.pred_dir
    acc, acc_path = load_optional_map(acc_dir, stem, ["_acc.npy", "_acc.png", ".npy", ".png"])
    if acc is not None:
        acc = resize_like(acc, lidar.shape, cv2.INTER_LINEAR)
        if np.nanmax(acc) > 1.5:
            acc = acc / 255.0

    object_boundary, object_boundary_path = load_optional_map(
        args.object_boundary_dir, stem, ["_obj_bound.npy", "_obj_bound.png", ".npy", ".png"]
    )
    if object_boundary is not None:
        object_boundary = resize_like(object_boundary, lidar.shape, cv2.INTER_NEAREST) > 0

    prior_depth, prior_depth_path = load_optional_map(
        args.prior_depth_dir, stem, ["_depth.npy", "_prior_depth.npy", ".npy", ".png"]
    )
    if prior_depth is not None:
        prior_depth = resize_like(prior_depth, lidar.shape, cv2.INTER_LINEAR)

    base_valid = (
        lidar_mask
        & np.isfinite(lidar)
        & np.isfinite(rendered)
        & (lidar > args.min_depth)
        & (lidar < args.max_depth)
        & (rendered > args.min_depth)
        & (rendered < args.max_depth)
    )
    acc_valid = base_valid & (acc >= args.min_acc) if acc is not None else None
    abs_error = np.abs(rendered - lidar)

    if rgb is not None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        canny_edge = cv2.Canny(gray, args.canny_low, args.canny_high) > 0
    else:
        canny_edge = None

    rendered_depth_edge = edge_from_depth(rendered, np.isfinite(rendered), args.depth_edge_percentile)
    prior_depth_edge = (
        edge_from_depth(prior_depth, np.isfinite(prior_depth), args.depth_edge_percentile)
        if prior_depth is not None
        else None
    )
    thin_structure = rendered_depth_edge
    if canny_edge is not None:
        thin_structure = thin_structure & dilate_mask(canny_edge, 1)

    any_boundary = np.zeros(lidar.shape, dtype=bool)
    if object_boundary is not None:
        any_boundary |= object_boundary
    if canny_edge is not None:
        any_boundary |= canny_edge
    any_boundary |= rendered_depth_edge
    boundary_band = dilate_mask(any_boundary, max(args.boundary_band_radius))
    stable_non_boundary = base_valid & (~boundary_band)

    frame_regions = {
        "all_valid": base_valid,
        "boundary_band": boundary_band,
        "rendered_depth_edge_band_mask": rendered_depth_edge,
        "thin_structure_band": thin_structure,
        "stable_non_boundary": stable_non_boundary,
    }
    if acc_valid is not None:
        frame_regions["acc_depth_filtered"] = acc_valid

    for radius in args.boundary_band_radius:
        if canny_edge is not None:
            frame_regions[f"canny_band_r{radius}"] = dilate_mask(canny_edge, radius)
        if object_boundary is not None:
            frame_regions[f"object_boundary_band_r{radius}"] = dilate_mask(object_boundary, radius)
        if prior_depth_edge is not None:
            frame_regions[f"prior_depth_edge_band_r{radius}"] = dilate_mask(prior_depth_edge, radius)
        frame_regions[f"rendered_depth_edge_band_r{radius}"] = dilate_mask(rendered_depth_edge, radius)

    for name, mask in frame_regions.items():
        add_region(region_accumulator, name, rendered, lidar, base_valid, mask)

    ensure_dir(frame_dir)
    cv2.imwrite(os.path.join(frame_dir, "rendered_depth.png"), colorize(rendered, np.isfinite(rendered)))
    cv2.imwrite(os.path.join(frame_dir, "lidar_depth.png"), colorize(lidar, lidar > 0))
    cv2.imwrite(os.path.join(frame_dir, "global_error_map.png"), colorize(abs_error, base_valid))

    if args.save_region_visuals:
        region_dir = ensure_dir(os.path.join(frame_dir, "region_error_maps"))
        if canny_edge is not None:
            save_mask(os.path.join(frame_dir, "canny_band_mask.png"), dilate_mask(canny_edge, max(args.boundary_band_radius)))
        if object_boundary is not None:
            save_mask(os.path.join(frame_dir, "object_boundary_band_mask.png"), dilate_mask(object_boundary, max(args.boundary_band_radius)))
        save_mask(os.path.join(frame_dir, "boundary_band_mask.png"), boundary_band)
        save_mask(
            os.path.join(frame_dir, "rendered_depth_edge_band_mask.png"),
            dilate_mask(rendered_depth_edge, max(args.boundary_band_radius)),
        )
        save_mask(os.path.join(frame_dir, "thin_structure_mask.png"), thin_structure)
        for name, mask in frame_regions.items():
            save_region_error_map(os.path.join(region_dir, f"{name}_error.png"), abs_error, base_valid, mask)

    crops = []
    if args.save_crops:
        crops = save_high_error_crops(
            frame_dir,
            stem,
            rgb,
            rendered,
            lidar,
            abs_error,
            base_valid,
            boundary_band,
            args,
        )

    frame_metrics = {
        name: dict({"available": True}, **compute_metrics(rendered, lidar, base_valid, mask))
        for name, mask in frame_regions.items()
    }
    if acc_valid is None:
        frame_metrics["acc_depth_filtered"] = unavailable_region("render_acc/alpha map was not provided")
    for radius in args.boundary_band_radius:
        if canny_edge is None:
            frame_metrics[f"canny_band_r{radius}"] = unavailable_region("RGB image was not found")
        if object_boundary is None:
            frame_metrics[f"object_boundary_band_r{radius}"] = unavailable_region(
                "object boundary or boundary mask was not provided"
            )
        if prior_depth is None:
            frame_metrics[f"prior_depth_edge_band_r{radius}"] = unavailable_region("DA3 prior depth was not provided")

    return {
        "stem": stem,
        "paths": {
            "rendered_depth": pred_path,
            "lidar_depth": lidar_path,
            "rgb": rgb_path,
            "acc": acc_path,
            "object_boundary": object_boundary_path,
            "prior_depth": prior_depth_path,
        },
        "metrics": frame_metrics,
        "crops": crops,
    }


def main():
    args = parse_args()
    if args.checkpoint:
        raise NotImplementedError(
            "--checkpoint rendering is intentionally not implemented here. Run render.py for the checkpoint first, "
            "then pass the rendered ours_* directory with --pred-dir."
        )

    output_dir = args.output_dir or os.path.join(args.pred_dir, "geometry_credibility_eval")
    ensure_dir(output_dir)

    frames = args.frames if args.frames is not None else list(range(args.frame_start, args.frame_end + 1))
    cam_ids = args.cam_ids if args.cam_ids is not None else [args.cam_id]
    region_accumulator = defaultdict(list)
    frame_results = []

    for frame in frames:
        for cam_id in cam_ids:
            stem = f"{frame:06d}_{cam_id}"
            result = evaluate_frame(args, stem, ensure_dir(os.path.join(output_dir, stem)), region_accumulator)
            if result is not None:
                frame_results.append(result)

    aggregate = {name: aggregate_region_metrics(values) for name, values in sorted(region_accumulator.items())}
    for name in ["all_valid", "acc_depth_filtered", "boundary_band", "thin_structure_band", "stable_non_boundary"]:
        aggregate.setdefault(name, unavailable_region("region was not generated"))
    for radius in args.boundary_band_radius:
        for prefix, reason in [
            ("canny_band", "RGB image was not found"),
            ("object_boundary_band", "object boundary or boundary mask was not provided"),
            ("prior_depth_edge_band", "DA3 prior depth was not provided"),
            ("rendered_depth_edge_band", "region was not generated"),
        ]:
            aggregate.setdefault(f"{prefix}_r{radius}", unavailable_region(reason))

    result_json = {
        "config": {
            "pred_dir": args.pred_dir,
            "lidar_dir": args.lidar_dir,
            "image_dir": args.image_dir,
            "object_boundary_dir": args.object_boundary_dir,
            "prior_depth_dir": args.prior_depth_dir,
            "acc_dir": args.acc_dir or args.pred_dir,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "min_acc": args.min_acc,
            "min_acc_available": True,
            "boundary_band_radius": args.boundary_band_radius,
            "frames": frames,
            "cam_ids": cam_ids,
            "save_region_visuals": args.save_region_visuals,
            "save_crops": args.save_crops,
            "metric_ground_truth": "LiDAR depth only; DA3 prior depth is never treated as ground truth.",
        },
        "aggregate": aggregate,
        "frames": frame_results,
    }
    result_json.update(aggregate)

    metrics_path = os.path.join(output_dir, "geometry_credibility_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    print(f"Evaluated {len(frame_results)} frames.")
    print(f"Saved metrics: {metrics_path}")
    if not frame_results:
        print("No frames were evaluated. Check --pred-dir, --lidar-dir, --frames, and --cam-id.")


if __name__ == "__main__":
    main()
