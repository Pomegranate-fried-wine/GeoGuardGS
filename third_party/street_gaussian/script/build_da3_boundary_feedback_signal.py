import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Build DA3 boundary-structure guided feedback signal.")
    parser.add_argument("--image-dir", default="data/waymo/002/images")
    parser.add_argument("--render-dir", default="output/local_formal/p15_allcam_A_da3_only_5000/train/ours_5000")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", default="../.cache/huggingface/hub/models--depth-anything--DA3-LARGE-1.1/snapshots/main")
    parser.add_argument("--da3-depth-dir", default=None, help="Optional precomputed DA3 depth cache directory.")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only")
    parser.add_argument("--views", nargs="*", default=None)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=15)
    parser.add_argument("--cameras", nargs="*", type=int, default=None)
    parser.add_argument("--max-regions", type=int, default=30)
    parser.add_argument("--patch-radius", type=int, default=24)
    parser.add_argument("--process-res", type=int, default=128)
    parser.add_argument("--da3-edge-percentile", type=float, default=92.0)
    parser.add_argument("--render-edge-percentile", type=float, default=85.0)
    parser.add_argument("--min-risk-pixels", type=int, default=20)
    parser.add_argument("--save-visuals", action="store_true")
    return parser.parse_args()


def resolve_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def stems_from_args(args):
    if args.views:
        return args.views
    cams = args.cameras if args.cameras else [0, 1, 2, 3, 4]
    return [f"{frame:06d}_{cam}" for frame in range(args.frame_start, args.frame_end) for cam in cams]


def load_da3_model(model_dir, local_files_only=True):
    from depth_anything_3.api import DepthAnything3

    model_path = str(resolve_path(model_dir)) if Path(model_dir).exists() or str(model_dir).startswith(".") else model_dir
    model = DepthAnything3.from_pretrained(model_path, local_files_only=local_files_only).to("cuda")
    model.eval()
    return model


def load_cached_da3_depth(directory, stem, shape):
    if not directory:
        return None
    root = Path(directory)
    for suffix in ["_da3_depth.npy", "_depth.npy", "_prior_depth.npy", ".npy"]:
        path = root / f"{stem}{suffix}"
        if path.exists():
            depth = np.load(path, allow_pickle=True)
            if isinstance(depth, np.ndarray) and depth.shape == () and isinstance(depth.item(), dict):
                depth = depth.item().get("relative_depth", depth.item().get("depth"))
            depth = np.asarray(depth).squeeze().astype(np.float32)
            if depth.shape[:2] != shape:
                depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            return depth
    return None


def infer_da3(model, image_bgr, process_res):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        out = model.inference([image_rgb], process_res=process_res, process_res_method="upper_bound_resize")
    if isinstance(out, dict):
        depth = out.get("relative_depth", out.get("depth", out.get("pred_depth")))
    elif isinstance(out, (list, tuple)):
        depth = out[0]
    elif hasattr(out, "depth"):
        depth = out.depth
    elif hasattr(out, "relative_depth"):
        depth = out.relative_depth
    elif hasattr(out, "pred_depth"):
        depth = out.pred_depth
    else:
        depth = out
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[0]
    return cv2.resize(depth.astype(np.float32), (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


def normalize_map(x):
    valid = np.isfinite(x)
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(x[valid], [5, 95])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def grad_mag(depth):
    depth = normalize_map(depth)
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def colorize(x, vmin=None, vmax=None):
    x = np.asarray(x, np.float32)
    valid = np.isfinite(x)
    if not np.any(valid):
        return np.zeros((*x.shape[:2], 3), np.uint8)
    if vmin is None:
        vmin = float(np.percentile(x[valid], 2))
    if vmax is None:
        vmax = float(np.percentile(x[valid], 98))
    img = (np.clip((x - vmin) / max(vmax - vmin, 1e-6), 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(img, cv2.COLORMAP_TURBO)


def choose_patches(stem, risk, da3_edge, render_edge, patch_radius, min_pixels, max_pixels_per_region=256):
    mask = risk > 0
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    patches = []
    h, w = risk.shape
    for label in range(1, num_labels):
        component = labels == label
        count = int(component.sum())
        if count < min_pixels:
            continue
        cx, cy = centroids[label]
        x0 = max(0, int(round(cx)) - patch_radius)
        x1 = min(w, int(round(cx)) + patch_radius + 1)
        y0 = max(0, int(round(cy)) - patch_radius)
        y1 = min(h, int(round(cy)) + patch_radius + 1)
        patch = np.zeros_like(component)
        patch[y0:y1, x0:x1] = True
        patch_component = component & patch & np.isfinite(risk) & (risk > 0)
        pys, pxs = np.where(patch_component)
        feedback_pixels = []
        if len(pxs) > 0:
            order = np.argsort(risk[pys, pxs])[::-1][: min(max_pixels_per_region, len(pxs))]
            feedback_pixels = [[int(pxs[i]), int(pys[i])] for i in order]
        score = float(risk[component].mean() * np.log1p(count))
        patches.append(
            {
                "view_id": stem,
                "region_id": f"da3risk_{int(round(cx))}_{int(round(cy))}",
                "region_key": f"{stem}:regionda3risk_{int(round(cx))}_{int(round(cy))}",
                "region_type": "da3_boundary_risk_region",
                "bbox": [x0, y0, x1, y1],
                "risk_pixel_count": count,
                "mean_da3_edge": float(da3_edge[component].mean()),
                "mean_render_edge": float(render_edge[component].mean()),
                "risk_score": score,
                "evidence_status": "ok",
                "feedback_pixels": feedback_pixels,
            }
        )
    return patches


def main():
    args = parse_args()
    out_dir = ensure_dir(Path(args.output_dir))
    visual_dir = ensure_dir(out_dir / "risk_visuals")
    model = None
    if not args.da3_depth_dir:
        model = load_da3_model(args.model_dir, args.local_files_only)
    regions = []
    pixel_feedback = []

    for stem in stems_from_args(args):
        image_path = Path(args.image_dir) / f"{stem}.png"
        if not image_path.exists():
            image_path = Path(args.image_dir) / f"{stem}.jpg"
        depth_path = Path(args.render_dir) / f"{stem}_depth.npy"
        if not image_path.exists() or not depth_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        rendered = np.load(depth_path, allow_pickle=True).squeeze().astype(np.float32)
        target_shape = rendered.shape[:2]
        image_vis = image
        if image_vis.shape[:2] != target_shape:
            image_vis = cv2.resize(image_vis, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
        da3 = load_cached_da3_depth(args.da3_depth_dir, stem, target_shape)
        if da3 is None:
            if model is None:
                model = load_da3_model(args.model_dir, args.local_files_only)
            da3 = infer_da3(model, image, args.process_res)
            if da3.shape[:2] != target_shape:
                da3 = cv2.resize(da3, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        da3_edge = grad_mag(da3)
        render_edge = grad_mag(rendered)
        da3_thr = np.percentile(da3_edge[np.isfinite(da3_edge)], args.da3_edge_percentile)
        render_thr = np.percentile(render_edge[np.isfinite(render_edge)], args.render_edge_percentile)
        da3_band = da3_edge >= da3_thr
        render_band = render_edge >= render_thr
        render_band_dilated = cv2.dilate(render_band.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        weak_render = da3_band & (render_edge < render_thr)
        misaligned = da3_band & (~render_band_dilated)
        risk = np.zeros_like(da3_edge, dtype=np.float32)
        risk[weak_render] += normalize_map(da3_edge)[weak_render]
        risk[misaligned] += 0.5 * normalize_map(da3_edge)[misaligned]
        patches = choose_patches(stem, risk, da3_edge, render_edge, args.patch_radius, args.min_risk_pixels)
        regions.extend(patches)
        if args.save_visuals:
            vis = np.concatenate(
                [
                    image_vis,
                    colorize(da3_edge),
                    colorize(render_edge),
                    colorize(risk, vmin=0, vmax=max(float(np.percentile(risk, 99)), 1e-6)),
                ],
                axis=1,
            )
            cv2.imwrite(str(visual_dir / f"{stem}_da3_boundary_risk.png"), vis)

    regions = sorted(regions, key=lambda item: item["risk_score"], reverse=True)[: args.max_regions]
    pixel_feedback_by_view = {}
    for region in regions:
        pixels = region.get("feedback_pixels", [])
        if not pixels:
            continue
        view_record = pixel_feedback_by_view.setdefault(
            region["view_id"], {"view_id": region["view_id"], "bad_pixels": [], "good_pixels": [], "regions": []}
        )
        view_record["bad_pixels"].extend(pixels)
        view_record["regions"].append(region["region_key"])
    pixel_feedback = []
    for view_record in pixel_feedback_by_view.values():
        seen = set()
        unique_pixels = []
        for x, y in view_record["bad_pixels"]:
            key = (int(x), int(y))
            if key in seen:
                continue
            seen.add(key)
            unique_pixels.append([int(x), int(y)])
        view_record["bad_pixels"] = unique_pixels
        pixel_feedback.append(view_record)

    bad_contributors = [
        {
            "region_key": r["region_key"],
            "view_id": r["view_id"],
            "region_id": r["region_id"],
            "region_type": r["region_type"],
            "gaussian_id": -1,
            "counterfactual_label": "da3_boundary_risk_region",
            "recommended_feedback": "upweight_da3_boundary_structure",
        }
        for r in regions
    ]
    signal = {
        "debug_only": False,
        "source": "DA3 depth-edge/rendered-depth-edge mismatch boundary-risk map; no LiDAR training supervision",
        "feedback_type": "da3_boundary_structure",
        "counts": {
            "regions": len(regions),
            "bad_contributors": len(bad_contributors),
            "good_contributors": 0,
            "low_evidence_regions": 0,
            "bad_feedback_pixels": int(sum(len(v["bad_pixels"]) for v in pixel_feedback)),
            "pixel_feedback_views": len(pixel_feedback),
        },
        "regions": regions,
        "bad_contributors": bad_contributors,
        "good_contributors": [],
        "neutral_contributors": [],
        "low_evidence_regions": [],
        "pixel_feedback_by_view": pixel_feedback,
        "notes": [
            "DA3 is used only as a structure prior: edge/discontinuity/ranking, not absolute depth ground truth.",
            "CUDA contributor classification can be layered on top in the next pass by running selected-pixel dump on these risk pixels.",
        ],
    }
    with open(out_dir / "guided_training_feedback_signal_da3_boundary_top30.json", "w", encoding="utf-8") as f:
        json.dump(signal, f, indent=2)
    with open(out_dir / "da3_boundary_regions.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["view_id", "region_id", "region_key", "region_type", "bbox", "risk_pixel_count", "mean_da3_edge", "mean_render_edge", "risk_score", "evidence_status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(regions)
    with open(out_dir / "da3_boundary_feedback_summary.json", "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "counts": signal["counts"], "top_regions": regions[:10]}, f, indent=2)
    print(json.dumps({"output_dir": str(out_dir), "counts": signal["counts"]}, indent=2))


if __name__ == "__main__":
    main()
