import argparse
import csv
import json
import os
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PWD", PROJECT_ROOT)


def parse_script_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--contribution-summary", default="output/local_feedback/da3_boundary_contribution_debug_A5000_top30_regionpixels/contribution_responsibility_all_views_summary.json")
    parser.add_argument("--softpatch-signal", default="output/local_feedback/da3_boundary_soft_contribution_feedback_A5000_top30/da3_contribution_softpatch_feedback_signal.json")
    parser.add_argument("--output-dir", default="output/local_feedback/da3_structure_counterfactual_dryrun_A5000_top30")
    parser.add_argument("--top-regions", type=int, default=30)
    parser.add_argument("--top-contributors", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=150)
    parser.add_argument("--weaken-scale", type=float, default=0.0)
    parser.add_argument("--min-support-pixels", type=int, default=3)
    parser.add_argument("--min-mean-weight", type=float, default=0.01)
    parser.add_argument("--border-margin", type=int, default=8)
    parser.add_argument("--structure-epsilon", type=float, default=0.01)
    parser.add_argument("--rgb-error-tolerance", type=float, default=0.02)
    parser.add_argument("--patch-radius", type=int, default=8)
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--help-dryrun", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.help_dryrun:
        parser.print_help()
        sys.exit(0)
    sys.argv = [sys.argv[0]] + remaining
    return args


SCRIPT_ARGS = parse_script_args()

from lib.config import cfg  # noqa: E402
from lib.datasets.dataset import Dataset  # noqa: E402
from lib.models.scene import Scene  # noqa: E402
from lib.models.street_gaussian_model import StreetGaussianModel  # noqa: E402
from lib.utils.camera_utils import make_rasterizer  # noqa: E402
from lib.utils.da3_structure_feedback_utils import make_da3_bridge  # noqa: E402
from lib.utils.general_utils import safe_state  # noqa: E402
from lib.utils.sh_utils import eval_sh  # noqa: E402


STABLE_ID_STRIDE = 10_000_000


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def stable_namespace_id(model_name):
    if model_name == "background":
        return 0
    if model_name.startswith("obj_"):
        try:
            return 1 + int(model_name.split("_", 1)[1])
        except ValueError:
            pass
    return 1_000_000 + (zlib.crc32(model_name.encode("utf-8")) % 8_000_000)


def build_stable_ids(pc, count):
    stable_ids = np.full(count, -1, dtype=np.int64)
    model_names = np.full(count, "unknown", dtype="<U64")
    model_local = np.full(count, -1, dtype=np.int64)
    for model_name, span in getattr(pc, "graph_gaussian_range", {}).items():
        start, end = int(span[0]), int(span[1])
        local = np.arange(max(0, end - start), dtype=np.int64)
        stable_ids[start:end] = stable_namespace_id(model_name) * STABLE_ID_STRIDE + local
        model_names[start:end] = model_name
        model_local[start:end] = local
    fallback = stable_ids < 0
    if np.any(fallback):
        rows = np.flatnonzero(fallback).astype(np.int64)
        stable_ids[fallback] = 9_999_999 * STABLE_ID_STRIDE + rows
        model_local[fallback] = rows
    return stable_ids, model_names, model_local


def prepare_raster_inputs(camera, pc, opacity_override=None, semantics=None):
    if not hasattr(pc, "graph_gaussian_range") or not getattr(pc, "graph_gaussian_range", None):
        pc.set_visibility(list(pc.model_name_id.keys()))
        pc.parse_camera(camera)
    means3D = pc.get_xyz
    opacity = pc.get_opacity if opacity_override is None else opacity_override
    scales = None
    rotations = None
    cov3D_precomp = None
    if cfg.render.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(cfg.render.scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if cfg.render.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = pc.get_xyz - camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        shs = pc.get_features

    means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=False, device=means3D.device)[:, :2]
    rasterizer = make_rasterizer(camera, pc.active_sh_degree)
    return rasterizer, dict(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        shs=shs,
        colors_precomp=colors_precomp,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        semantics=semantics,
    )


def render_custom(camera, pc, opacity_override=None):
    pc.set_visibility(list(pc.model_name_id.keys()))
    pc.parse_camera(camera)
    rasterizer, kwargs = prepare_raster_inputs(camera, pc, opacity_override=opacity_override)
    color, radii, depth, alpha, semantic = rasterizer(**kwargs)
    return {"rgb": color, "radii": radii, "depth": depth, "acc": alpha, "semantic": semantic}


def normalize_depth_np(depth, mask):
    valid = mask & np.isfinite(depth)
    if not np.any(valid):
        return np.zeros_like(depth, dtype=np.float32)
    vals = depth[valid].astype(np.float32)
    lo, hi = np.quantile(vals, [0.05, 0.95])
    return np.clip((depth.astype(np.float32) - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def grad_mag_np(depth):
    dx = np.zeros_like(depth, dtype=np.float32)
    dy = np.zeros_like(depth, dtype=np.float32)
    dx[:, :-1] = depth[:, 1:] - depth[:, :-1]
    dy[:-1, :] = depth[1:, :] - depth[:-1, :]
    return np.sqrt(dx * dx + dy * dy + 1e-8)


def da3_structure_components(rendered_depth, acc, da3_depth, mask, edge_margin=0.05, ranking_margin=0.02):
    valid = mask & np.isfinite(rendered_depth) & np.isfinite(da3_depth) & np.isfinite(acc) & (acc > 0.03)
    if not np.any(valid):
        z = np.zeros_like(rendered_depth, dtype=np.float32)
        return {"valid_count": 0, "total": None, "edge": None, "ranking": None, "side": None, "loss_map": z}
    rd = normalize_depth_np(rendered_depth / np.maximum(acc, 1e-6), valid)
    dd = normalize_depth_np(da3_depth, valid)
    rg = grad_mag_np(rd)
    dg = grad_mag_np(dd)
    edge_map = np.maximum(dg - rg - edge_margin, 0.0).astype(np.float32)
    side_map = np.maximum(np.abs(dg - rg) - edge_margin, 0.0).astype(np.float32)

    ranking_maps = []
    for sy, sx in [(0, 1), (1, 0)]:
        h = rd.shape[0] - sy if sy else rd.shape[0]
        w = rd.shape[1] - sx if sx else rd.shape[1]
        rd_a = rd[:h, :w]
        rd_b = rd[sy:sy + h, sx:sx + w]
        dd_a = dd[:h, :w]
        dd_b = dd[sy:sy + h, sx:sx + w]
        vm = valid[:h, :w]
        da3_delta = dd_b - dd_a
        render_delta = rd_b - rd_a
        confident = vm & (np.abs(da3_delta) > ranking_margin)
        local = np.zeros_like(rd, dtype=np.float32)
        vals = np.maximum(ranking_margin - np.sign(da3_delta) * render_delta, 0.0)
        tmp = np.zeros((h, w), dtype=np.float32)
        tmp[confident] = vals[confident].astype(np.float32)
        local[:h, :w] = tmp
        ranking_maps.append(local)
    ranking_map = np.mean(ranking_maps, axis=0) if ranking_maps else np.zeros_like(rd, dtype=np.float32)
    loss_map = edge_map + ranking_map + 0.5 * side_map
    return {
        "valid_count": int(np.count_nonzero(valid)),
        "total": float(np.mean(loss_map[valid])),
        "edge": float(np.mean(edge_map[valid])),
        "ranking": float(np.mean(ranking_map[valid])),
        "side": float(np.mean(side_map[valid])),
        "loss_map": loss_map,
    }


def bbox_from_frame(frame, shape, pad=0):
    bbox = (frame.get("input_region") or {}).get("bbox") or frame.get("pixel_bbox")
    if not bbox:
        pixels = np.asarray(frame.get("selected_pixels", []), dtype=np.int64)
        if pixels.size == 0:
            return [0, 0, shape[1], shape[0]]
        x0, y0 = pixels.min(axis=0)
        x1, y1 = pixels.max(axis=0) + 1
    else:
        x0, y0, x1, y1 = [int(v) for v in bbox]
    x0 = max(0, min(shape[1], x0 - pad))
    x1 = max(0, min(shape[1], x1 + pad))
    y0 = max(0, min(shape[0], y0 - pad))
    y1 = max(0, min(shape[0], y1 + pad))
    return [x0, y0, x1, y1]


def patch_mask_from_pixels(shape, pixels, bbox, radius):
    mask = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = bbox
    mask[y0:y1, x0:x1] = True
    if pixels.size:
        soft = np.zeros(shape, dtype=np.uint8)
        for x, y in pixels:
            if 0 <= x < shape[1] and 0 <= y < shape[0]:
                soft[int(y), int(x)] = 1
        k = 2 * max(0, radius) + 1
        if k > 1:
            soft = cv2.dilate(soft, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
        mask &= soft.astype(bool)
        if not np.any(mask):
            mask[y0:y1, x0:x1] = True
    return mask


def load_da3_depth(da3_bridge, camera, shape):
    guidance = da3_bridge(camera)
    depth = guidance["relative_depth"]
    if torch.is_tensor(depth):
        depth = to_numpy(depth.float())
    depth = np.squeeze(depth).astype(np.float32)
    if depth.shape != shape:
        depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return depth


def aggregate_candidate_rows(npz, stable_ids, model_names, model_local, top_n):
    ids = np.asarray(npz["cuda_contribution_ids"], dtype=np.int64)
    weights = np.asarray(npz["contribution_weights"], dtype=np.float32)
    depth_order = np.asarray(npz["cuda_depth_order"], dtype=np.int64)
    depths = np.asarray(npz["cuda_depth"], dtype=np.float32)
    rows = {}
    for p_idx in range(ids.shape[0]):
        for k in range(ids.shape[1]):
            row = int(ids[p_idx, k])
            if row < 0:
                continue
            if row >= len(stable_ids):
                continue
            weight = float(weights[p_idx, k])
            if weight <= 0:
                continue
            rec = rows.setdefault(
                row,
                {
                    "view_local_index": row,
                    "stable_gaussian_id": int(stable_ids[row]),
                    "model_name": str(model_names[row]),
                    "model_local_index": int(model_local[row]),
                    "support_pixel_count": 0,
                    "weight_sum": 0.0,
                    "max_talpha": 0.0,
                    "depth_orders": [],
                    "depths": [],
                },
            )
            rec["support_pixel_count"] += 1
            rec["weight_sum"] += weight
            rec["max_talpha"] = max(rec["max_talpha"], weight)
            rec["depth_orders"].append(int(depth_order[p_idx, k]))
            rec["depths"].append(float(depths[p_idx, k]))
    candidates = []
    for rec in rows.values():
        rec["mean_talpha"] = float(rec["weight_sum"] / max(rec["support_pixel_count"], 1))
        rec["mean_depth_order"] = float(np.mean(rec["depth_orders"])) if rec["depth_orders"] else None
        rec["mean_contributor_depth"] = float(np.mean(rec["depths"])) if rec["depths"] else None
        candidates.append(rec)
    candidates.sort(key=lambda r: (r["weight_sum"], r["support_pixel_count"], r["max_talpha"]), reverse=True)
    return candidates[:top_n]


def rgb_patch_metrics(rgb, original, mask):
    if not np.any(mask):
        return {"mae": None, "psnr": None}
    diff = np.abs(rgb - original)
    mse = np.mean((rgb[mask] - original[mask]) ** 2)
    return {
        "mae": float(np.mean(diff[mask])),
        "psnr": float(-10.0 * np.log10(max(float(mse), 1e-10))),
    }


def classify(delta_structure, delta_rgb, rec, args, border_suspect):
    low_reasons = []
    if rec["support_pixel_count"] < args.min_support_pixels:
        low_reasons.append("low_support_pixels")
    if rec["mean_talpha"] < args.min_mean_weight:
        low_reasons.append("weak_contribution")
    if border_suspect:
        low_reasons.append("border_suspect")
    if low_reasons:
        return "low_evidence", 0.2, "skip", low_reasons
    if delta_structure > args.structure_epsilon and delta_rgb <= args.rgb_error_tolerance:
        conf = min(1.0, 0.5 + delta_structure * 5.0 + rec["mean_talpha"])
        return "bad_contributor", float(conf), "opacity_regularization_candidate", []
    if delta_structure < -args.structure_epsilon or delta_rgb > args.rgb_error_tolerance:
        conf = min(1.0, 0.5 + abs(delta_structure) * 5.0 + max(delta_rgb, 0.0))
        return "good_contributor", float(conf), "protect", []
    return "neutral_contributor", 0.5, "skip", []


def save_overlay_panel(path, original, base_rgb, da3_loss_map, after_loss_map, patch_mask, pixels, bbox, tag_text):
    x0, y0, x1, y1 = bbox
    def bgr(img):
        return cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    def color(x):
        v = x.copy()
        valid = np.isfinite(v)
        if np.any(valid):
            lo, hi = np.percentile(v[valid], [1, 99])
            v = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
        else:
            v[:] = 0
        return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    mask_vis = bgr(base_rgb)
    mask_vis[patch_mask] = (0.5 * mask_vis[patch_mask] + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
    for x, y in pixels:
        cv2.circle(mask_vis, (int(x), int(y)), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    panels = [bgr(original), bgr(base_rgb), mask_vis, color(da3_loss_map), color(after_loss_map), color(np.maximum(da3_loss_map - after_loss_map, 0))]
    panels = [p[y0:y1, x0:x1] for p in panels]
    h = max(max(p.shape[0], 1) for p in panels)
    w = max(max(p.shape[1], 1) for p in panels)
    panels = [cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in panels]
    sheet = np.concatenate(panels, axis=1)
    cv2.putText(sheet, tag_text[:120], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, sheet)


def process(args):
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    summary = read_json(args.contribution_summary)
    frames = summary.get("frames", [])[: args.top_regions]
    signal = read_json(args.softpatch_signal)

    safe_state(False)
    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        cameras = {cam.image_name: cam for cam in scene.getTrainCameras()}
        if cfg.loaded_iter == -1:
            ckpt_path = os.path.join(cfg.trained_model_dir, "iteration_5000.pth")
        else:
            ckpt_path = os.path.join(cfg.trained_model_dir, f"iteration_{cfg.loaded_iter}.pth")
        state_dict = torch.load(ckpt_path)
        gaussians.load_state_dict(state_dict)
        pc = gaussians
        da3_bridge = make_da3_bridge(cfg.geovit)

        candidates_json = []
        per_region = []
        per_gaussian = []
        low_regions = []
        total_candidates = 0
        t0 = time.perf_counter()
        for frame in frames:
            if total_candidates >= args.max_candidates:
                break
            stem = str(frame.get("stem"))
            region_id = str(frame.get("region_id"))
            region_key = f"{stem}:region{region_id}"
            if frame.get("status") != "ok":
                low_regions.append({"region_key": region_key, "reason": frame.get("status", "not_ok")})
                continue
            camera = cameras.get(stem)
            npz_path = Path(frame.get("paths", {}).get("npz", ""))
            if camera is None or not npz_path.exists():
                low_regions.append({"region_key": region_key, "reason": "missing_camera_or_npz"})
                continue
            npz = np.load(npz_path, allow_pickle=True)
            pixels = np.asarray(npz["selected_pixels"], dtype=np.int64)
            if pixels.size == 0:
                low_regions.append({"region_key": region_key, "reason": "no_selected_da3_risk_pixels"})
                continue
            base = render_custom(camera, pc)
            stable_ids, model_names, model_local = build_stable_ids(pc, pc.get_xyz.shape[0])
            base_opacity = pc.get_opacity
            base_depth = np.squeeze(to_numpy(base["depth"]).astype(np.float32))
            base_acc = np.squeeze(to_numpy(base["acc"]).astype(np.float32))
            base_rgb = np.transpose(to_numpy(base["rgb"]).astype(np.float32), (1, 2, 0))
            original = np.transpose(to_numpy(camera.original_image).astype(np.float32), (1, 2, 0))
            da3_depth = load_da3_depth(da3_bridge, camera, base_depth.shape)
            bbox = bbox_from_frame(frame, base_depth.shape, pad=args.patch_radius)
            patch_mask = patch_mask_from_pixels(base_depth.shape, pixels, bbox, args.patch_radius)
            border = bbox[0] <= args.border_margin or bbox[1] <= args.border_margin or bbox[2] >= base_depth.shape[1] - args.border_margin or bbox[3] >= base_depth.shape[0] - args.border_margin
            before = da3_structure_components(base_depth, base_acc, da3_depth, patch_mask)
            rgb_before = rgb_patch_metrics(base_rgb, original, patch_mask)
            candidate_rows = aggregate_candidate_rows(npz, stable_ids, model_names, model_local, args.top_contributors)
            if before["valid_count"] == 0 or not candidate_rows:
                low_regions.append({"region_key": region_key, "reason": "no_valid_da3_structure_or_contributors"})
                continue
            region_records = []
            for rec in candidate_rows:
                if total_candidates >= args.max_candidates:
                    break
                row = int(rec["view_local_index"])
                opacity = base_opacity.clone()
                opacity[row] = opacity[row] * float(args.weaken_scale)
                weakened = render_custom(camera, pc, opacity_override=opacity)
                after_depth = np.squeeze(to_numpy(weakened["depth"]).astype(np.float32))
                after_acc = np.squeeze(to_numpy(weakened["acc"]).astype(np.float32))
                after_rgb = np.transpose(to_numpy(weakened["rgb"]).astype(np.float32), (1, 2, 0))
                after = da3_structure_components(after_depth, after_acc, da3_depth, patch_mask)
                rgb_after = rgb_patch_metrics(after_rgb, original, patch_mask)
                delta_structure = float(before["total"] - after["total"]) if before["total"] is not None and after["total"] is not None else 0.0
                delta_rgb = float(rgb_after["mae"] - rgb_before["mae"]) if rgb_after["mae"] is not None and rgb_before["mae"] is not None else 0.0
                tag, confidence, action, evidence_flags = classify(delta_structure, delta_rgb, rec, args, border)
                visual_path = None
                if args.save_visuals:
                    visual_path = str(out_dir / "overlays" / f"{stem}_{region_id}_gid{rec['stable_gaussian_id']}_{tag}.png")
                    save_overlay_panel(
                        visual_path,
                        original,
                        base_rgb,
                        before["loss_map"],
                        after["loss_map"],
                        patch_mask,
                        pixels,
                        bbox,
                        f"{region_key} gid={rec['stable_gaussian_id']} {tag} conf={confidence:.2f}",
                    )
                item = {
                    "region_key": region_key,
                    "view_id": stem,
                    "region_id": region_id,
                    "stable_gaussian_id": int(rec["stable_gaussian_id"]),
                    "view_local_index": int(rec["view_local_index"]),
                    "model_name": rec["model_name"],
                    "model_local_index": int(rec["model_local_index"]),
                    "mean_talpha": float(rec["mean_talpha"]),
                    "max_talpha": float(rec["max_talpha"]),
                    "support_pixel_count": int(rec["support_pixel_count"]),
                    "mean_depth_order": rec["mean_depth_order"],
                    "mean_contributor_depth": rec["mean_contributor_depth"],
                    "da3_structure_before": before["total"],
                    "da3_structure_after": after["total"],
                    "da3_structure_delta": delta_structure,
                    "da3_edge_before": before["edge"],
                    "da3_edge_after": after["edge"],
                    "da3_ranking_before": before["ranking"],
                    "da3_ranking_after": after["ranking"],
                    "da3_side_before": before["side"],
                    "da3_side_after": after["side"],
                    "rgb_patch_mae_before": rgb_before["mae"],
                    "rgb_patch_mae_after": rgb_after["mae"],
                    "rgb_patch_mae_delta": delta_rgb,
                    "rgb_patch_psnr_before": rgb_before["psnr"],
                    "rgb_patch_psnr_after": rgb_after["psnr"],
                    "tag": tag,
                    "confidence": confidence,
                    "suggested_future_action": action,
                    "evidence_flags": evidence_flags,
                    "border_suspect": bool(border),
                    "counterfactual_objective": "da3_structure",
                    "uses_lidar_for_labeling": False,
                    "uses_lidar_for_evaluation_only": True,
                    "overlay_path": visual_path,
                }
                candidates_json.append(item)
                per_gaussian.append(item)
                region_records.append(item)
                total_candidates += 1
                del opacity, weakened
                torch.cuda.empty_cache()
            counts = defaultdict(int)
            for r in region_records:
                counts[r["tag"]] += 1
            per_region.append(
                {
                    "region_key": region_key,
                    "view_id": stem,
                    "region_id": region_id,
                    "candidate_count": len(region_records),
                    "selected_pixel_count": int(len(pixels)),
                    "da3_structure_valid_pixels": before["valid_count"],
                    "da3_structure_before": before["total"],
                    "rgb_patch_mae_before": rgb_before["mae"],
                    "border_suspect": bool(border),
                    "label_counts": dict(counts),
                }
            )

    label_counts = defaultdict(int)
    action_counts = defaultdict(int)
    for item in candidates_json:
        label_counts[item["tag"]] += 1
        action_counts[item["suggested_future_action"]] += 1
    payload = {
        "counterfactual_objective": "da3_structure",
        "uses_lidar_for_labeling": False,
        "uses_lidar_for_evaluation_only": True,
        "debug_only": True,
        "does_not_modify_checkpoint": True,
        "weaken_scale": args.weaken_scale,
        "top_regions": args.top_regions,
        "top_contributors": args.top_contributors,
        "thresholds": {
            "min_support_pixels": args.min_support_pixels,
            "min_mean_weight": args.min_mean_weight,
            "border_margin": args.border_margin,
            "structure_epsilon": args.structure_epsilon,
            "rgb_error_tolerance": args.rgb_error_tolerance,
            "patch_radius": args.patch_radius,
        },
        "input_paths": {
            "checkpoint_dir": cfg.trained_model_dir,
            "contribution_summary": args.contribution_summary,
            "softpatch_signal": args.softpatch_signal,
        },
        "counts": {
            "regions_processed": len(per_region),
            "candidate_count": len(candidates_json),
            "low_evidence_region_count": len(low_regions),
            "labels": dict(label_counts),
            "suggested_actions": dict(action_counts),
        },
        "runtime_sec": float(time.perf_counter() - t0),
        "stopping_policy": "If most regions are low-evidence or stable ids are unreliable, keep this as an audit-only dry run and do not proceed to repair.",
    }
    write_json(out_dir / "counterfactual_candidates.json", candidates_json)
    write_json(out_dir / "counterfactual_summary.json", payload)
    write_json(out_dir / "low_evidence_regions.json", low_regions)
    with open(out_dir / "per_region_counterfactual_table.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["region_key", "view_id", "region_id", "candidate_count", "selected_pixel_count", "da3_structure_valid_pixels", "da3_structure_before", "rgb_patch_mae_before", "border_suspect", "label_counts"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_region)
    with open(out_dir / "per_gaussian_counterfactual_table.csv", "w", newline="", encoding="utf-8") as f:
        fields = [
            "stable_gaussian_id", "view_id", "region_id", "view_local_index", "model_name", "model_local_index",
            "mean_talpha", "max_talpha", "support_pixel_count", "mean_depth_order", "mean_contributor_depth",
            "da3_structure_before", "da3_structure_after", "da3_structure_delta",
            "rgb_patch_mae_before", "rgb_patch_mae_after", "rgb_patch_mae_delta",
            "tag", "confidence", "suggested_future_action", "evidence_flags", "border_suspect",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_gaussian)
    return payload


def main():
    ensure_dir(SCRIPT_ARGS.output_dir)
    payload = process(SCRIPT_ARGS)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
