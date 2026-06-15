#!/usr/bin/env python3
"""Run paper-grade final evaluation for GeoGuardGS/StreetGS experiments.

The default protocol evaluates the held-out test split only. Training split
evaluation is available with ``--splits test train`` but is intentionally not
the default because the formal Waymo setup has hundreds of training views per
experiment. Per-view CSV files are appended during evaluation so interrupted
runs keep completed rows and can resume.
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


SCOPES = ["full_image", "object_region", "background_region"]
METRICS = ["l1", "psnr", "ssim", "lpips"]
PER_VIEW_FIELDS = [
    "split", "scope", "view_index", "cam_id", "frame", "frame_idx",
    "image_name", "valid_pixel_count", "status", "l1", "psnr", "ssim",
    "lpips", "warning",
]
SUMMARY_FIELDS = ["scope", "split", "view_count"]
for metric in METRICS:
    SUMMARY_FIELDS.extend([
        f"{metric}_mean", f"{metric}_median", f"{metric}_std",
        f"{metric}_min", f"{metric}_max",
    ])


def _deep_merge(base, override):
    result = dict(base or {})
    for key, value in (override or {}).items():
        if key == "_BASE_":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_merged_config(config_path):
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        current = yaml.safe_load(f) or {}
    base_ref = current.get("_BASE_")
    if not base_ref:
        return current
    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = (config_path.parent / base_path).resolve()
    return _deep_merge(_load_merged_config(base_path), current)


def _materialize_config(repo_root, config_path, out_dir, loaded_iter):
    payload = _load_merged_config(config_path)
    payload["workspace"] = str(repo_root)
    payload["mode"] = "evaluate"
    payload["loaded_iter"] = int(loaded_iter)
    existing_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing_visible:
        payload["gpus"] = [-1]
        print(
            "[FinalEval][CUDA] Respect existing "
            f"CUDA_VISIBLE_DEVICES={existing_visible}; disable cfg.gpus override",
            flush=True,
        )
    else:
        print(
            "[FinalEval][CUDA] CUDA_VISIBLE_DEVICES is unset; "
            f"StreetGS may use cfg.gpus={payload.get('gpus', '<missing>')}",
            flush=True,
        )
    payload.setdefault("eval", {})
    payload["eval"]["skip_train"] = False
    payload["eval"]["skip_test"] = False
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(config_path).stem}_final_eval.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return out_path, payload


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def append_csv_row(path, row, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _scope_csv_path(exp_out, scope):
    return exp_out / f"metrics_{scope}.csv"


def _completed_view_keys(exp_out):
    per_scope = []
    for scope in SCOPES:
        keys = {
            (row.get("split", ""), row.get("image_name", ""))
            for row in read_csv(_scope_csv_path(exp_out, scope))
            if row.get("split") and row.get("image_name")
        }
        per_scope.append(keys)
    return set.intersection(*per_scope) if per_scope else set()


def _all_metric_rows(exp_out):
    rows = []
    for scope in SCOPES:
        for row in read_csv(_scope_csv_path(exp_out, scope)):
            row.setdefault("scope", scope)
            rows.append(row)
    return rows


def to_float(value):
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def summarize(rows, scope, split):
    subset = [
        r for r in rows
        if r.get("scope") == scope and r.get("split") == split
        and r.get("status") == "valid"
    ]
    out = {"scope": scope, "split": split, "view_count": len(subset)}
    for metric in METRICS:
        values = [to_float(r.get(metric)) for r in subset]
        values = [v for v in values if v is not None]
        if not values:
            for stat in ["mean", "median", "std", "min", "max"]:
                out[f"{metric}_{stat}"] = ""
            continue
        sorted_values = sorted(values)
        mid = len(sorted_values) // 2
        mean = sum(values) / len(values)
        out[f"{metric}_mean"] = mean
        out[f"{metric}_median"] = (
            sorted_values[mid]
            if len(sorted_values) % 2
            else 0.5 * (sorted_values[mid - 1] + sorted_values[mid])
        )
        out[f"{metric}_std"] = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        out[f"{metric}_min"] = min(values)
        out[f"{metric}_max"] = max(values)
    return out


def write_current_summaries(exp_out, splits):
    rows = _all_metric_rows(exp_out)
    summary_rows = [summarize(rows, scope, split) for scope in SCOPES for split in splits]
    write_csv(exp_out / "summary_by_scope.csv", summary_rows, SUMMARY_FIELDS)
    return summary_rows


def _save_panel(path, title_images):
    import cv2
    import numpy as np

    rows = []
    for title, img in title_images:
        arr = img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        arr = (arr * 255).astype(np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.putText(arr, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(arr, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(arr)
    panel = np.concatenate(rows, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)


def _metric_row(torch, loss_utils, lpips_fn, split, idx, camera, image, gt, mask, scope, include_lpips, lpips_scopes):
    mask = mask.bool()
    valid = int(torch.count_nonzero(mask).item())
    if valid == 0:
        return {
            "split": split,
            "scope": scope,
            "view_index": idx,
            "cam_id": camera.meta.get("cam", ""),
            "frame": camera.meta.get("frame", ""),
            "frame_idx": camera.meta.get("frame_idx", ""),
            "image_name": getattr(camera, "image_name", ""),
            "valid_pixel_count": 0,
            "status": "not_applicable",
            "warning": "empty_scope_mask",
        }
    row = {
        "split": split,
        "scope": scope,
        "view_index": idx,
        "cam_id": camera.meta.get("cam", ""),
        "frame": camera.meta.get("frame", ""),
        "frame_idx": camera.meta.get("frame_idx", ""),
        "image_name": getattr(camera, "image_name", ""),
        "valid_pixel_count": valid,
        "status": "valid",
        "warning": "",
    }
    row["l1"] = float(loss_utils.l1_loss(image, gt, mask).detach().cpu().item())
    row["psnr"] = float(loss_utils.psnr(image, gt, mask).detach().cpu().item())
    try:
        row["ssim"] = float(loss_utils.ssim(image, gt, mask=mask).detach().cpu().item())
    except Exception as exc:
        row["ssim"] = ""
        row["warning"] = f"ssim_failed:{exc}"
    if include_lpips and scope in lpips_scopes:
        try:
            masked_image = torch.where(mask, image, torch.zeros_like(image))
            masked_gt = torch.where(mask, gt, torch.zeros_like(gt))
            row["lpips"] = float(lpips_fn(masked_image, masked_gt, net_type="alex").detach().cpu().item())
        except Exception as exc:
            row["lpips"] = ""
            row["warning"] = (row["warning"] + ";" if row["warning"] else "") + f"lpips_failed:{exc}"
    else:
        row["lpips"] = ""
    return row


def evaluate_one(repo_root, config_path, final_root, args):
    streetgs_root = repo_root / "third_party" / "street_gaussian"
    sys.path.insert(0, str(streetgs_root))
    sys.path.insert(0, str(repo_root))
    os.chdir(streetgs_root)

    exp_out = final_root / Path(config_path).stem
    if args.overwrite and exp_out.exists():
        print(f"[FinalEval] overwrite=true; removing {exp_out}", flush=True)
        shutil.rmtree(exp_out)
    materialized, payload = _materialize_config(repo_root, config_path, exp_out / "configs", args.loaded_iter)
    sys.argv = ["final_evaluate_experiments.py", "--config", str(materialized)]

    import torch
    from lib.datasets.dataset import Dataset
    from lib.models.scene import Scene
    from lib.models.street_gaussian_model import StreetGaussianModel
    from lib.models.street_gaussian_renderer import StreetGaussianRenderer
    from lib.utils import loss_utils
    from lib.utils.lpipsPyTorch import lpips as lpips_fn

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for final evaluation.")

    completed = _completed_view_keys(exp_out) if args.resume else set()
    if completed:
        print(f"[FinalEval] resume=true; completed views found={len(completed)}", flush=True)

    requested_splits = list(dict.fromkeys(args.splits))
    lpips_scopes = set(args.lpips_scopes or [])
    split_cameras = {}
    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        if "train" in requested_splits:
            split_cameras["train"] = scene.getTrainCameras()
        if "test" in requested_splits:
            split_cameras["test"] = scene.getTestCameras()

        print(
            f"[FinalEval] experiment={Path(config_path).stem} loaded_iter={args.loaded_iter}",
            flush=True,
        )
        print(
            "[FinalEval] metrics=psnr,ssim,l1,"
            f"lpips_scopes={','.join(sorted(lpips_scopes)) if not args.skip_lpips else 'disabled'}",
            flush=True,
        )

        for split, cameras in split_cameras.items():
            if args.max_views_per_split > 0:
                cameras = cameras[:args.max_views_per_split]
                split_cameras[split] = cameras
            print(f"[FinalEval] split={split} views={len(cameras)}", flush=True)
            try:
                from tqdm import tqdm
                iterator = tqdm(cameras, desc=f"[FinalEval] {Path(config_path).stem} {split}", unit="view")
            except Exception:
                iterator = cameras
            for idx, camera in enumerate(iterator):
                image_name = getattr(camera, "image_name", "")
                if args.resume and (split, image_name) in completed:
                    continue
                render_pkg = renderer.render(camera, gaussians)
                image = torch.clamp(render_pkg["rgb"], 0.0, 1.0)
                gt = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
                full_mask = torch.ones_like(gt[0:1]).bool()
                if "mask" in camera.guidance:
                    full_mask = camera.guidance["mask"].to("cuda").bool()
                obj_mask = camera.guidance.get("obj_bound")
                if obj_mask is None:
                    obj_mask = torch.zeros_like(full_mask)
                else:
                    obj_mask = obj_mask.to("cuda").bool()
                    if obj_mask.ndim == 2:
                        obj_mask = obj_mask[None]
                bg_mask = full_mask & (~obj_mask)
                rows = [
                    _metric_row(torch, loss_utils, lpips_fn, split, idx, camera, image, gt, full_mask, "full_image", not args.skip_lpips, lpips_scopes),
                    _metric_row(torch, loss_utils, lpips_fn, split, idx, camera, image, gt, obj_mask & full_mask, "object_region", not args.skip_lpips, lpips_scopes),
                    _metric_row(torch, loss_utils, lpips_fn, split, idx, camera, image, gt, bg_mask, "background_region", not args.skip_lpips, lpips_scopes),
                ]
                for row in rows:
                    append_csv_row(_scope_csv_path(exp_out, row["scope"]), row, PER_VIEW_FIELDS)
                if idx < args.max_panels_per_split:
                    err = torch.clamp(torch.abs(image - gt) * 4.0, 0.0, 1.0)
                    panel_path = exp_out / "figures" / "final_comparison_panels" / split / f"{camera.image_name}_panel.jpg"
                    _save_panel(panel_path, [("GT RGB", gt), ("Rendered RGB", image), ("RGB Error x4", err)])
                if (idx + 1) % 10 == 0 or idx + 1 == len(cameras):
                    print(f"[FinalEval] progress: {idx + 1}/{len(cameras)} split={split}", flush=True)
            write_current_summaries(exp_out, list(split_cameras.keys()))

    summary_rows = write_current_summaries(exp_out, list(split_cameras.keys()))
    protocol = "full_final_evaluation_test_only" if list(split_cameras.keys()) == ["test"] else "full_final_evaluation"
    manifest = {
        "experiment": Path(config_path).stem,
        "model_path": payload.get("model_path", ""),
        "loaded_iter": int(args.loaded_iter),
        "eval_protocol": protocol,
        "splits": {split: len(cameras) for split, cameras in split_cameras.items()},
        "include_obj": payload.get("model", {}).get("nsg", {}).get("include_obj", ""),
        "include_lpips": not args.skip_lpips,
        "lpips_scopes": sorted(lpips_scopes) if not args.skip_lpips else [],
        "resume": bool(args.resume),
    }
    (exp_out / "final_evaluation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return exp_out, summary_rows, manifest


def _summary_fields_with_experiment():
    return ["experiment", "eval_protocol", *SUMMARY_FIELDS]


def aggregate_existing(final_root):
    by_scope_rows = []
    for path in sorted(final_root.glob("*/summary_by_scope.csv")):
        exp = path.parent.name
        manifest_path = path.parent / "final_evaluation_manifest.json"
        protocol = "full_final_evaluation"
        if manifest_path.exists():
            try:
                protocol = json.loads(manifest_path.read_text(encoding="utf-8")).get("eval_protocol", protocol)
            except Exception:
                pass
        for row in read_csv(path):
            by_scope_rows.append({"experiment": exp, "eval_protocol": protocol, **row})
    main_rows = [
        row for row in by_scope_rows
        if row.get("scope") == "full_image" and row.get("split") == "test"
    ]
    fields = _summary_fields_with_experiment()
    write_csv(final_root / "summary_main.csv", main_rows, fields)
    write_csv(final_root / "summary_by_scope.csv", by_scope_rows, fields)
    print(json.dumps({
        "output_root": str(final_root),
        "experiment_count": len({row["experiment"] for row in by_scope_rows}),
        "summary_main": str(final_root / "summary_main.csv"),
        "summary_by_scope": str(final_root / "summary_by_scope.csv"),
    }, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-root", default="outputs/final_evaluation_test_only_v2")
    parser.add_argument("--loaded-iter", type=int, default=30000)
    parser.add_argument("--max-panels-per-split", type=int, default=12)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--splits", nargs="+", choices=["test", "train"], default=["test"])
    parser.add_argument("--lpips-scopes", nargs="*", choices=SCOPES, default=["full_image"])
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-views-per-split", type=int, default=0)
    parser.add_argument("--single-config-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    final_root = repo_root / args.output_root
    if final_root.exists():
        print(f"[FinalEval] Using existing output root: {final_root}", flush=True)
    final_root.mkdir(parents=True, exist_ok=True)

    if not args.single_config_worker and len(args.configs) > 1:
        for config in args.configs:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--configs", config,
                "--output-root", args.output_root,
                "--loaded-iter", str(args.loaded_iter),
                "--max-panels-per-split", str(args.max_panels_per_split),
                "--splits", *args.splits,
                "--lpips-scopes", *args.lpips_scopes,
                "--max-views-per-split", str(args.max_views_per_split),
                "--single-config-worker",
            ]
            cmd.append("--resume" if args.resume else "--no-resume")
            if args.skip_lpips:
                cmd.append("--skip-lpips")
            if args.overwrite:
                cmd.append("--overwrite")
            ret = subprocess.call(cmd, cwd=str(repo_root))
            if ret != 0:
                raise SystemExit(ret)
        aggregate_existing(final_root)
        return

    by_scope_rows = []
    for config in args.configs:
        exp_dir, summaries, manifest = evaluate_one(repo_root, Path(config).resolve(), final_root, args)
        for row in summaries:
            by_scope_rows.append({"experiment": exp_dir.name, "eval_protocol": manifest["eval_protocol"], **row})
    main_rows = [
        row for row in by_scope_rows
        if row.get("scope") == "full_image" and row.get("split") == "test"
    ]
    fields = _summary_fields_with_experiment()
    write_csv(final_root / "summary_main.csv", main_rows, fields)
    write_csv(final_root / "summary_by_scope.csv", by_scope_rows, fields)
    print(json.dumps({
        "output_root": str(final_root),
        "experiment_count": len(args.configs),
        "summary_main": str(final_root / "summary_main.csv"),
        "summary_by_scope": str(final_root / "summary_by_scope.csv"),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
