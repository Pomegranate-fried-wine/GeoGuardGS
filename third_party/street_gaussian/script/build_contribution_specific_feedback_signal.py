import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build contribution-specific guided feedback signals from CUDA contribution dumps."
    )
    parser.add_argument("--contribution-summary", required=True)
    parser.add_argument("--region-csv", required=True)
    parser.add_argument("--source-signal", default=None)
    parser.add_argument("--random-signal", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bad-ratio-threshold", type=float, default=0.35)
    parser.add_argument("--good-ratio-threshold", type=float, default=0.50)
    parser.add_argument("--min-total-weight", type=float, default=0.02)
    parser.add_argument("--min-pixels-per-region", type=int, default=3)
    parser.add_argument(
        "--soft-contribution",
        action="store_true",
        help="Keep CUDA-ok high-risk pixels as soft contribution-aware bad pixels instead of only hard bad contributors.",
    )
    parser.add_argument("--soft-min-total-weight", type=float, default=0.10)
    parser.add_argument("--soft-max-pixels-per-region", type=int, default=96)
    parser.add_argument("--soft-bad-weight", type=float, default=1.25)
    parser.add_argument("--soft-neutral-weight", type=float, default=0.75)
    parser.add_argument("--hard-bad-extra-weight", type=float, default=2.0)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_regions(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["bbox"] = ast.literal_eval(row["bbox"])
            row["region_key"] = f"{row['view_id']}:region{row['region_id']}"
            rows.append(row)
    return rows


def contributor_ids(signal, label):
    result = defaultdict(set)
    for item in signal.get(f"{label}_contributors", []):
        key = item.get("region_key")
        gid = item.get("gaussian_id", item.get("view_local_index"))
        if key is not None and gid is not None:
            result[key].add(int(gid))
    return result


def build_pixel_signal(summary, signal, args):
    bad_ids_by_region = contributor_ids(signal, "bad")
    good_ids_by_region = contributor_ids(signal, "good")
    pixel_by_view = defaultdict(lambda: {"bad_pixels": [], "good_pixels": [], "regions": []})
    region_records = []

    for frame in summary.get("frames", []):
        if frame.get("status") != "ok":
            continue
        stem = str(frame.get("stem"))
        region_id = str(frame.get("region_id"))
        region_key = f"{stem}:region{region_id}"
        bad_ids = bad_ids_by_region.get(region_key, set())
        good_ids = good_ids_by_region.get(region_key, set())
        if not bad_ids and not good_ids and not args.soft_contribution:
            continue
        npz_path = Path(frame.get("paths", {}).get("npz", ""))
        if not npz_path.exists():
            continue
        npz = np.load(npz_path, allow_pickle=True)
        pixels = np.asarray(npz["selected_pixels"], dtype=np.int64)
        ids = np.asarray(npz["cuda_contribution_ids"], dtype=np.int64)
        weights = np.asarray(npz["contribution_weights"], dtype=np.float32)
        total = np.maximum(weights.sum(axis=1), 1e-8)
        bad_mask = np.isin(ids, list(bad_ids)) if bad_ids else np.zeros_like(ids, dtype=bool)
        good_mask = np.isin(ids, list(good_ids)) if good_ids else np.zeros_like(ids, dtype=bool)
        bad_weight = (weights * bad_mask).sum(axis=1)
        good_weight = (weights * good_mask).sum(axis=1)
        bad_ratio = bad_weight / total
        good_ratio = good_weight / total
        bad_pixels = []
        good_pixels = []
        soft_pixel_candidates = []
        for idx, xy in enumerate(pixels):
            if total[idx] < args.min_total_weight:
                continue
            x, y = int(xy[0]), int(xy[1])
            if bad_ratio[idx] >= args.bad_ratio_threshold and bad_weight[idx] >= good_weight[idx]:
                pixel_weight = args.soft_bad_weight + args.hard_bad_extra_weight * float(bad_ratio[idx])
                bad_pixels.append([x, y, float(pixel_weight), float(bad_weight[idx])])
            elif good_ratio[idx] >= args.good_ratio_threshold and good_weight[idx] > bad_weight[idx]:
                good_pixels.append([x, y, float(good_ratio[idx]), float(good_weight[idx])])
            elif args.soft_contribution and total[idx] >= args.soft_min_total_weight:
                # Soft contribution-aware feedback keeps high-risk DA3 boundary pixels
                # with real rasterizer contribution support, even when the sparse
                # counterfactual label is neutral or unavailable. Hard bad pixels
                # are handled above with stronger weights.
                soft_strength = min(1.0, float(total[idx]))
                soft_pixel_candidates.append([x, y, soft_strength, float(total[idx])])
        if args.soft_contribution and len(bad_pixels) < args.min_pixels_per_region:
            # Keep the most explained high-risk pixels. These are weaker than hard
            # bad counterfactual pixels, but still require nontrivial T*alpha mass.
            soft_pixel_candidates.sort(key=lambda item: item[3], reverse=True)
            for x, y, strength, total_weight in soft_pixel_candidates[: args.soft_max_pixels_per_region]:
                bad_pixels.append([x, y, float(args.soft_neutral_weight * strength), float(total_weight)])
        elif args.soft_contribution:
            # Add a small number of softer context pixels around hard bad evidence.
            soft_pixel_candidates.sort(key=lambda item: item[3], reverse=True)
            for x, y, strength, total_weight in soft_pixel_candidates[: max(0, args.soft_max_pixels_per_region - len(bad_pixels))]:
                bad_pixels.append([x, y, float(args.soft_bad_weight * strength), float(total_weight)])
        evidence_status = "ok"
        if len(bad_pixels) < args.min_pixels_per_region and len(good_pixels) < args.min_pixels_per_region:
            evidence_status = "low_pixel_evidence"
        pixel_by_view[stem]["bad_pixels"].extend(bad_pixels)
        pixel_by_view[stem]["good_pixels"].extend(good_pixels)
        pixel_by_view[stem]["regions"].append(region_key)
        region_records.append(
            {
                "region_key": region_key,
                "view_id": stem,
                "region_id": region_id,
                "region_type": frame.get("region_type"),
                "bbox": frame.get("input_region", {}).get("bbox") or frame.get("pixel_bbox"),
                "bad_pixel_count": len(bad_pixels),
                "good_pixel_count": len(good_pixels),
                "selected_pixel_count": int(frame.get("selected_pixel_count", 0)),
                "mean_bad_ratio": float(np.mean([p[2] for p in bad_pixels])) if bad_pixels else 0.0,
                "mean_good_ratio": float(np.mean([p[2] for p in good_pixels])) if good_pixels else 0.0,
                "evidence_status": evidence_status,
            }
        )

    pixel_feedback_by_view = []
    for view_id, rec in sorted(pixel_by_view.items()):
        # Keep JSON compact: training only needs x/y. Detailed ratios are in CSV.
        pixel_feedback_by_view.append(
            {
                "view_id": view_id,
                "bad_pixels": [[int(p[0]), int(p[1]), float(p[2])] for p in rec["bad_pixels"]],
                "good_pixels": [[int(p[0]), int(p[1]), float(p[2])] for p in rec["good_pixels"]],
                "regions": sorted(set(rec["regions"])),
            }
        )
    return pixel_feedback_by_view, region_records


def build_error_region_only_signal(regions):
    records = []
    bad_contributors = []
    for row in regions:
        key = row["region_key"]
        records.append(
            {
                "region_key": key,
                "view_id": row["view_id"],
                "region_id": row["region_id"],
                "region_type": row.get("region_type"),
                "bbox": row["bbox"],
                "evidence_status": "ok",
                "valid_lidar_high_error_pixels": int(float(row.get("valid_lidar_high_error_pixels", 0))),
                "mean_geometry_error": float(row.get("mean_geometry_error", 0) or 0),
                "mean_depth_error": float(row.get("mean_depth_error", 0) or 0),
            }
        )
        bad_contributors.append(
            {
                "region_key": key,
                "view_id": row["view_id"],
                "region_id": row["region_id"],
                "region_type": row.get("region_type"),
                "gaussian_id": -1,
                "counterfactual_label": "error_region_only",
                "recommended_feedback": "upweight_local_geometry_supervision",
            }
        )
    return {
        "debug_only": False,
        "source": "geometry_error_map top30 regions only; no contribution or counterfactual evidence",
        "feedback_type": "error_region_only",
        "counts": {
            "regions": len(records),
            "bad_contributors": len(bad_contributors),
            "good_contributors": 0,
            "neutral_contributors": 0,
            "low_evidence_regions": 0,
        },
        "regions": records,
        "bad_contributors": bad_contributors,
        "good_contributors": [],
        "neutral_contributors": [],
        "low_evidence_regions": [],
    }


def audit_random_signal(random_signal, regions):
    if not random_signal:
        return {"available": False}
    signal = load_json(random_signal)
    region_lookup = {row["region_key"]: row for row in regions}
    records = []
    for item in signal.get("bad_contributors", []):
        key = item.get("region_key")
        row = region_lookup.get(key)
        if row:
            records.append(row)
    if records:
        vals = [int(float(r.get("valid_lidar_high_error_pixels", 0))) for r in records]
        errs = [float(r.get("mean_depth_error", 0) or 0) for r in records]
        source = "matched_high_error_region_keys"
    else:
        # Random-region signal was intentionally generated with random LiDAR-valid
        # centers and same bbox size distribution, so it is not expected to match
        # the mined high-error region ids.
        vals, errs, source = [], [], "random_lidar_valid_pixels_with_top30_bbox_size_distribution"
    return {
        "available": True,
        "signal_path": str(random_signal),
        "matched_mined_high_error_region_count": len(records),
        "sampling_interpretation": source,
        "matched_valid_lidar_high_error_pixels_mean": float(np.mean(vals)) if vals else None,
        "matched_mean_depth_error_mean": float(np.mean(errs)) if errs else None,
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = load_json(args.contribution_summary)
    if args.source_signal:
        source_signal = load_json(args.source_signal)
    else:
        summary_path = Path(args.contribution_summary)
        preferred = summary_path.with_name("guided_training_feedback_signal.json")
        legacy = summary_path.with_name("guided_training_feedback_signal_75view_top30.json")
        source_signal = load_json(preferred if preferred.exists() else legacy)
    regions = load_regions(args.region_csv)

    pixel_feedback_by_view, region_records = build_pixel_signal(summary, source_signal, args)
    signal = dict(source_signal)
    signal["debug_only"] = False
    signal["feedback_type"] = "contribution_specific_pixel_feedback"
    signal["selected_pixel_source"] = "da3_boundary_risk_map"
    signal["uses_lidar_selected_pixels"] = False
    signal["training_feedback_supervision"] = "da3_unsupervised"
    signal["soft_patch_note"] = (
        "Training expands selected DA3 boundary-risk contribution pixels into local soft patches "
        "via train.guided_feedback.pixel_radius; LiDAR is not used for feedback selection or loss."
    )
    signal["pixel_feedback_by_view"] = pixel_feedback_by_view
    signal["contribution_specific_config"] = {
        "bad_ratio_threshold": args.bad_ratio_threshold,
        "good_ratio_threshold": args.good_ratio_threshold,
        "min_total_weight": args.min_total_weight,
        "min_pixels_per_region": args.min_pixels_per_region,
        "soft_contribution": args.soft_contribution,
        "soft_min_total_weight": args.soft_min_total_weight,
        "soft_max_pixels_per_region": args.soft_max_pixels_per_region,
        "soft_bad_weight": args.soft_bad_weight,
        "soft_neutral_weight": args.soft_neutral_weight,
        "hard_bad_extra_weight": args.hard_bad_extra_weight,
    }
    signal["counts"] = dict(signal.get("counts", {}))
    signal["counts"]["bad_feedback_pixels"] = int(sum(len(v["bad_pixels"]) for v in pixel_feedback_by_view))
    signal["counts"]["good_feedback_pixels"] = int(sum(len(v["good_pixels"]) for v in pixel_feedback_by_view))
    signal["counts"]["pixel_feedback_views"] = len(pixel_feedback_by_view)
    write_json(out_dir / "guided_training_feedback_signal_contribution_specific_top30.json", signal)
    write_json(out_dir / "da3_contribution_specific_feedback_signal.json", signal)
    if args.soft_contribution:
        write_json(out_dir / "guided_training_feedback_signal_soft_contribution_top30.json", signal)
        write_json(out_dir / "da3_contribution_softpatch_feedback_signal.json", signal)

    error_signal = build_error_region_only_signal(regions)
    write_json(out_dir / "guided_training_feedback_signal_error_region_only_top30.json", error_signal)

    audit = {
        "random_region_audit": audit_random_signal(Path(args.random_signal) if args.random_signal else None, regions),
        "contribution_specific_counts": signal["counts"],
        "region_pixel_feedback_summary": {
            "regions": len(region_records),
            "regions_with_bad_pixels": int(sum(1 for r in region_records if r["bad_pixel_count"] > 0)),
            "regions_with_good_pixels": int(sum(1 for r in region_records if r["good_pixel_count"] > 0)),
            "low_pixel_evidence_regions": int(sum(1 for r in region_records if r["evidence_status"] != "ok")),
        },
    }
    write_json(out_dir / "feedback_signal_audit_summary.json", audit)

    with open(out_dir / "contribution_specific_region_pixels.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "region_key",
            "view_id",
            "region_id",
            "region_type",
            "bbox",
            "bad_pixel_count",
            "good_pixel_count",
            "selected_pixel_count",
            "mean_bad_ratio",
            "mean_good_ratio",
            "evidence_status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(region_records)

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
