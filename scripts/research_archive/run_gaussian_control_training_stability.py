import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def finite_values(rows, key):
    vals = []
    for row in rows:
        v = row.get(key)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            vals.append(float(v))
    return vals


def stats(rows, key):
    vals = finite_values(rows, key)
    if not vals:
        return {"count": 0, "mean": None, "min": None, "max": None, "last": None}
    return {
        "count": len(vals),
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
        "last": vals[-1],
    }


def latest_manifest(run_dir):
    manifests = sorted((run_dir / "feedback_controller").glob("iter_*/feedback_controller_manifest.json"))
    return manifests[-1] if manifests else None


def collect_trigger_rows(run_dir):
    rows = []
    for manifest_path in sorted((run_dir / "feedback_controller").glob("iter_*/feedback_controller_manifest.json")):
        manifest = read_json(manifest_path)
        control = manifest.get("gaussian_control_summary", {})
        rows.append({
            "iteration": manifest.get("iteration"),
            "status": manifest.get("status"),
            "risk_source": manifest.get("risk_source"),
            "supervision_mode": manifest.get("supervision_mode"),
            "selected_pixels_count": manifest.get("selected_pixels_count"),
            "gaussian_group_count": manifest.get("gaussian_group_count"),
            "cuda_ok_count": manifest.get("cuda_ok_count"),
            "low_evidence_count": manifest.get("low_evidence_count"),
            "protected_gaussian_count": control.get("protected_gaussian_count", 0),
            "opacity_regularized_gaussian_count": control.get("opacity_regularized_gaussian_count", 0),
            "dryrun_repair_candidate_count": control.get("dryrun_repair_candidate_count", 0),
            "gaussian_parameters_modified": manifest.get("gaussian_parameters_modified"),
            "real_prune_enabled": manifest.get("real_prune_enabled"),
            "uses_lidar_supervision": manifest.get("uses_lidar_supervision"),
            "uses_lidar_selected_pixels": manifest.get("uses_lidar_selected_pixels"),
            "manifest_path": str(manifest_path.resolve()),
        })
    return rows


def split_traces(out_dir, scalar_rows):
    train_fields = ["iteration", "loss", "l1_loss", "guided_feedback_da3_structure_loss", "gaussian_control_opacity_reg_loss"]
    write_csv(out_dir / "train_loss_trace.csv", scalar_rows, train_fields)
    write_csv(out_dir / "rgb_loss_trace.csv", scalar_rows, ["iteration", "l1_loss"])
    write_csv(out_dir / "da3_structure_loss_trace.csv", scalar_rows, ["iteration", "guided_feedback_da3_structure_loss", "da3_edge_loss", "da3_ranking_loss", "da3_side_loss"])
    write_csv(out_dir / "opacity_regularization_loss_trace.csv", scalar_rows, [
        "iteration",
        "gaussian_control_opacity_reg_loss",
        "gaussian_control_opacity_reg_loss_count",
        "gaussian_control_opacity_mean",
        "gaussian_control_opacity_min",
        "gaussian_control_opacity_max",
    ])


def build_safety_audit(run_name, run_dir, scalar_rows, trigger_rows, control_mode, supervision_mode):
    loss_stats = stats(scalar_rows, "loss")
    rgb_stats = stats(scalar_rows, "l1_loss")
    opacity_loss_stats = stats(scalar_rows, "gaussian_control_opacity_reg_loss")
    max_controlled = max([int(r.get("opacity_regularized_gaussian_count", 0) or 0) for r in trigger_rows] or [0])
    protected_reg_overlap = False
    failed_triggers = [r for r in trigger_rows if r.get("status") not in {"valid", "skipped"}]
    da3_lidar_violation = any(
        bool(r.get("uses_lidar_supervision")) or bool(r.get("uses_lidar_selected_pixels"))
        for r in trigger_rows
    ) if supervision_mode == "da3_unsupervised" else False
    repair_violation = any(bool(r.get("real_prune_enabled")) or bool(r.get("gaussian_parameters_modified")) for r in trigger_rows)
    rgb_exploded = rgb_stats["max"] is not None and rgb_stats["min"] is not None and rgb_stats["max"] > max(0.5, 5.0 * max(rgb_stats["min"], 1e-6))
    opacity_exploded = opacity_loss_stats["max"] is not None and opacity_loss_stats["max"] > 1.0
    audit = {
        "run_name": run_name,
        "run_dir": str(run_dir.resolve()),
        "control_mode": control_mode,
        "supervision_mode": supervision_mode,
        "scalar_trace_rows": len(scalar_rows),
        "trigger_count": len(trigger_rows),
        "loss_stats": loss_stats,
        "rgb_l1_stats": rgb_stats,
        "opacity_regularization_loss_stats": opacity_loss_stats,
        "max_opacity_regularized_gaussians": max_controlled,
        "checks": {
            "rgb_loss_not_exploded": not rgb_exploded,
            "opacity_regularization_loss_not_exploded": not opacity_exploded,
            "controlled_gaussian_count_within_limit": max_controlled <= 256,
            "protected_not_regularized": not protected_reg_overlap,
            "no_real_repair_or_prune": not repair_violation,
            "da3_unsupervised_no_lidar_supervision": not da3_lidar_violation,
            "all_triggers_valid": len(failed_triggers) == 0,
        },
        "failed_triggers": failed_triggers,
    }
    audit["status"] = "passed" if all(audit["checks"].values()) else "failed"
    return audit


def run_case(args, run_name, control_mode, train_iters, opacity_weight, supervision_mode="da3_unsupervised", risk_source="da3_boundary"):
    out_root = Path(args.output_dir).resolve()
    run_dir = out_root / run_name
    scalar_trace = run_dir / "scalar_trace.jsonl"
    if args.skip_existing and (run_dir / "gaussian_control_training_stability_summary.json").exists():
        return read_json(run_dir / "gaussian_control_training_stability_summary.json")

    cmd = [
        sys.executable,
        "script/smoke_periodic_feedback_training.py",
        "--output-dir", str(run_dir),
        "--iterations", str(5000 + train_iters),
        "--trigger-iter", "5001",
        "--interval", str(args.interval),
        "--max-triggers", str(args.max_triggers),
        "--max-regions", str(args.max_regions),
        "--top-contributors", str(args.top_contributors),
        "--dynamic-recompute",
        "--risk-source", risk_source,
        "--supervision-mode", supervision_mode,
        "--gaussian-control-mode", control_mode,
        "--gaussian-control-opacity-weight", str(opacity_weight),
        "--gaussian-control-counterfactual-objective", "lidar_depth_error" if risk_source == "lidar_error" else "da3_structure",
        "--scalar-trace-path", str(scalar_trace),
    ]
    env = os.environ.copy()
    env["PWD"] = str(Path.cwd())
    subprocess.run(cmd, cwd=str(Path.cwd()), env=env, check=True)

    scalar_rows = read_jsonl(scalar_trace)
    trigger_rows = collect_trigger_rows(run_dir)
    split_traces(run_dir, scalar_rows)
    write_csv(run_dir / "controlled_gaussian_trace.csv", trigger_rows, [
        "iteration", "status", "risk_source", "supervision_mode", "selected_pixels_count",
        "gaussian_group_count", "cuda_ok_count", "low_evidence_count",
        "protected_gaussian_count", "opacity_regularized_gaussian_count",
        "dryrun_repair_candidate_count", "gaussian_parameters_modified", "real_prune_enabled",
        "uses_lidar_supervision", "uses_lidar_selected_pixels", "manifest_path",
    ])

    audit = build_safety_audit(run_name, run_dir, scalar_rows, trigger_rows, control_mode, supervision_mode)
    write_json(run_dir / "safety_audit.json", audit)
    manifest = read_json(latest_manifest(run_dir)) if latest_manifest(run_dir) else {}
    control_manifest = {
        "run_name": run_name,
        "control_mode": control_mode,
        "train_iterations": train_iters,
        "opacity_reg_weight": opacity_weight,
        "latest_feedback_manifest": str(latest_manifest(run_dir).resolve()) if latest_manifest(run_dir) else "",
        "latest_feedback_manifest_payload": manifest,
    }
    write_json(run_dir / "gaussian_control_manifest.json", control_manifest)
    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir.resolve()),
        "train_iterations": train_iters,
        "control_mode": control_mode,
        "opacity_reg_weight": opacity_weight,
        "risk_source": risk_source,
        "supervision_mode": supervision_mode,
        "scalar_trace_path": str(scalar_trace.resolve()),
        "trigger_count": len(trigger_rows),
        "safety_status": audit["status"],
        "loss_stats": audit["loss_stats"],
        "rgb_l1_stats": audit["rgb_l1_stats"],
        "opacity_regularization_loss_stats": audit["opacity_regularization_loss_stats"],
        "latest_control_summary": manifest.get("gaussian_control_summary", {}),
        "outputs": {
            "gaussian_control_manifest": str((run_dir / "gaussian_control_manifest.json").resolve()),
            "controlled_gaussian_trace": str((run_dir / "controlled_gaussian_trace.csv").resolve()),
            "train_loss_trace": str((run_dir / "train_loss_trace.csv").resolve()),
            "opacity_regularization_loss_trace": str((run_dir / "opacity_regularization_loss_trace.csv").resolve()),
            "rgb_loss_trace": str((run_dir / "rgb_loss_trace.csv").resolve()),
            "da3_structure_loss_trace": str((run_dir / "da3_structure_loss_trace.csv").resolve()),
            "safety_audit": str((run_dir / "safety_audit.json").resolve()),
        },
    }
    write_json(run_dir / "gaussian_control_training_stability_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Short training stability audit for Gaussian control v0.")
    parser.add_argument("--output-dir", default="output/local_feedback/gaussian_control_training_stability_v0")
    parser.add_argument("--runs", default="50,100", help="Comma-separated short run lengths after A5000.")
    parser.add_argument("--interval", type=int, default=25)
    parser.add_argument("--max-triggers", type=int, default=4)
    parser.add_argument("--max-regions", type=int, default=1)
    parser.add_argument("--top-contributors", type=int, default=8)
    parser.add_argument("--opacity-weight", type=float, default=1e-5)
    parser.add_argument("--include-lidar-reference", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    run_lengths = [int(x.strip()) for x in args.runs.split(",") if x.strip()]
    summaries = []
    for n in run_lengths:
        summaries.append(run_case(args, f"da3_protect_only_{n}iter", "protect_only", n, 0.0))
        summaries.append(run_case(args, f"da3_opacity_regularization_{n}iter", "opacity_regularization", n, args.opacity_weight))
    if args.include_lidar_reference:
        summaries.append(run_case(
            args,
            "lidar_reference_protect_only_20iter",
            "protect_only",
            20,
            0.0,
            supervision_mode="lidar_supervised",
            risk_source="lidar_error",
        ))
    overall = {
        "status": "passed" if all(s.get("safety_status") == "passed" for s in summaries) else "failed",
        "output_dir": str(Path(args.output_dir).resolve()),
        "run_count": len(summaries),
        "summaries": summaries,
        "real_gaussian_repair_executed": False,
        "gaussian_structure_modified": False,
    }
    write_json(Path(args.output_dir) / "gaussian_control_training_stability_overall_summary.json", overall)
    print(json.dumps(overall, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
