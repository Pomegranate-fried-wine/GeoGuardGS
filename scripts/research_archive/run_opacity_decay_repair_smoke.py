import argparse
import csv
import json
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


def read_csv(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def summarize_scalar(rows, key):
    vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return {"count": 0, "mean": None, "min": None, "max": None, "last": None}
    return {"count": len(vals), "mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals), "last": vals[-1]}


def main():
    parser = argparse.ArgumentParser(description="Run minimal real opacity decay repair smoke from A5000 copy.")
    parser.add_argument("--output-dir", default="output/local_feedback/gaussian_repair_v1_opacity_decay_smoke")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--interval", type=int, default=25)
    parser.add_argument("--max-triggers", type=int, default=2)
    parser.add_argument("--max-regions", type=int, default=1)
    parser.add_argument("--top-contributors", type=int, default=8)
    parser.add_argument("--decay-factor", type=float, default=0.95)
    parser.add_argument("--max-decay", type=int, default=10)
    parser.add_argument("--max-decay-ratio", type=float, default=0.00005)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    scalar_trace = out / "scalar_trace.jsonl"
    summary_path = out / "gaussian_repair_v1_opacity_decay_smoke_summary.json"
    if args.skip_existing and summary_path.exists():
        print(summary_path.read_text(encoding="utf-8-sig"))
        return

    cmd = [
        sys.executable,
        "script/smoke_periodic_feedback_training.py",
        "--output-dir", str(out),
        "--iterations", str(5000 + args.iterations),
        "--trigger-iter", "5001",
        "--interval", str(args.interval),
        "--max-triggers", str(args.max_triggers),
        "--max-regions", str(args.max_regions),
        "--top-contributors", str(args.top_contributors),
        "--dynamic-recompute",
        "--risk-source", "da3_boundary",
        "--supervision-mode", "da3_unsupervised",
        "--gaussian-control-mode", "opacity_decay_apply",
        "--gaussian-control-counterfactual-objective", "da3_structure",
        "--gaussian-control-allow-parameter-modification",
        "--gaussian-control-opacity-decay-factor", str(args.decay_factor),
        "--gaussian-control-max-decay", str(args.max_decay),
        "--gaussian-control-max-decay-ratio", str(args.max_decay_ratio),
        "--scalar-trace-path", str(scalar_trace),
    ]
    env = os.environ.copy()
    env["PWD"] = str(Path.cwd())
    subprocess.run(cmd, cwd=str(Path.cwd()), env=env, check=True)

    manifests = sorted(out.glob("feedback_controller/iter_*/feedback_controller_manifest.json"))
    decay_manifests = sorted(out.glob("feedback_controller/iter_*/opacity_decay_apply/opacity_decay_apply_manifest.json"))
    all_decay_rows = []
    all_skipped_rows = []
    all_light = []
    for manifest_path in decay_manifests:
        decay_dir = manifest_path.parent
        all_decay_rows += read_csv(decay_dir / "opacity_decay_gaussians.csv")
        all_skipped_rows += read_csv(decay_dir / "skipped_gaussians.csv")
        light_path = decay_dir / "opacity_decay_light_eval.json"
        if light_path.exists():
            all_light.append(read_json(light_path))
    write_csv(out / "controlled_gaussian_opacity_trace.csv", all_decay_rows)
    write_csv(out / "skipped_gaussians_all.csv", all_skipped_rows)
    scalar_rows = read_jsonl(scalar_trace)
    safety = {
        "status": "passed",
        "gaussian_parameters_modified": bool(all_decay_rows),
        "real_prune_enabled": False,
        "real_split_enabled": False,
        "real_shrink_enabled": False,
        "modified_gaussian_count": len(all_decay_rows),
        "trigger_count": len(manifests),
        "decay_trigger_count": len(decay_manifests),
        "checks": {
            "has_real_opacity_modification": len(all_decay_rows) > 0,
            "no_prune_shrink_split": True,
            "all_decay_factors_valid": all(float(r.get("decay_factor", 0.0)) == args.decay_factor for r in all_decay_rows),
            "all_opacity_decreased": all(float(r.get("new_opacity", 1.0)) < float(r.get("old_opacity", 0.0)) for r in all_decay_rows),
            "all_modified_not_protected": all(str(r.get("is_protected", "False")).lower() in {"false", "0"} for r in all_decay_rows),
            "all_modified_not_low_evidence": all(str(r.get("is_low_evidence", "False")).lower() in {"false", "0"} for r in all_decay_rows),
            "da3_unsupervised_no_lidar_selected_pixels": all(not read_json(m).get("uses_lidar_selected_pixels", False) for m in manifests),
            "da3_unsupervised_no_lidar_supervision": all(not read_json(m).get("uses_lidar_supervision", False) for m in manifests),
        },
        "rgb_l1_stats": summarize_scalar(scalar_rows, "l1_loss"),
        "loss_stats": summarize_scalar(scalar_rows, "loss"),
        "opacity_decay_modified_by_trigger": [
            {
                "path": str(p.resolve()),
                "modified_count": read_json(p).get("modified_count"),
                "light_eval": read_json(p).get("light_eval", {}),
            }
            for p in decay_manifests
        ],
    }
    if not all(safety["checks"].values()):
        safety["status"] = "failed"
    write_json(out / "gaussian_repair_safety_audit.json", safety)
    summary = {
        "status": safety["status"],
        "output_dir": str(out),
        "iterations": args.iterations,
        "decay_factor": args.decay_factor,
        "max_decay_gaussians_per_trigger": args.max_decay,
        "max_decay_ratio": args.max_decay_ratio,
        "manifest_count": len(manifests),
        "decay_manifest_count": len(decay_manifests),
        "modified_gaussian_count": len(all_decay_rows),
        "rgb_l1_stats": safety["rgb_l1_stats"],
        "loss_stats": safety["loss_stats"],
        "light_eval_count": len(all_light),
        "light_eval": all_light,
        "files": {
            "summary": str(summary_path),
            "safety_audit": str(out / "gaussian_repair_safety_audit.json"),
            "controlled_gaussian_opacity_trace": str(out / "controlled_gaussian_opacity_trace.csv"),
            "skipped_gaussians": str(out / "skipped_gaussians_all.csv"),
            "scalar_trace": str(scalar_trace),
        },
        "real_prune_enabled": False,
        "real_split_enabled": False,
        "real_shrink_enabled": False,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
