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


def main():
    parser = argparse.ArgumentParser(description="Run periodic closed-loop repair dry-run smoke.")
    parser.add_argument("--output-dir", default="output/local_feedback/gaussian_repair_operator_dryrun_smoke")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--interval", type=int, default=25)
    parser.add_argument("--max-triggers", type=int, default=2)
    parser.add_argument("--max-regions", type=int, default=1)
    parser.add_argument("--top-contributors", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    summary_path = out / "repair_operator_dryrun_smoke_summary.json"
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
        "--gaussian-control-mode", "repair_dryrun",
        "--gaussian-control-counterfactual-objective", "da3_structure",
        "--scalar-trace-path", str(out / "scalar_trace.jsonl"),
    ]
    env = os.environ.copy()
    env["PWD"] = str(Path.cwd())
    subprocess.run(cmd, cwd=str(Path.cwd()), env=env, check=True)

    manifests = sorted(out.glob("feedback_controller/iter_*/feedback_controller_manifest.json"))
    repair_manifests = sorted(out.glob("feedback_controller/iter_*/gaussian_repair_operator/repair_operator_manifest.json"))
    all_candidates = []
    all_skipped = []
    op_counts = {}
    for path in repair_manifests:
        payload = read_json(path)
        for k, v in payload.get("operation_counts", {}).items():
            op_counts[k] = op_counts.get(k, 0) + int(v)
        all_candidates += read_csv(path.parent / "repair_dryrun_candidates.csv")
        all_skipped += read_csv(path.parent / "repair_dryrun_skipped.csv")
    write_csv(out / "repair_dryrun_candidates_all.csv", all_candidates)
    write_csv(out / "repair_dryrun_skipped_all.csv", all_skipped)

    manifest_payloads = [read_json(p) for p in manifests]
    safety = {
        "status": "passed",
        "trigger_count": len(manifests),
        "repair_manifest_count": len(repair_manifests),
        "candidate_count": len(all_candidates),
        "operation_counts": op_counts,
        "checks": {
            "no_gaussian_parameter_modification": all(not p.get("gaussian_control_summary", {}).get("gaussian_parameters_modified", False) for p in manifest_payloads),
            "no_real_prune": all(not p.get("real_prune_enabled", False) for p in manifest_payloads),
            "no_lidar_supervision_in_da3_branch": all(not p.get("uses_lidar_supervision", False) for p in manifest_payloads),
            "no_lidar_selected_pixels_in_da3_branch": all(not p.get("uses_lidar_selected_pixels", False) for p in manifest_payloads),
            "repair_manifests_exist": len(repair_manifests) == len(manifests),
            "candidate_table_generated": len(all_candidates) > 0,
        },
        "gaussian_parameters_modified": False,
        "real_prune_enabled": False,
        "real_split_enabled": False,
        "real_shrink_enabled": False,
    }
    if not all(safety["checks"].values()):
        safety["status"] = "failed"
    write_json(out / "repair_operator_dryrun_safety_audit.json", safety)
    summary = {
        "status": safety["status"],
        "output_dir": str(out),
        "iterations": args.iterations,
        "manifest_count": len(manifests),
        "repair_manifest_count": len(repair_manifests),
        "candidate_count": len(all_candidates),
        "operation_counts": op_counts,
        "files": {
            "summary": str(summary_path),
            "safety_audit": str(out / "repair_operator_dryrun_safety_audit.json"),
            "candidates": str(out / "repair_dryrun_candidates_all.csv"),
            "skipped": str(out / "repair_dryrun_skipped_all.csv"),
        },
        "gaussian_parameters_modified": False,
        "real_prune_enabled": False,
        "real_split_enabled": False,
        "real_shrink_enabled": False,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
