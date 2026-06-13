import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def _read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _run_case(name, out_root, base_args, extra_args):
    out_dir = out_root / name
    cmd = [
        sys.executable,
        "script/smoke_periodic_feedback_training.py",
        "--output-dir",
        str(out_dir),
        "--iterations",
        str(base_args.iterations),
        "--trigger-iter",
        str(base_args.trigger_iter),
        "--interval",
        str(base_args.interval),
        "--max-regions",
        str(base_args.max_regions),
        "--top-contributors",
        str(base_args.top_contributors),
        "--dynamic-recompute",
    ] + extra_args
    env = os.environ.copy()
    env["PWD"] = str(Path.cwd())
    subprocess.run(cmd, cwd=str(Path.cwd()), env=env, check=True)
    manifest = out_dir / "feedback_controller" / f"iter_{base_args.trigger_iter:06d}" / "feedback_controller_manifest.json"
    payload = _read_json(manifest) if manifest.exists() else {"status": "missing_manifest"}
    return {
        "case": name,
        "output_dir": str(out_dir),
        "manifest_path": str(manifest),
        "status": payload.get("status"),
        "risk_source": payload.get("risk_source"),
        "supervision_mode": payload.get("supervision_mode"),
        "gaussian_parameters_modified": payload.get("gaussian_parameters_modified"),
        "real_prune_enabled": payload.get("real_prune_enabled"),
        "uses_lidar_supervision": payload.get("uses_lidar_supervision"),
        "uses_lidar_selected_pixels": payload.get("uses_lidar_selected_pixels"),
        "live_cuda_contribution": payload.get("live_cuda_contribution"),
        "uses_cached_contribution": payload.get("uses_cached_contribution"),
        "gaussian_control_summary": payload.get("gaussian_control_summary", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="Smoke unified Gaussian control for DA3 and LiDAR branches.")
    parser.add_argument("--output-dir", default="output/local_feedback/gaussian_control_unified_smoke")
    parser.add_argument("--iterations", type=int, default=5002)
    parser.add_argument("--trigger-iter", type=int, default=5001)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--max-regions", type=int, default=1)
    parser.add_argument("--top-contributors", type=int, default=8)
    parser.add_argument("--skip-lidar", action="store_true")
    args = parser.parse_args()

    out_root = _resolve(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cases = []

    cases.append(_run_case(
        "da3_protect_only",
        out_root,
        args,
        [
            "--risk-source", "da3_boundary",
            "--supervision-mode", "da3_unsupervised",
            "--gaussian-control-mode", "protect_only",
            "--gaussian-control-counterfactual-objective", "da3_structure",
        ],
    ))
    cases.append(_run_case(
        "da3_opacity_regularization",
        out_root,
        args,
        [
            "--risk-source", "da3_boundary",
            "--supervision-mode", "da3_unsupervised",
            "--gaussian-control-mode", "opacity_regularization",
            "--gaussian-control-opacity-weight", "0.001",
            "--gaussian-control-counterfactual-objective", "da3_structure",
        ],
    ))
    if not args.skip_lidar:
        cases.append(_run_case(
            "lidar_reference_protect_only",
            out_root,
            args,
            [
                "--risk-source", "lidar_error",
                "--supervision-mode", "lidar_supervised",
                "--gaussian-control-mode", "protect_only",
                "--gaussian-control-counterfactual-objective", "lidar_depth_error",
            ],
        ))

    summary = {
        "status": "valid",
        "output_dir": str(out_root),
        "case_count": len(cases),
        "cases": cases,
        "real_gaussian_repair_executed": False,
        "gaussian_parameters_modified": False,
    }
    _write_json(out_root / "gaussian_control_unified_smoke_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
