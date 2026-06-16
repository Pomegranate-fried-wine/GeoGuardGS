#!/usr/bin/env python3
"""Verify that a GeoFeedback-GS checkout is ready for GitHub submission/migration."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    ".gitignore",
    "requirements.txt",
    "environment.yml",
    "pyproject.toml",
    "configs/base/geoguardgs_base.yaml",
    "configs/experiments/a100_baseline_streetgs.yaml",
    "configs/experiments/a100_da3_periodic_group_softpatch.yaml",
    "docs/server_a100_experiment_guide.md",
    "docs/code_audit_report.md",
    "scripts/train.py",
    "scripts/launch_a100_experiments.py",
    "scripts/check_closed_loop_config.py",
    "scripts/install_server_extensions.sh",
    "scripts/check_imports.py",
    "third_party/street_gaussian/train.py",
    "third_party/street_gaussian/lib/config/config.py",
    "third_party/diff_gaussian_rasterization/setup.py",
    "third_party/simple_knn/setup.py",
    "third_party/depth_anything_3",
    "data/waymo/.gitkeep",
    "weights/da3/.gitkeep",
    "weights/streetgs/.gitkeep",
    "outputs/.gitkeep",
]

FORBIDDEN_EXTS = {
    ".pth",
    ".pt",
    ".ckpt",
    ".pkl",
    ".npz",
    ".npy",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".log",
    ".pyd",
    ".dll",
    ".so",
    ".lib",
    ".exp",
    ".obj",
    ".pyc",
}

FORBIDDEN_DIR_NAMES = {"__pycache__", "build", "dist", ".cache", ".pytest_cache"}


def main():
    missing = [p for p in REQUIRED_PATHS if not (ROOT / p).exists()]
    forbidden_files = [
        str(p.relative_to(ROOT))
        for p in ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in FORBIDDEN_EXTS
    ]
    forbidden_dirs = [
        str(p.relative_to(ROOT))
        for p in ROOT.rglob("*")
        if p.is_dir() and (p.name in FORBIDDEN_DIR_NAMES or p.name.endswith(".egg-info"))
    ]
    payload = {
        "root": str(ROOT),
        "status": "passed" if not missing and not forbidden_files and not forbidden_dirs else "failed",
        "missing_required_paths": missing,
        "forbidden_files": forbidden_files[:200],
        "forbidden_dirs": forbidden_dirs[:200],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
