#!/usr/bin/env python3
"""Check whether COLMAP can be used for image-only initialization."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml


LIBRARY_ERROR_TOKENS = (
    "libfreeimage",
    "libtiff",
    "TIFFFieldDataType",
    "symbol lookup error",
)


def deep_merge(base, child):
    out = dict(base or {})
    for key, value in (child or {}).items():
        if key == "_BASE_":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.get("_BASE_")
    if not base:
        return cfg
    base_path = Path(base)
    if not base_path.is_absolute():
        base_path = (path.parent / base_path).resolve()
    return deep_merge(load_config(base_path), cfg)


def resolve_colmap_binary(config_path="", explicit_bin=""):
    if explicit_bin:
        return explicit_bin, "argument"
    if config_path:
        cfg = load_config(config_path)
        configured = ((cfg.get("data") or {}).get("colmap_executable") or "").strip()
        if configured:
            return configured, "data.colmap_executable"
    env_bin = os.environ.get("COLMAP_BIN", "").strip()
    if env_bin:
        return env_bin, "COLMAP_BIN"
    path_bin = shutil.which("colmap")
    if path_bin:
        return path_bin, "PATH"
    return "colmap", "PATH"


def run_command(cmd):
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "command": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "library_issue": False,
        }
    output = "\n".join(part for part in [proc.stdout, proc.stderr] if part)
    return {
        "ok": proc.returncode == 0,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "library_issue": any(token in output for token in LIBRARY_ERROR_TOKENS),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", help="Optional GeoFeedback-GS yaml config.")
    parser.add_argument("--colmap-bin", default="", help="Override COLMAP executable.")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--check-gui", action="store_true", help="Also run 'colmap gui -h'.")
    args = parser.parse_args()

    colmap_bin, source = resolve_colmap_binary(args.config, args.colmap_bin)
    checks = [run_command([colmap_bin, "-h"])]
    if args.check_gui:
        checks.append(run_command([colmap_bin, "gui", "-h"]))

    ok = all(check["ok"] for check in checks)
    library_issue = any(check["library_issue"] for check in checks)
    payload = {
        "status": "passed" if ok else "failed",
        "colmap_binary": colmap_bin,
        "binary_source": source,
        "config": args.config,
        "library_issue": library_issue,
        "checks": checks,
    }
    if not ok:
        if library_issue:
            payload["error"] = (
                "COLMAP failed with a likely libfreeimage/libtiff dynamic library mismatch. "
                "Set data.colmap_executable or COLMAP_BIN to a working conda/local COLMAP binary."
            )
        else:
            payload["error"] = (
                "COLMAP is not usable from this environment. Install COLMAP or set "
                "data.colmap_executable / COLMAP_BIN to a working binary."
            )

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
