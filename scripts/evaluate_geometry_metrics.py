#!/usr/bin/env python3
"""Wrapper entry for geometry metrics.

For full LiDAR-valid evaluation, use geoguardgs/evaluation/lidar_geometry_metrics.py
or the legacy StreetGS evaluate_lidar_depth.py script copied from the research
workspace.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-script", default="geoguardgs/evaluation/lidar_geometry_metrics.py")
    parser.add_argument("args", nargs="*")
    ns = parser.parse_args()
    script = Path(ns.legacy_script)
    if not script.exists():
        raise SystemExit(f"Missing evaluation script: {script}")
    raise SystemExit(subprocess.call([sys.executable, str(script)] + ns.args))


if __name__ == "__main__":
    main()
