#!/usr/bin/env python3
"""Unified evaluation launcher placeholder for GeoFeedback-GS."""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["geometry"], default="geometry")
    parser.add_argument("args", nargs="*")
    ns = parser.parse_args()
    if ns.mode == "geometry":
        return subprocess.call([sys.executable, "scripts/evaluate_geometry_metrics.py"] + ns.args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
