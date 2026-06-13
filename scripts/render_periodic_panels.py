#!/usr/bin/env python3
"""Placeholder panel renderer for periodic outputs.

This script documents the expected interface. It can be extended to compose
rendered RGB, GT RGB, RGB error, depth, risk maps, and contribution overlays.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration-dir", required=True)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()
    iteration_dir = Path(args.iteration_dir)
    out_dir = Path(args.out_dir) if args.out_dir else iteration_dir / "panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    readme = out_dir / "README.txt"
    readme.write_text(
        "Panel renderer placeholder. Expected inputs: rgb/, depth/, risk_maps/, contribution/, gaussian_control/.\n",
        encoding="utf-8",
    )
    print(f"Prepared panel directory: {out_dir}")


if __name__ == "__main__":
    main()
