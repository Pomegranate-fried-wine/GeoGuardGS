#!/usr/bin/env python3
"""Smoke-test depth visualization on empty and invalid depth maps."""

import sys
from pathlib import Path

import numpy as np


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "third_party" / "street_gaussian"))
    from lib.utils.img_utils import visualize_depth_numpy

    cases = {
        "all_zero": np.zeros((8, 8), dtype=np.float32),
        "all_negative": -np.ones((8, 8), dtype=np.float32),
        "nan_inf": np.array([[np.nan, np.inf], [-np.inf, 0.0]], dtype=np.float32),
        "positive": np.linspace(1.0, 10.0, 64, dtype=np.float32).reshape(8, 8),
    }
    for name, depth in cases.items():
        image, minmax = visualize_depth_numpy(depth)
        assert image.shape[:2] == depth.shape[:2], name
        assert image.shape[-1] == 3, name
        assert image.dtype == np.uint8, name
        assert len(minmax) == 2, name
        print(f"[OK] {name}: minmax={minmax}")


if __name__ == "__main__":
    main()
