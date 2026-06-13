#!/usr/bin/env python3
"""Minimal import check after server migration."""

import importlib


MODULES = [
    "torch",
    "numpy",
    "cv2",
    "diff_gaussian_rasterization",
    "simple_knn",
]


def main():
    failed = []
    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"[OK] {name}")
        except Exception as exc:
            failed.append((name, str(exc)))
            print(f"[FAIL] {name}: {exc}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
