#!/usr/bin/env bash
set -euo pipefail

# Run from GitHub_main repository root.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[GeoGuardGS] Installing Python requirements"
pip install -r requirements.txt

echo "[GeoGuardGS] Installing diff-gaussian-rasterization"
pip install --no-build-isolation -e third_party/diff_gaussian_rasterization

echo "[GeoGuardGS] Installing simple-knn"
pip install --no-build-isolation -e third_party/simple_knn

if [ -f "third_party/nvdiffrast/setup.py" ]; then
  echo "[GeoGuardGS] Installing nvdiffrast"
  pip install --no-build-isolation -e third_party/nvdiffrast
fi

echo "[GeoGuardGS] Installing simple-waymo-open-dataset-reader if available"
if [ -f "third_party/simple_waymo_open_dataset_reader/setup.py" ]; then
  pip install -e third_party/simple_waymo_open_dataset_reader
fi

echo "[GeoGuardGS] Done. Verify imports before training."
