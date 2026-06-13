# Third-party components

GeoGuardGS builds on several third-party projects. This directory contains
source snapshots needed for migration and server-side compilation. It must not
contain datasets, pretrained weights, checkpoints, or local compiled binaries.

## Street Gaussian / Street Scene Gaussian Splatting

Used as the base dynamic street-scene Gaussian reconstruction framework. Check
the upstream license and citation requirements before redistribution.

## Depth Anything 3

Used as a geometry-structure prior provider. DA3 depth is not treated as metric
ground truth in the DA3-unsupervised branch. Check the upstream model license
and citation requirements before downloading or redistributing weights.

## diff-gaussian-rasterization

Used for Gaussian rasterization and selected-pixel contribution debug dumping.
CUDA extensions may require local compilation. Check upstream license and CUDA
compatibility before redistribution.

## Server-side rebuild requirement

After migration to an A100 server, rebuild these extensions:

```bash
pip install --no-build-isolation -e third_party/diff_gaussian_rasterization
pip install --no-build-isolation -e third_party/simple_knn
pip install --no-build-isolation -e third_party/nvdiffrast
```

Do not commit compiled `.so`, `.pyd`, `.dll`, `build/`, or `*.egg-info`
artifacts.

## Citation note

Please verify exact citation metadata before paper submission.
