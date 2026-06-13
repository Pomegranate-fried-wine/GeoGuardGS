# CUDA Selected-Pixel Contribution Dump

This debug path exposes sparse, selected-pixel Gaussian contribution records from
`diff_gaussian_rasterization` for contribution-aware responsibility diagnosis.
It is not used by default training or normal rendering.

## Interface

Python binding:

```python
from diff_gaussian_rasterization import _C
_C.rasterize_gaussians_contrib(...)
```

The signature follows the existing rasterizer inputs and adds:

- `selected_pixels`: int32 tensor with shape `[N, 2]`, storing `(x, y)`.
- `top_k`: number of contributors to keep per selected pixel.

Returned tensors:

- `ids`: `[N, top_k]` Gaussian row ids.
- `alpha`: `[N, top_k]` per-Gaussian alpha at that pixel.
- `transmittance`: `[N, top_k]` accumulated transmittance before that Gaussian.
- `weight`: `[N, top_k]` contribution weight `T_i * alpha_i`.
- `depth`: `[N, top_k]` Gaussian depth used by the rasterizer.
- `depth_order`: `[N, top_k]` position in the rasterizer's sorted per-tile
  traversal for that pixel. This is the alpha-compositing traversal order used
  by the rasterizer, but it is not an object/layer id.

Invalid entries use id `-1` and zero-valued statistics.

## Implementation

The debug function reuses the standard forward rasterizer setup to build geometry,
binning, image buffers, tile ranges, and sorted Gaussian lists. For each selected
pixel, a CUDA kernel scans only the corresponding tile range, recomputes the same
Gaussian falloff, alpha, and front-to-back transmittance as the normal forward
pass, and keeps the top-K contributors ranked by `T_i * alpha_i`.

This gives real rasterizer contribution evidence for selected high-error pixels
without allocating a full image-sized contributor map.

## Debug Script

`script/debug_contribution_responsibility.py` now supports:

```bash
--contribution-backend auto|cuda|semantic
```

- `cuda`: requires `_C.rasterize_gaussians_contrib`; raises if unavailable.
- `auto`: uses CUDA if available, otherwise falls back to semantic one-hot probe.
- `semantic`: uses the older one-hot semantic probe.

The script saves CUDA contribution tensors in `contribution_responsibility_debug.npz`
and uses the same responsibility formula:

```text
R_i = sum_p (T_i(p) * alpha_i(p) * E_geo(p))
```

Counterfactual weakening is still temporary and debug-only; it does not save or
modify Gaussian parameters.

## A5000 Smoke Validation

Command:

```bash
python script/debug_contribution_responsibility.py --config output/local_formal/p15_allcam_A_da3_only_5000/configs/config_000000.yaml mode evaluate --regions-csv output/local_formal/structure_candidates_v0_A/region_candidates_v0_filtered/filtered_region_candidates.csv --top-regions 1 --max-pixels 8 --min-evidence-pixels 2 --top-k-per-pixel 8 --counterfactual-top-k 5 --contribution-backend cuda --output-dir output/local_formal/contribution_responsibility_cuda_debug_A5000_smoke_v2
```

Result for `000002_0:region0`:

- CUDA backend available: true.
- Nonzero contributors: 13.
- CUDA top-3 Gaussian ids: `902074`, `967346`, `950007`.
- Semantic probe top-3 Gaussian ids on the same pixels: `902074`, `967346`, `950007`.
- Top-3 overlap: 3/3.
- CUDA contribution capture time: about 0.03 s.
- Semantic one-hot capture time: about 46 s.
- Counterfactual labels for CUDA top-5:
  - bad contributors: `902074`, `967346`, `950007`;
  - good contributors: `261467`, `714177`.

The top-10 CUDA and semantic lists differ beyond the first three because the
semantic fallback remains bounded by a screen-space candidate pool, while CUDA
scans the rasterizer's true sorted per-tile list.

## Sanity Checks

Additional A5000 checks on `000002_0:region0` verify the math behind the dump:

- With `top_k=16`, the sparse top-K dump intentionally covers only part of full
  alpha compositing on the selected pixels:
  - mean `sum(T*alpha) / rendered_acc`: about `0.611`;
  - mean `sum(T*alpha*depth) / rendered_depth`: about `0.514`.
- With `top_k=128`, the selected-pixel dump reconstructs the normal rendered
  quantities to numerical precision:
  - mean absolute `sum(T*alpha) - rendered_acc`: about `8e-8`;
  - mean absolute `sum(T*alpha*depth) - rendered_depth`: about `7e-6`.
- For shared tested contributors, CUDA `T*alpha` and semantic one-hot probe
  weights match exactly in the current run.

The compact sanity report is saved at:

```text
output/local_formal/contribution_responsibility_cuda_sanity_A5000_regions/cuda_sanity_check_summary.json
```

## Limitations

- This is a debug/evaluation path, not a training-time path.
- It currently targets sparse selected pixels, not dense contributor maps.
- The function re-runs forward setup internally, so it is not yet optimized for
large batched view processing.
- Depth order is the rasterizer's sorted per-tile order, useful for occlusion
diagnosis but not yet a full layer-conflict model.
