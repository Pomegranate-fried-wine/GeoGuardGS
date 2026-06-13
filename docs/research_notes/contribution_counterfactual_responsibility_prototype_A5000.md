# Contribution-Aware / Counterfactual Gaussian Responsibility Prototype

## Goal

Move beyond screen-space overlap responsibility. The target is to identify Gaussians that truly participate in rendering high-error pixels and may cause geometry error.

This prototype is debug-only and runs on A5000 selected high-error pixels / regions.

## Implemented Script

`script/debug_contribution_responsibility.py`

Main idea:

1. Select LiDAR-valid high-error pixels from `geometry_error_map`.
2. Build a loose candidate pool only to limit debug cost.
3. Encode candidate Gaussians as one-hot semantic channels in chunks.
4. Re-render with the existing rasterizer.
5. Read the returned semantic image at selected pixels.

Because the rasterizer accumulates semantic features with the same term used for RGB/depth blending:

`semantic_i(p) = T_i(p) * alpha_i(p)`

the captured value is the real rasterizer contribution weight for that Gaussian at that pixel. The final responsibility is:

`R_i = sum_p T_i(p) * alpha_i(p) * E_geo(p)`

This is not a screen-space overlap score.

## Depth / Occlusion Awareness

For each per-pixel contributor, the script records:

- Gaussian id;
- view-local index;
- model name;
- contribution weight `T_i * alpha_i`;
- depth order among captured top contributors;
- Gaussian camera depth;
- rendered depth contribution `depth_i * T_i * alpha_i`;
- pixel geometry error.

This lets later logic distinguish primary foreground contributors, background contributors, and occlusion/mixed-layer cases.

## Counterfactual Probe

For top suspicious contributors, the script temporarily weakens a Gaussian opacity and re-renders. It compares LiDAR-valid depth error before and after weakening:

`C_i = E_with_i - E_with_weakened_i`

Positive `C_i` means weakening reduced error, so the Gaussian is labeled as a likely `bad_contributor`.

No Gaussian parameters are saved or modified.

## A5000 Smoke Result

Command:

```powershell
python script\debug_contribution_responsibility.py --config output/local_formal/p15_allcam_A_da3_only_5000/configs/config_000000.yaml mode evaluate --views 000002_0 --pixel-bbox 345 415 532 545 --max-pixels 32 --max-candidate-gaussians 20000 --old-v0-pool-size 20000 --semantic-chunk-size 64 --top-k-per-pixel 8 --counterfactual-top-k 5 --output-dir output/local_formal/contribution_responsibility_debug_A5000_region000002_0
```

Output:

`output/local_formal/contribution_responsibility_debug_A5000_region000002_0/000002_0/`

Observed:

- selected LiDAR-valid high-error pixels in the region: 2
- candidate pool: 12,356 Gaussians
- nonzero true-contribution Gaussians: 10
- semantic probe sanity: nonzero, confirming the rasterizer semantic path records real contribution weights
- selected-pixel acc: mean 0.968, so these are not empty/background-only pixels

Top contribution-aware Gaussians:

- `902074`, responsibility `0.1338`, max contribution weight `0.08365`
- `967346`, responsibility `0.1222`, max contribution weight `0.07603`
- `950007`, responsibility `0.1053`, max contribution weight `0.06399`

Old v0 overlap top-100 and new contribution top-100 overlap count: 0.

This is strong evidence that screen-space overlap and true contribution select different Gaussians.

## Counterfactual Result

Weakening the top contributors reduced LiDAR depth error on the selected pixels:

- `902074`: mean error `21.06 -> 16.97`, reduction `4.09`, improved ratio `1.0`
- `967346`: mean error `21.06 -> 16.97`, reduction `4.09`, improved ratio `1.0`
- `950007`: mean error `21.06 -> 18.19`, reduction `2.87`, improved ratio `1.0`
- `793536`: mean error `21.06 -> 19.71`, reduction `1.35`, improved ratio `1.0`
- `965603`: mean error `21.06 -> 19.71`, reduction `1.35`, improved ratio `1.0`

These are prototype-level `bad_contributor` signals.

## Limitations

- The current prototype still uses a candidate pool before contribution probing, so it is not a full all-Gaussian per-pixel dump.
- It uses repeated semantic-channel rendering, which is correct but slow.
- Selected pixels are sparse because Waymo LiDAR is sparse.
- Counterfactual is per-Gaussian and local; interactions among Gaussians are not fully modeled.
- Existing old v0 ids may be legacy view-local ids, so old/new id comparison is indicative.

## Paper Potential

This mechanism has stronger paper potential than screen-space responsibility:

1. It directly measures true rasterizer contribution weight `T_i * alpha_i`.
2. It captures depth order and occlusion context.
3. It distinguishes bad contributors from good contributors through counterfactual error change.
4. It can bridge pixel-level geometry error and Gaussian-level structure intervention more rigorously.

## Recommended Next Step

Turn this debug prototype into a two-level method:

1. Fast no-CUDA mode:
   - run semantic-probe contribution capture on selected high-error pixels / regions;
   - add depth-aware filters;
   - aggregate contribution responsibility over views.

2. CUDA debug mode:
   - expose sparse selected-pixel top-K contributor ids, alpha, transmittance, depth order, and `T_i * alpha_i` directly from the rasterizer;
   - avoid repeated semantic re-rendering;
   - keep it debug-only at first.

Only after this contribution/counterfactual evidence is stable should shrink, opacity decay, surface-align, or split be tested.
