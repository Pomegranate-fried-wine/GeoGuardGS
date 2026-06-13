# Gaussian Responsibility Chain Audit for A10000

## Scope

This audit checks the post-training chain used after `p15_allcam_A_da3_only_10000`:

LiDAR-valid `geometry_error_map` -> single-view Gaussian responsibility -> multi-view aggregation -> v1.5 diagnostics -> candidate tags -> region candidates.

No training, Gaussian parameter edits, split, shrink, prune, opacity decay, or surface alignment was performed.

## Findings

### 1. Current responsibility is screen-space support, not true contribution

The Python rasterizer wrapper returns rendered image, radii, depth, alpha, semantic, and `visibility_filter`. It does not expose per-pixel Gaussian contributor ids or alpha weights to Python.

Therefore `diagnose_gaussian_responsibility.py` currently computes:

`responsibility = weighted mean geometry_error_map over a projected 2D support window`

This can identify Gaussians whose projected footprint overlaps LiDAR-valid geometry error, but it cannot prove that the Gaussian actually caused that pixel error through alpha compositing.

### 2. Gaussian id stability bug

The old v0 files saved `gaussian_id` as the renderer graph row index from `visibility_filter`. In StreetGS, `graph_gaussian_range` is rebuilt per camera/frame with background plus active object models. A row index is not a stable global Gaussian identity across views.

Impact:

- v1 aggregation can merge unrelated graph rows across views.
- candidate tags and region clustering can treat different Gaussians as the same primitive.
- existing A10000 v1/candidate outputs lack `stable_id_schema_version`, so they are legacy outputs.

Fix:

- v0 now saves stable ids as `model namespace * 10000000 + model-local index`.
- v0 also saves `view_local_gaussian_indices`, `model_local_indices`, and `model_names`.
- v1 summary now records whether the input used stable ids, and warns when legacy view-local ids are detected.
- v1.5, candidate tagging, region clustering, and filtered-region summaries now propagate id-schema status so legacy outputs are not silently treated as reliable.

### 3. Sensitivity indexing bug

The old v0 sensitivity test implicitly used `gaussian_ids` as `centers/radii` row indices. That only worked while `gaussian_id == view-local row`.

Fix:

- sensitivity now uses `view_local_gaussian_indices`.

### 4. A10000 mask / overlap propagation bug

Existing `geometry_error_map_A10000` has 75 views and 662,022 LiDAR-valid pixels, but every saved mask family is empty:

- boundary mask: 0 nonzero views
- canny mask: 0 nonzero views
- rendered-depth-edge mask: 0 nonzero views
- thin-structure mask: 0 nonzero views

The A10000 evaluation folders only contain `global_error_map.png`, `lidar_depth.png`, and `rendered_depth.png`, while `build_geometry_error_map.py` expected `boundary_band_mask.png`, `canny_band_mask.png`, `rendered_depth_edge_band_mask.png`, and `thin_structure_mask.png`.

Impact:

- `thin_structure_responsible` cannot be validated.
- `stable_boundary_edge_conflict` can become a false positive when percentile thresholds equal 0.
- region types based on boundary/thin/edge overlap are not reliable.

Fix:

- `build_geometry_error_map.py` now derives fallback canny, rendered-depth-edge, boundary, and thin masks when saved masks are absent.
- mask source strings are saved into `geometry_error_components.npz` and per-view summary.

### 5. Candidate tag false positives from all-zero overlaps

Existing A10000 candidate summary used:

- `boundary_overlap_p = 0`
- `rendered_edge_overlap_p = 0`
- `thin_overlap_p = 0`

Yet it produced 80,650 `stable_boundary_edge_conflict` and 5,298 `shrink_candidate` labels. This is not valid boundary evidence because the underlying masks were empty.

Fix:

- `tag_structure_candidates.py` now disables overlap-dependent tags when the corresponding mask/overlap family is all zero or unavailable.
- A guard probe on old A10000 inputs reduced high-confidence candidates from 38,780 to 2 and shrink candidates from 5,298 to 0, confirming that the old overlap-dependent labels were inflated.

### 6. Downstream schema visibility

Candidate and region scripts previously had no explicit way to tell whether their upstream ids were stable or legacy.

Fix:

- `gaussian_responsibility_global.npz` now stores `uses_stable_gaussian_ids`, `stable_id_schema_versions`, and `legacy_view_local_id_inputs`.
- `structure_candidates_summary.json` records upstream Gaussian id schema and warning.
- `region_candidates_summary.json` records candidate and v0 id schema.
- `filtered_region_summary.json` propagates the region id schema.

### 7. A10000 error map semantics need to be explicit

The existing `geometry_error_map_A10000` components include `B_minus_A`, `D_minus_A`, and `D_minus_B`, and `final_geometry_error_map` is not equal to raw `A_abs_error`.

For A10000 mainline candidate discovery, the preferred map should be an A-only LiDAR-valid self-error map unless the experiment explicitly asks for A/B/D degradation analysis.

## What Current Responsibility Can Prove

It can support this limited claim:

Visible Gaussians whose projected screen-space support overlaps LiDAR-valid high geometry error can be ranked more strongly than random or large-radius baselines.

It cannot yet prove:

- the Gaussian caused the pixel error;
- the Gaussian is the front-most contributor;
- the Gaussian alpha contribution was high;
- the same physical Gaussian was tracked across views in old legacy outputs;
- boundary/thin/edge explanations are valid in existing A10000 outputs with empty masks.

## Recommended Stronger Methods

### Recommended next MVP: depth-aware screen-space responsibility

This avoids CUDA changes and is the best next step.

For each visible Gaussian:

1. use stable Gaussian id;
2. project center/radius;
3. sample LiDAR-valid high-error pixels in its support;
4. compare approximate Gaussian camera depth to rendered depth or LiDAR depth;
5. downweight support pixels where the Gaussian depth is far behind or far in front of the error surface;
6. aggregate across views with support confidence and border penalties.

This improves over pure 2D overlap by reducing false attribution from background or large-radius Gaussians.

### Stronger but more invasive: top-K alpha contribution rasterizer output

Expose per-pixel top-K contributor Gaussian ids and alpha weights from the rasterizer. Responsibility becomes:

`sum_p error(p) * alpha_i(p) / sum_p alpha_i(p)`

This is the cleanest attribution signal, but it requires CUDA/rasterizer interface changes and should be treated as a later v2/v3 task.

### Alternative diagnostic: leave-one-out / mask-out rendering

For selected high-confidence regions only, temporarily suppress one Gaussian or a small candidate cluster during rendering and measure local error/RGB/depth change. This is more causal than overlap, but too expensive for full-scale use.

## Minimum Viable Next Task

Before any shrink-only or surface-align pilot:

1. Rebuild A10000 geometry error maps with fallback/available masks and an A-only LiDAR-valid self-error mode.
2. Rerun v0 with stable Gaussian ids.
3. Rerun v1/v1.5/candidate/region filtering from the regenerated v0.
4. Verify that:
   - v1 reports `uses_stable_gaussian_ids=true`;
   - overlap masks are non-empty for relevant views;
   - overlap-dependent candidate tags are not produced from all-zero masks;
   - selected regions remain after border and low-support guards.

Only after these checks should 1-3 single-region candidate-only pilot regions be selected.
