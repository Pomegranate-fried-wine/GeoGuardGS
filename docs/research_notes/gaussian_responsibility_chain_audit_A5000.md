# Gaussian Responsibility Chain Audit for A5000 Mainline

## Scope

This audit focuses on the 5000-iteration mainline:

- `output/local_formal/p15_allcam_A_da3_only_5000/`
- `output/local_formal/geometry_error_map/`
- `output/local_formal/gaussian_responsibility_v0_1_A/`
- `output/local_formal/gaussian_responsibility_v1_A/`
- `output/local_formal/responsibility_v1_5_A_compact/`
- `output/local_formal/structure_candidates_v0_A/`

The A10000 branch is intentionally ignored for current research decisions.

No training, Gaussian parameter edit, split, shrink, prune, opacity decay, or surface alignment was performed.

## Current A5000 State

`geometry_error_map` contains 75 views and 664,587 LiDAR-valid pixels.

Mask availability is good enough for the current 5000 mainline:

- boundary mask nonzero views: 75 / 75
- rendered-depth-edge mask nonzero views: 75 / 75
- canny mask nonzero views: 62 / 75
- thin-structure mask nonzero views: 61 / 75

This means A5000 does not suffer from the all-zero mask failure found in the A10000 branch.

The current candidate summary remains stable after the new overlap-availability guard:

- multi_view_persistent: 119,483
- global_responsibility_high: 80,191
- thin_structure_responsible: 39
- layer_conflict_high: 139
- stable_boundary_edge_conflict: 13,970
- border_suspect: 355,801
- low_support_uncertain: 181,622
- high_confidence_candidate: 9,037
- shrink_candidate: 293
- surface_align_candidate: 19
- opacity_decay_candidate: 71,154

Guard probe output:

`output/local_formal/responsibility_chain_audit/tag_guard_probe_A5000/`

## Responsibility-Chain Issues Found

### 1. Current responsibility is not true Gaussian contribution

The renderer/rasterizer Python wrapper exposes rendered RGB/depth/alpha, radii, and `visibility_filter`, but not per-pixel contributing Gaussian ids or alpha weights.

Therefore v0 responsibility is currently:

`2D screen-space support overlap with LiDAR-valid geometry_error_map`

It is useful for ranking candidates, but it is not a true causal or alpha-contribution attribution.

### 2. Legacy Gaussian id risk remains in existing A5000 outputs

Existing A5000 v0/v1/candidate npz files do not contain `stable_id_schema_version` or `uses_stable_gaussian_ids`.

The old v0 saved renderer graph row indices as `gaussian_ids`. Because StreetGS rebuilds `graph_gaussian_range` per view, these ids can be view-local rather than globally stable.

Impact:

- Single-view overlays remain meaningful because the row indices are valid within each view.
- Multi-view aggregation and candidate ids are weaker evidence until v0/v1 are regenerated with stable ids.
- Region-level clustering still gives useful image-space regions, but region-to-global-Gaussian identity should be treated cautiously.

Fix already implemented for future runs:

- v0 now saves stable ids as `model namespace * 10000000 + model-local index`.
- v0 keeps `view_local_gaussian_indices` for projection/overlay.
- v1, v1.5, candidate, region, and filtered-region summaries now propagate Gaussian id schema warnings.

### 3. Candidate tagging was too fragile when overlaps are missing

The candidate logic used percentile thresholds. If a mask family is all zero, thresholds can become 0 and generate false overlap-dependent labels.

Fix already implemented:

- overlap-dependent tags are disabled when the corresponding overlap family is unavailable or all zero.
- On A5000 this guard does not change tag counts, because boundary/rendered-edge/thin overlap signals are present.

### 4. Sensitivity test indexing bug

The v0 sensitivity test used `gaussian_ids` as indices into `centers/radii`. That only works when ids are view-local rows.

Fix already implemented:

- sensitivity now uses `view_local_gaussian_indices`.

### 5. Region filtering is useful but not yet a structure-operation proof

Existing A5000 region filtering produces 3 review regions:

- 2 `boundary_region`
- 1 `general_high_error_region`

The top review region is `000002_0`, with 1,350 candidate Gaussians, non-border ratio acceptable, high rendered-depth-edge overlap, and 3 shrink candidates.

This is suitable for manual review and pilot planning, but not yet enough to modify Gaussian parameters.

## What A5000 Responsibility Can Prove

It can support:

- high-error image regions are recoverable from LiDAR-valid geometry error;
- candidate tags are not caused by all-zero overlap masks;
- boundary/rendered-edge and thin evidence exists in the 5000 mainline;
- region-level clustering can identify localized non-border review regions such as `000002_0`.

It cannot yet prove:

- the same Gaussian id is correctly tracked across views in old outputs;
- a candidate Gaussian truly contributed alpha to the high-error pixels;
- removing/shrinking a candidate will improve geometry;
- candidate labels are sufficient for direct split/shrink/prune.

## Recommended Stronger Methods

### 1. No-CUDA MVP: depth-aware responsibility

Recommended next.

Keep screen-space support, but add:

- projected Gaussian depth;
- rendered depth consistency;
- LiDAR depth consistency only on valid LiDAR pixels;
- support confidence;
- opacity / radius guards;
- border penalty.

This should reduce false attribution from large background or behind-surface Gaussians while preserving the existing pipeline.

### 2. Lightweight debug mode: contribution-aware responsibility

For selected high-error pixels or selected review regions, record:

- top-K contributing Gaussian ids;
- alpha;
- transmittance;
- depth order;
- approximate `T_i * alpha_i` contribution weight.

This requires renderer/rasterizer debug output, but can be limited to sparse pixels or selected regions to avoid full-scale overhead.

### 3. Long-term: counterfactual / influence responsibility

For a small region, suppress or attenuate one Gaussian or a candidate cluster and rerender. If local RGB/depth/geometry error decreases, the Gaussian is more likely a bad contributor; if error increases, it may be a good contributor located in a bad area.

This is the strongest diagnostic but is expensive and should only be applied to a few review regions.

## Recommended Next Minimum Task

For the 5000 mainline:

1. Regenerate A5000 v0 with stable Gaussian ids.
2. Rerun v1/v1.5/candidate/region filtering using the stable-id outputs.
3. Confirm the A5000 tag counts and top review regions remain close to the current results.
4. Then generate a focused review pack for 1-3 regions, starting with `000002_0`.

Do not perform shrink/split/prune yet. The next implementation step should be stable-id rerun plus depth-aware responsibility, not structure correction.
