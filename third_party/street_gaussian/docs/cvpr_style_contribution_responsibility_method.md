# Contribution-Aware and Counterfactual Gaussian Responsibility

## Method Motivation

Current Gaussian responsibility based on screen-space overlap answers only a weak question: whether a Gaussian footprint spatially intersects a high-error image region. This is insufficient for trustworthy structure diagnosis, because a Gaussian may overlap an error pixel while contributing little or nothing to the rendered color/depth after alpha compositing. Conversely, a small or partially occluded Gaussian may have decisive contribution at a boundary or depth-transition pixel but be missed by footprint-based ranking.

We therefore define Gaussian responsibility through the actual rasterization process. A Gaussian is considered responsible only when it contributes to the rendered high-error pixel through the differentiable alpha-compositing chain, and it is considered a bad contributor only when counterfactually weakening it reduces the local geometry error without significantly degrading RGB fidelity.

## High-Error Region Selection

Given a rendered view and sparse LiDAR supervision, we first compute a LiDAR-valid geometry error map

`E_geo(p) = |D_render(p) - D_lidar(p)|`

only for pixels with valid LiDAR measurement. Invalid LiDAR pixels are excluded and are never treated as zero depth.

For a candidate region `Omega`, selected from region-level diagnostic outputs or high-error connected components, we sample high-evidence pixels

`P_Omega = {p in Omega | valid_lidar(p)=1, E_geo(p) >= tau_Omega}`.

Regions with too few LiDAR-valid high-error pixels are marked as `low_evidence` and excluded from strong responsibility claims.

## Rasterizer Contribution Extraction

For each selected pixel `p`, Gaussian splatting renders the image by front-to-back alpha compositing. For Gaussian `i` at pixel `p`, let

- `alpha_i(p)` be the Gaussian opacity contribution after projected Gaussian falloff;
- `T_i(p) = product_{j < i} (1 - alpha_j(p))` be the accumulated transmittance before Gaussian `i`;
- `w_i(p) = T_i(p) alpha_i(p)` be the true rendering contribution weight.

Rendered color and depth can be expressed as

`C(p) = sum_i w_i(p) c_i + T_bg c_bg`

`D(p) = sum_i w_i(p) d_i`

where `d_i` is the view-space depth of Gaussian `i`.

Our debug implementation extracts `w_i(p)` without modifying CUDA by using a semantic one-hot probe. For a chunk of candidate Gaussians, we assign one-hot semantic features and re-render the selected view. Since the rasterizer accumulates semantic features using the same `T_i alpha_i` term as RGB/depth, the rendered semantic value at pixel `p` directly equals the true contribution weight of the probed Gaussian.

This converts responsibility from geometric overlap to actual rasterizer participation.

## Contribution-Aware Responsibility

For each Gaussian `i`, contribution-aware responsibility is defined as

`R_i = sum_{p in P_Omega} w_i(p) E_geo(p)`.

We also store:

- support pixel count: `|{p | w_i(p) > epsilon}|`;
- maximum contribution weight: `max_p w_i(p)`;
- depth order among top contributors;
- rendered depth contribution: `w_i(p) d_i`;
- model namespace and Gaussian id.

This makes it possible to distinguish front-layer contributors, background contributors, occlusion-layer contributors, and weak overlap-only Gaussians.

## Counterfactual Weakening Verification

Contribution alone does not prove whether a Gaussian is bad or good. A Gaussian can contribute to an error pixel but still be necessary for preserving correct structure or color. We therefore introduce a local counterfactual score.

For a top suspicious Gaussian `i`, we temporarily weaken its opacity

`alpha_i'(p) = lambda alpha_i(p), 0 <= lambda < 1`

and re-render the selected region without saving any parameter change. Let

`E_with_i = mean_{p in P_Omega} |D_render(p) - D_lidar(p)|`

`E_weak_i = mean_{p in P_Omega} |D_render^{weak(i)}(p) - D_lidar(p)|`.

The counterfactual geometry score is

`C_i = E_with_i - E_weak_i`.

We also evaluate RGB preservation:

`Delta_RGB_i = mean_{p in P_Omega} |I_render^{weak(i)}(p) - I_gt(p)| - |I_render(p) - I_gt(p)|`.

Classification:

- bad contributor: `C_i > epsilon_geo` and `Delta_RGB_i <= epsilon_rgb`;
- good contributor: `C_i < -epsilon_geo`;
- neutral contributor: otherwise.

This separates harmful contributors from useful Gaussians that merely lie in a difficult region.

## Prototype Evidence on A5000

In A5000, we ran the prototype on the filtered region `000002_0`, region 0, a boundary region.

Summary:

- selected LiDAR-valid high-error pixels: 2;
- candidate pool: 12,356 Gaussians;
- true nonzero contributors: 10;
- old screen-space v0 top-100 vs contribution-aware top-100 overlap: 0;
- counterfactual bad contributors: 5;
- good contributors: 0;
- neutral contributors: 0;
- semantic-probe time: about 46.3 seconds;
- counterfactual time: about 0.38 seconds.

Top contribution-aware Gaussians:

- `902074`, `R_i = 0.1338`;
- `967346`, `R_i = 0.1222`;
- `950007`, `R_i = 0.1053`.

Counterfactual weakening reduced LiDAR depth error:

- `902074`: `21.06 -> 16.97`, RGB error increase `0.0024`;
- `967346`: `21.06 -> 16.97`, RGB error increase `0.0024`;
- `950007`: `21.06 -> 18.19`, RGB error increase `0.0041`.

The old overlap-based method selected many Gaussians that had no measured true contribution at the selected pixels, while the contribution-aware method selected actual alpha-compositing participants.

Two additional filtered regions were marked `low_evidence` because they had fewer than two LiDAR-valid high-error pixels. They should not be used for strong claims.

## Core Innovation Claim

The core innovation is a geometry responsibility attribution mechanism that moves from correlation to rendering participation and then to counterfactual verification:

`high-error region -> true rasterizer contribution -> contribution-weighted responsibility -> counterfactual weakening -> bad/good/neutral contributor`

Compared with screen-space responsibility, this method:

1. uses true `T_i alpha_i` contribution instead of projected overlap;
2. incorporates occlusion and depth order through the rasterizer compositing process;
3. validates causality through local counterfactual weakening;
4. provides a principled interface for training feedback or later structure correction.

## Suggested Paper Framing

Possible title:

**From Pixel Geometry Error to Responsible Gaussians: Contribution-Aware and Counterfactual Attribution for Trustworthy Street Gaussian Reconstruction**

Abstract draft:

Street-view Gaussian reconstruction often produces visually plausible renderings while retaining localized geometric failures around object boundaries, thin structures, and depth transitions. Existing diagnostics typically compare rendered depth with sparse LiDAR at the pixel level, but they do not identify which Gaussian primitives are responsible for the error. We propose a contribution-aware and counterfactual Gaussian responsibility mechanism. Instead of assigning responsibility by screen-space overlap, we extract the true alpha-compositing contribution `T_i alpha_i` of Gaussians at LiDAR-valid high-error pixels, define a contribution-weighted geometry responsibility score, and verify suspicious Gaussians through local counterfactual opacity weakening. This allows us to distinguish bad contributors from good or neutral contributors and provides a reliable interface for geometry-guided training feedback or structure correction. Experiments on Waymo street scenes show that contribution-aware responsibility selects substantially different primitives from overlap-based methods and identifies Gaussians whose weakening reduces LiDAR depth error while preserving RGB fidelity.

Contribution points:

1. A LiDAR-valid high-error region formulation for trustworthy geometry diagnosis in sparse autonomous-driving data.
2. A rasterizer-contribution extraction mechanism that measures true `T_i alpha_i` Gaussian participation at high-error pixels.
3. A contribution-aware responsibility score that accounts for alpha compositing, occlusion, and depth contribution.
4. A counterfactual weakening test to classify Gaussians as bad, good, or neutral contributors.
5. A responsibility-to-training-feedback interface for future iterative reconstruction optimization.

## Recommended Method Section Structure

1. Problem setup and LiDAR-valid geometry error.
2. High-error pixel and region selection.
3. Rasterizer contribution extraction.
4. Contribution-aware Gaussian responsibility.
5. Depth-order and occlusion-aware diagnostics.
6. Counterfactual weakening and contributor classification.
7. Training feedback / structure correction interface.

## Recommended Experiments

Required comparisons:

- screen-space v0 responsibility vs contribution-aware responsibility;
- contribution-aware with and without counterfactual verification;
- weakening bad contributors vs weakening random / screen-space top-K / large-radius Gaussians;
- RGB fidelity before and after weakening;
- multi-region and multi-view stability;
- low-evidence region rejection.

Recommended visualizations:

- method pipeline diagram;
- old v0 top-K vs true contribution top-K overlay;
- per-pixel top-K contributor table with `T_i alpha_i`, depth order, and depth contribution;
- counterfactual before/after depth-error crop;
- bad/good/neutral contributor examples;
- region-level qualitative panels.

## What Can and Cannot Be Claimed Now

Can claim:

- The prototype captures true rasterizer contribution weights using the semantic one-hot probe.
- Contribution-aware top-K can differ sharply from screen-space top-K.
- On one A5000 high-evidence region, weakening top contribution-aware Gaussians reduced LiDAR depth error with small RGB error increase.
- The method is a plausible core mechanism for responsible-Gaussian attribution.

Cannot yet claim:

- Full-scale robustness across all scenes/views.
- Superiority over all baselines without broader statistics.
- Safe structure correction.
- Training improvement from feedback, because training integration has not been run.
- Complete all-Gaussian CUDA-level extraction; current prototype uses a candidate pool and repeated semantic rendering.

## Next Engineering Step

Keep the semantic-probe debug path for correctness, but add a CUDA debug mode that directly dumps selected-pixel top-K:

- Gaussian id;
- alpha;
- transmittance;
- `T_i alpha_i`;
- depth;
- depth order.

This will make the method scalable while preserving the same mathematical definition.
