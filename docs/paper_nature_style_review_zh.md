# GeoGuardGS Nature-style Evidence Review

Date: 2026-06-16

This review follows an evidence-first, claim-bounded workflow inspired by the
installed `nature-reviewer`, `nature-writing`, and `nature-figure` skills. It
uses only the current local result artifacts under
`server_results_review/paper_results_formal_4groups_v2` and
`server_results_review/paper_evidence_formal_4groups_v2`.

## Review setup

- Input scope: four formal experiments on Waymo held-out test split, final
  checkpoint evaluation, training diagnostic curves, initialization and
  feedback audit tables, and fixed-view RGB/depth panels.
- Assessment boundary: the review can judge whether the current evidence
  supports a bounded methods/experimental paper. It cannot prove broad
  generality across datasets or scenes because only the current downloaded
  result package is available.
- Shared manuscript claim summary: GeoGuardGS explores replacing training-time
  LiDAR supervision in object-aware Street Gaussians with DA3-guided structure
  and periodic feedback, and includes a pure-vision object-aware variant using
  COLMAP background initialization and random-box object initialization.
- Visible evidence base:
  - Final held-out test evaluation on 245 views per group.
  - Full-image, object-region, and background-region CSV metrics.
  - A/B/C/PV-C main table with PSNR, SSIM, LPIPS, and L1.
  - 90 same-view RGB/depth comparison figures: 6 iterations x 15 views.
  - Initialization audit and feedback trigger summaries.
  - Sampled and full-split PSNR/L1 training diagnostic curves.
- Missing materials affecting confidence:
  - No `train_loss_trace.csv` was present in the downloaded server package, so
    true loss curves are missing.
  - Evidence is currently from one Waymo scene/split package; multi-scene
    generalization is not established.
  - Object-region metrics are valid on 84/245 held-out views; this is usable
    but should be reported as an object-present subset, not the full test set.
  - Final-evaluation depth metrics against ground-truth LiDAR are not included
    in the current main evidence pack.

## Reviewer 1: Technical soundness

### Overall assessment

The current result package is technically sufficient for a bounded
experimental-methods paper, but not yet for a strong broad-claim Nature-level
paper. The most solid evidence is that all four formal groups completed final
held-out test evaluation, object-aware rendering is active, and the pure-vision
variant remains competitive in full-image RGB metrics while passing the
available no-LiDAR initialization/supervision audit.

### Major strengths

- The formal evaluation is held-out test based rather than training-view only.
- The four groups separate LiDAR-supervised baseline, LiDAR-init DA3-only,
  LiDAR-init DA3+Feedback, and pure-vision DA3+Feedback.
- The current output includes both quantitative tables and same-view qualitative
  RGB/depth comparisons.
- Feedback groups have explicit trigger counts: C has 59/59 valid feedback
  triggers and PV-C has 30/30 valid triggers.

### Major concerns

- The loss trace is missing, so training objective dynamics cannot be plotted
  or discussed quantitatively.
- The main quantitative gains are not clearly in favor of the feedback method:
  C is slightly below B in final full-image PSNR/SSIM/LPIPS, while PV-C is
  strongest in PSNR/SSIM but not LPIPS.
- Depth images are qualitative rendered-depth panels, not metric depth
  evaluation against a held-out sensor source.
- Object-region rows are valid only for the object-present subset. This is
  acceptable if stated, but misleading if presented as 245-view object metrics.

### Technical failings to address

1. Add or recover `train_loss_trace.csv` if the paper will discuss loss curves.
2. Add a table for object-present view count and object-region metric scope.
3. Add final depth/geometry evaluation if claiming geometric improvement.
4. Separate "diagnostic training evaluation" from "final held-out test
   evaluation" in all figure captions.

## Reviewer 2: Originality and significance

### Overall assessment

The strongest novelty is not that DA3+Feedback beats StreetGS in RGB metrics;
the current table does not show that. The stronger paper angle is a controlled
study of LiDAR dependency in object-aware street Gaussian reconstruction,
including a pure-vision object-aware DA3+Feedback setting that preserves
vehicles and reaches StreetGS-level full-image PSNR/SSIM on the tested split.

### Major strengths

- The experiment design directly addresses a meaningful autonomy question:
  what remains possible when LiDAR supervision, and eventually LiDAR
  initialization, are removed.
- PV-C is a useful stage-level innovation: COLMAP background + random-box
  object initialization + DA3 feedback, with no LiDAR initialization and no
  LiDAR supervision.
- The audit table makes the supervision boundary explicit, which is important
  for paper credibility.

### Major concerns

- A strong claim that feedback improves reconstruction quality is not supported
  by the current final metrics.
- A strong claim that the method improves geometry is not supported without
  final geometry/depth metrics.
- A high-impact venue would expect broader scenes, more baselines, and clearer
  ablation of feedback components.

### Recommended claim boundary

Supported:

> A pure-vision object-aware DA3+Feedback configuration can be trained without
> LiDAR initialization or LiDAR supervision while retaining competitive
> held-out RGB rendering quality in the evaluated Waymo scene.

Not yet supported:

> DA3+Feedback consistently outperforms StreetGS.

Not yet supported:

> GeoGuardGS improves 3D geometry or safety-critical object structure across
> driving scenes.

## Reviewer 3: Readability and figure readiness

### Overall assessment

The result package is now readable enough for a first manuscript draft. The
figure set has a usable logic: main quantitative comparison, no-LiDAR audit,
feedback timeline, training diagnostics, and fixed-view RGB/depth plates.
However, figure captions must carefully distinguish diagnostic curves from main
evaluation.

### Figure contract

Figure 1: Method overview. Still missing. Should show the four experimental
conditions and the DA3/feedback loop.

Figure 2: Main held-out test table/plot. Supported by
`final_full_evaluation_summary.csv` and `table_main_results`.

Figure 3: Same-view qualitative RGB/depth plate. Supported by
`formal_rgb_depth_comparisons`, preferably using iteration 30000 and 4-6
representative views.

Figure 4: Training diagnostics. Supported by sampled/full-split PSNR/L1 curves;
not supported for loss curves until `train_loss_trace.csv` is recovered.

Figure 5: Audit and feedback evidence. Supported by initialization audit and
feedback trigger timeline.

## Cross-review synthesis

The current package can support a credible workshop/conference-style or
methods-oriented journal manuscript draft with bounded claims. It does not yet
support a broad high-impact claim that the method is generally superior to
StreetGS or that feedback repairs geometry in a validated safety sense.

The strongest defensible paper is:

> "Auditing LiDAR dependency in object-aware street Gaussian reconstruction
> with DA3-guided feedback."

The strongest empirical result is:

> PV-C, the pure-vision object-aware variant, achieved 25.63 PSNR and 0.8468
> SSIM on the held-out test split, comparable to or slightly above the
> LiDAR-initialized StreetGS baseline in this run, while passing the no-LiDAR
> initialization and supervision audit.

The main caveat is:

> This is currently a one-scene formal result with missing loss traces and no
> final metric geometry/depth validation.

## Required additions before a stronger paper claim

1. Recover or regenerate training scalar traces if loss curves are required.
2. Add final held-out depth/geometry metrics if claiming structural or depth
   improvement.
3. Add at least one more Waymo scene or a clear statement that this is a
   single-scene study.
4. Add a method overview figure.
5. Add object-region summary table with `valid_object_views=84/245`.
6. Decide whether the paper targets a methods workshop/conference or a broader
   journal, because the current evidence fits the former more naturally.

