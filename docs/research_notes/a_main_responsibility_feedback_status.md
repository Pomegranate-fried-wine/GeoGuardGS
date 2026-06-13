# A Mainline Responsibility Feedback Status

## Scope

This document freezes the current A-mainline status after the 5000-iteration DA3-only baseline and the no-structure 6000-iteration feedback experiments.

The current goal is not to make DA3-guided feedback match LiDAR-supervised feedback. DA3 is not used as metric depth ground truth. The goal is to establish a reproducible branch where DA3 provides boundary, depth-ordering, and side-structure priors without LiDAR depth loss during training.

## Supervision Modes

`train.guided_feedback.supervision_mode` defines the feedback branch:

- `lidar_supervised`: LiDAR depth loss is allowed. This is an upper-bound reference branch, not the main unsupervised method.
- `da3_unsupervised`: LiDAR depth loss is forbidden. The training loss can use DA3 structure feedback only; LiDAR is used for evaluation only.
- `hybrid_reference`: LiDAR and DA3 can both be enabled. This is a diagnostic reference branch only.

For `da3_unsupervised`, the run must satisfy:

- `optim.lambda_depth_lidar == 0`
- `train.guided_feedback.use_lidar_depth == False`
- `train.guided_feedback.use_da3_structure == True`
- `train.guided_feedback.assert_no_lidar_supervision == True`

The verified log message is:

`LiDAR supervision disabled; LiDAR is used for evaluation only.`

## Why A5000 Is The Current Starting Point

A5000 is the current mainline baseline because it is the 75-view p15 all-camera DA3-only run with a complete downstream pipeline:

- enhanced LiDAR-valid evaluation;
- geometry error maps;
- screen-space and contribution-aware responsibility diagnostics;
- candidate and region-level review artifacts;
- no-structure continue-training branches.

A10000 is intentionally set aside for now. The current research logic and existing evidence are cleaner around A5000.

## DA3-Unsupervised Feedback Branch

The current DA3 feedback branch uses DA3 only as a structure prior:

- DA3 depth edge / gradient;
- DA3 boundary-risk regions;
- rendered-depth edge weakness or misalignment;
- local relative depth ranking;
- boundary-side structure consistency.

It deliberately avoids pointwise metric fitting of rendered depth to DA3 depth.

The implemented training loss is in:

- `lib/utils/da3_structure_feedback_utils.py`

The active losses are:

- edge-aware discontinuity loss;
- relative depth ranking loss;
- boundary-side consistency loss.

## Contribution-Aware CUDA Status

CUDA selected-pixel contribution dump is available in debug mode. It records, for selected DA3 boundary-risk pixels:

- top-K Gaussian ids;
- alpha;
- transmittance;
- `T * alpha` contribution weight;
- depth;
- depth order.

Current top30 DA3-risk contribution debug status:

- 30 selected regions;
- 27 CUDA-ok regions;
- 3 low-evidence regions;
- 573 nonzero contribution Gaussians;
- 6 bad contributors;
- 0 good contributors;
- 12 neutral contributors.

The hard bad/good contributor evidence is still sparse. Therefore the reproducible branch should be described as DA3 risk + CUDA contribution-supported softpatch feedback, not as fully causal bad/good contributor correction.

## Key Artifacts

DA3 boundary-risk signal:

- `output/local_feedback/da3_boundary_feedback_A5000_top30_regionpixels/guided_training_feedback_signal_da3_boundary_top30.json`

CUDA contribution debug:

- `output/local_feedback/da3_boundary_contribution_debug_A5000_top30_regionpixels/contribution_responsibility_all_views_summary.json`

Softpatch feedback signal:

- `output/local_feedback/da3_boundary_soft_contribution_feedback_A5000_top30/da3_contribution_softpatch_feedback_signal.json`

DA3-unsupervised no-structure run:

- `output/local_feedback/p15_A_da3_unsup_softpatch_continue_6000_nostruct_top30/`

Training log:

- `output/local_feedback/p15_A_da3_unsup_softpatch_continue_6000_nostruct_top30/train_da3_unsup_softpatch_6000.log`

Saved config:

- `output/local_feedback/p15_A_da3_unsup_softpatch_continue_6000_nostruct_top30/configs/config_005000.yaml`

Evaluation:

- `output/local_feedback/eval_da3_unsup_softpatch_6000_nostruct_top30_all/geometry_credibility_metrics.json`

Summary and audit:

- `output/local_feedback/da3_boundary_guided_6000_summary/six_way_da3_supervision_mode_summary.json`
- `output/local_feedback/da3_boundary_guided_6000_summary/da3_unsupervised_feedback_audit.json`

## Current Evidence

Common LiDAR support, global:

| Branch | AbsRel | RMSE | MAE | delta<1.25 |
| --- | ---: | ---: | ---: | ---: |
| plain6000_nostruct | 0.305810 | 7.514890 | 4.534798 | 0.533306 |
| lidar_guided_region_ref6000 | 0.152787 | 5.526700 | 2.443669 | 0.795725 |
| da3_boundary_region_guided6000 | 0.303844 | 7.455243 | 4.498644 | 0.536729 |
| da3_unsup_softpatch_guided6000 | 0.304818 | 7.473322 | 4.514384 | 0.535196 |

Common LiDAR support, DA3 top30 risk regions:

| Branch | AbsRel | RMSE | MAE | delta<1.25 |
| --- | ---: | ---: | ---: | ---: |
| plain6000_nostruct | 0.190767 | 4.419667 | 3.018514 | 0.783835 |
| lidar_guided_region_ref6000 | 0.141304 | 2.976059 | 1.958153 | 0.802632 |
| da3_boundary_region_guided6000 | 0.189743 | 4.292455 | 2.999812 | 0.795113 |
| da3_unsup_softpatch_guided6000 | 0.188993 | 4.169492 | 2.936612 | 0.800752 |

## What Can Be Claimed

- The code now cleanly separates LiDAR-supervised reference, DA3-unsupervised main branch, and hybrid reference.
- DA3-unsupervised feedback can resume A5000 to 6000 with structure updates disabled.
- Training log and saved config prove LiDAR supervision is disabled in the DA3-unsupervised branch.
- DA3 structure feedback gives small but consistent improvements over plain continue training on global common LiDAR support and DA3 top30 risk regions.
- LiDAR-supervised feedback is still much stronger, which is expected because it uses sparse ground-truth depth during training.

## What Cannot Be Claimed Yet

- Do not claim DA3 depth improves metric depth accuracy as a ground truth teacher.
- Do not claim fully causal bad/good Gaussian contributor correction.
- Do not claim contribution-specific feedback is solved. Hard bad/good labels are currently sparse.
- Do not claim readiness for shrink / split / prune.

## Next Method Direction

The next technical step should be DA3-structure counterfactual labeling:

`weaken contributor -> compare DA3 edge/ranking/side structure loss change -> classify bad/good/neutral`

This would make contributor classification consistent with the DA3-unsupervised training objective, instead of relying on sparse LiDAR error for counterfactual labels.

Do not run 7000 or structural edits until this 6000-level logic is cleaner.

## Live CUDA Periodic Feedback Update

The feedback controller now supports training-loop dynamic responsibility diagnosis with live selected-pixel CUDA contribution capture.

New engineering status:

- `lib/utils/cuda_contribution_utils.py` exposes `capture_contributions_cuda_live(...)`.
- The live API uses the current in-memory Street Gaussian model, current camera, and current Gaussian tensors.
- It does not reload checkpoints, does not modify opacity / scale / xyz, and runs under `torch.no_grad()`.
- It returns view-local row ids, stable Gaussian ids, alpha, transmittance, `T * alpha`, depth, depth order, support counts, and stable-id mapping diagnostics.
- `FeedbackController` can now run with `contribution_source=live_current_model`.
- Cached contribution summaries are now fallback/debug inputs, not the required path for dynamic controller smoke.

Validation:

- Offline live-vs-cached CUDA validation output:
  `output/local_feedback/live_cuda_contribution_api_validation/live_cuda_contribution_api_validation.json`
- In that validation, the tested cached and live CUDA contributors had shared top-K ratio 1.0, `T*alpha` mean/max absolute difference 0.0, depth-order difference 0.0, and unmapped id count 0.
- Training-loop live CUDA smoke output:
  `output/local_feedback/periodic_feedback_live_cuda_smoke_v2/feedback_controller/iter_005001/`
- The smoke manifest records `live_cuda_contribution=true`, `uses_cached_contribution=false`, `stable_id_map_available=true`, `unmapped_id_count=0`, `uses_lidar_supervision=false`, `uses_lidar_selected_pixels=false`, and `gaussian_parameters_modified=false`.

Current boundary:

- This is still feedback and diagnosis only.
- Group responsibility and candidate tags are generated for future repair design, but no real prune / shrink / split / opacity decay is enabled.
- The next stage may design Gaussian parameter operations, but they must remain config-gated and should start with very small dry-run or single-region pilots.
