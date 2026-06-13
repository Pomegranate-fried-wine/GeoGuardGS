# Responsible Gaussian Controlled Repair Design

## Purpose

This document defines the next-stage controlled repair design. It is a design only. No Gaussian parameters should be modified at this stage.

The current pipeline has established:

`DA3 boundary-risk region -> CUDA selected-pixel contribution -> DA3-unsupervised softpatch feedback -> no-structure 6000 evaluation`

The next research question is how to convert reliable responsibility evidence into conservative Gaussian-level control without damaging RGB reconstruction.

## Why Not Shrink / Split / Prune Yet

The current evidence is not strong enough for permanent structure edits:

- screen-space responsibility is not causal;
- CUDA contribution is real but hard bad/good labels are sparse;
- DA3-risk softpatch feedback is reproducible, but it is still region/patch-level;
- LiDAR is sparse and should remain evaluation-only in the DA3-unsupervised branch;
- RGB preservation has to be treated as a first-class constraint.

Therefore, the next stage should be candidate tagging and dry-run scoring before any real shrink, split, prune, opacity decay, or surface alignment.

## Contributor Categories

### Bad Contributor

A bad contributor is a Gaussian that has strong rendered contribution in a DA3 boundary-risk region and whose temporary weakening improves the relevant structure objective.

Future possible actions:

- local geometry constraint;
- opacity regularization;
- shrink candidate;
- split candidate;
- surface-align candidate.

No action should run unless the contributor is supported by multi-view or high-confidence evidence.

### Good Contributor

A good contributor is a Gaussian that contributes to a risky region but weakening it worsens RGB or DA3 structure consistency.

Future possible actions:

- protect from pruning;
- avoid opacity decay;
- preserve contribution during local repair.

### Neutral Contributor

Neutral contributors have weak or ambiguous counterfactual effect.

Future action:

- skip;
- keep for statistics only.

### Low-Evidence Contributor

Low-evidence contributors or regions have too few valid pixels, unstable contribution, severe border risk, or weak DA3 confidence.

Future action:

- skip;
- request more views or better signal;
- do not promote to structure edits.

## Role Of DA3 Boundary-Risk Regions

DA3 boundary-risk regions are the unsupervised structure-control targets. They should encode:

- DA3 depth edge high, rendered depth edge weak;
- DA3/rendered edge misalignment;
- DA3 indicates foreground/background layering but rendered depth is over-smoothed;
- optional RGB edge support;
- optional DA3 confidence or multi-frame consistency later.

DA3 risk is not metric depth supervision. It should only guide boundary strength, local order, and side structure.

## Role Of LiDAR

LiDAR branch remains:

- sparse ground-truth evaluation;
- supervised upper-bound reference;
- not used in `da3_unsupervised` training feedback.

All papers, notes, and scripts must state this separation clearly.

## Future Code Attachment Points

Potential future structure-control code should be config-gated and default off.

Likely attachment points:

- `train.py`: only for scheduled dry-run scoring or config-gated repair calls.
- `lib/utils/guided_feedback_utils.py`: load control candidates and expose region/Gaussian decisions.
- `lib/models/gaussian_model.py` and related StreetGS Gaussian model classes: only if a future task explicitly enables controlled Gaussian edits.
- `lib/models/street_gaussian_renderer.py`: for contribution-aware diagnostics, not default behavior.
- `script/debug_contribution_responsibility.py`: continue as debug/evaluation evidence generator.
- new scripts such as `script/tag_controlled_repair_candidates.py` or `script/dryrun_controlled_repair.py`.

Do not put repair behavior into default training paths.

## Safety Checks Before Any Real Edit

A Gaussian-level operation can only be considered if all checks pass:

- config flag explicitly enables the operation;
- no operation is enabled by default;
- `train.disable_structure_updates` is respected unless the experiment explicitly overrides it;
- candidate is not `border_suspect`;
- candidate is not `low_support_uncertain`;
- selected pixels come from reliable DA3 boundary-risk or validated LiDAR evaluation, depending on branch;
- contribution evidence includes real `T * alpha`, not only screen-space overlap;
- RGB loss or patch RGB error does not worsen beyond a configured threshold;
- the candidate has enough support pixels;
- multi-view consistency is preferred before permanent edits;
- all operations must be logged with Gaussian ids, view ids, affected pixels, and before/after metrics.

## Operation Ideas

### Shrink Candidate

Use when:

- high contribution in a boundary-risk region;
- large projected radius or broad support;
- boundary / rendered-depth-edge overlap is high;
- evidence suggests the Gaussian crosses a depth transition.

Expected effect:

- reduce over-smoothing across boundary;
- reduce spatial bleeding.

Safety:

- dry-run first;
- verify RGB patch error;
- compare DA3 edge/ranking loss before/after.

### Split Candidate

Use when:

- one Gaussian appears to cover multiple depth layers;
- depth-order conflict is strong;
- multi-view evidence is stable.

Expected effect:

- separate foreground/background support.

Safety:

- postpone until contribution labeling is stronger;
- never use single-view evidence alone.

### Opacity Regularization / Decay Candidate

Use when:

- contributor is repeatedly bad or uncertain;
- RGB contribution is not essential;
- weakening improves structure without hurting RGB.

Safety:

- prefer soft regularization before direct opacity changes.

### Surface-Align Candidate

Use when:

- thin-structure or boundary structure is reliable;
- DA3 ranking supports local surface orientation;
- contribution is localized.

Safety:

- use only after stronger thin-structure evidence.

## DA3 Structure And CUDA Contribution Joint Rule

Future controlled repair should combine:

1. DA3 structure evidence:
   - edge mismatch;
   - relative depth ranking violation;
   - boundary-side inconsistency.
2. CUDA contribution evidence:
   - high `T * alpha`;
   - stable depth order;
   - contributor appears in risk pixels.
3. Counterfactual evidence:
   - weakening improves DA3 structure loss;
   - RGB does not degrade significantly.

Only the intersection of these signals should become high-confidence repair candidates.

## Evaluation Protocol

Even in DA3-unsupervised training, evaluation should remain LiDAR-valid:

- all_valid metrics;
- boundary-band metrics;
- canny / rendered-depth-edge metrics;
- thin-structure metrics;
- stable non-boundary control;
- selected DA3 risk region local metrics;
- RGB MAE / PSNR and qualitative crops.

Invalid LiDAR pixels must never be treated as depth 0.

## Recommended Next Step

Before any repair operation, implement a dry-run candidate scorer:

`DA3 structure counterfactual + CUDA contribution -> bad/good/neutral contributor confidence -> repair candidate tag`

The dry-run scorer should output JSON/NPZ and overlays only. It should not modify Gaussian parameters.
