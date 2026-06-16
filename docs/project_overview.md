# GeoFeedback-GS Project Overview

**English title:** GeoFeedback-GS: LiDAR-Reduced Dynamic Street Gaussian Reconstruction with Responsible Gaussian Feedback

**中文标题：** GeoFeedback-GS：基于责任高斯反馈的低 LiDAR 依赖动态街景高斯重建

## Positioning

GeoFeedback-GS is not a new renderer from scratch. It is a research extension of a Street Gaussians-style object-aware dynamic street reconstruction framework. The central question is:

> How far can object-aware dynamic street Gaussian reconstruction reduce LiDAR dependence while keeping vehicles, held-out RGB quality, and auditable geometry diagnostics?

The project separates LiDAR usage into:

1. **LiDAR initialization**: using LiDAR point clouds to initialize background or object Gaussians.
2. **LiDAR training supervision**: using LiDAR depth as a loss or as selected-pixel risk labels during training.
3. **LiDAR evaluation reference**: using held-out LiDAR only after training to audit geometry.

This separation is the main experimental framing. It avoids the ambiguous claim that a method is simply "LiDAR-free" without specifying which role of LiDAR has been removed.

## Core Method

The GeoFeedback-GS feedback path is:

```text
risk region selection
-> responsible Gaussian group attribution
-> softpatch region weight
-> DA3 relative-structure loss
-> periodic audit and visualization
```

DA3 is used as a **relative structure prior**, not as metric depth ground truth. The implemented DA3 structure loss contains:

- `da3_edge_loss`
- `da3_ranking_loss`
- `da3_side_loss`
- `guided_feedback_da3_structure_loss`

The feedback mechanism is not an independent loss. It produces softpatch/region weights; selected regions with `feedback_weight > 1.0` activate and modulate the DA3 structure loss.

## Formal Groups

| Group | Repository config | Interpretation |
| --- | --- | --- |
| A | `configs/experiments/a100_baseline_streetgs.yaml` | LiDAR-supervised StreetGS reference: LiDAR initialization + LiDAR depth supervision |
| B | `configs/experiments/a100_da3_only.yaml` | No-LiDAR-supervision control: LiDAR initialization, no LiDAR depth supervision; current global mode may not activate DA3 loss |
| C | `configs/experiments/a100_da3_periodic_group_softpatch.yaml` | LiDAR-init GeoFeedback-GS: no LiDAR supervision + feedback-activated DA3 structure loss |
| PV-C | `configs/experiments/a100_pv_da3_feedback_obj.yaml` | LiDAR-free GeoFeedback-GS under pose-and-box supervision: COLMAP background + random-box object init + feedback-activated DA3 structure loss |

PV-C does not read LiDAR point clouds for initialization or training supervision, but it still relies on camera poses, COLMAP/SfM, and object track boxes. This should be stated explicitly in papers and presentations.

## Current Evidence Boundary

Supported:

- Four formal groups can be trained with object-aware dynamic reconstruction enabled.
- C and PV-C exercise the periodic feedback chain and generate audit outputs.
- PV-C can preserve vehicles using random-box object initialization.
- On the current held-out Waymo test split, PV-C reaches RGB PSNR/SSIM close to or slightly above the LiDAR-supervised reference.

Not yet supported:

- A claim that GeoFeedback-GS comprehensively outperforms StreetGS.
- A claim that feedback always improves final RGB metrics over the no-LiDAR-supervision control.
- A claim that absolute metric geometry is solved.
- A claim of cross-scene generalization.

## Main Documents

- `README.md`: repository homepage and runnable commands.
- `docs/README.md`: documentation index and status.
- `docs/evaluation_protocol_audit.md`: evaluation protocol notes.
- `docs/geometry_consistency_eval_protocol_zh.md`: geometry-consistency evaluation design.
- `docs/server_a100_experiment_guide.md`: server-side experiment guide. Some server paths may still contain historical `GeoGuardGS` directory names.
- `docs/chinese_manuscript_draft_geoguardgs.md`: Chinese manuscript draft; should be updated to GeoFeedback-GS wording before publication.
