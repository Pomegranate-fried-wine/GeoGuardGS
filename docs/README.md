# GeoFeedback-GS Documentation Index

This folder contains protocol notes, server guides, audits, and draft writing materials for GeoFeedback-GS.

## Canonical Current Wording

Use this project name in new documents:

```text
GeoFeedback-GS
GeoFeedback-GS: LiDAR-Reduced Dynamic Street Gaussian Reconstruction with Responsible Gaussian Feedback
GeoFeedback-GS：基于责任高斯反馈的低 LiDAR 依赖动态街景高斯重建
```

Historical names such as `GeoGuardGS`, `VisionAudit-GS`, and `LiDAR-LiteGS` may still appear in older drafts, code paths, package names, or server directory names. They are retained where renaming would break imports, paths, or experiment reproducibility.

## Recommended Reading Order

1. `../README.md`: project overview, commands, and result boundaries.
2. `project_overview.md`: concise research framing.
3. `evaluation_protocol_audit.md`: sampled diagnostic eval versus final full evaluation.
4. `geometry_consistency_eval_protocol_zh.md`: held-out geometry evaluation protocol.
5. `server_a100_experiment_guide.md`: practical A100 running guide.
6. `output_specification.md`: expected training and evaluation output layout.

## Draft / Historical Notes

Some local paper/PPT drafts may contain older names or stronger claims. Treat them as working notes, not canonical repository claims. They should be updated to the GeoFeedback-GS wording before being committed or published.

Tracked historical notes under `research_notes/` are retained for reproducibility and should not be used as the current project homepage.

Before public release, update any draft manuscripts or presentation scripts to the GeoFeedback-GS naming and the bounded interpretation in `README.md`.
