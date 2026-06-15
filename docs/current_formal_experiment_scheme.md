# Current Formal Experiment Scheme

## Hard rule

All formal street-scene experiments after this revision must keep vehicles:

```yaml
model.nsg.include_obj: true
```

Configs with `include_obj=false` are static-background ablations only and must
not be used as the main autonomous-driving street-scene results.

## Main full-scene groups

| Group | Config | Object branch | Initialization | LiDAR training supervision | Supported conclusion |
| --- | --- | --- | --- | --- | --- |
| A | `configs/experiments/a100_baseline_streetgs.yaml` | on | StreetGS-style COLMAP + LiDAR/object init | yes, `lambda_depth_lidar=0.1` | Original StreetGS baseline reproduction |
| B | `configs/experiments/a100_da3_only.yaml` | on | StreetGS-style COLMAP + LiDAR/object init | no | DA3-only unsupervised structure signal under the original full-scene initialization |
| C | blocked until implemented | on | COLMAP-only, no LiDAR init | no | Strict no-LiDAR-init DA3-only full-scene test; requires non-LiDAR object initialization |
| D | `configs/experiments/a100_da3_periodic_group_softpatch.yaml` | on | StreetGS-style COLMAP + LiDAR/object init | no | Main DA3 + periodic group softpatch feedback with vehicles retained |
| E | blocked until implemented | on | COLMAP-only, no LiDAR init | no | Main DA3 + feedback strict no-LiDAR-init full-scene test; requires non-LiDAR object initialization |
| PV-C | `configs/experiments/a100_pv_da3_feedback_obj.yaml` | on | COLMAP background + random-box object init | no | Pure-vision object-aware DA3+Feedback prototype/formal comparison |

Use A as the baseline. Compare B/D against A to decide whether the method can
replace LiDAR training supervision while preserving vehicles under the original
StreetGS initialization protocol. Compare PV-C against D to quantify the cost
and promise of removing LiDAR initialization while retaining object modeling.

C/E are valid scientific questions, but they are not currently valid full-scene
configs in this codebase: the existing strict no-LiDAR initialization path is
COLMAP background-only and disables object Gaussians. Because all formal groups
must retain vehicles, C/E need a new non-LiDAR object initialization path before
they can be run as final full-scene experiments.

## Optional strict no-LiDAR-initialization ablation

These groups are retained only as static-background ablations because the
current no-LiDAR initialization path disables object Gaussians:

| Group | Config | Object branch | Initialization | LiDAR training supervision |
| --- | --- | --- | --- | --- |
| A-static | `configs/experiments/a100_static_bg_baseline_no_lidar_init.yaml` | off | COLMAP only | no |
| B-static | `configs/experiments/a100_static_bg_da3_only_no_lidar_init.yaml` | off | COLMAP only | no |
| C-static | `configs/experiments/a100_static_bg_da3_periodic_group_softpatch_no_lidar_init.yaml` | off | COLMAP only | no |

If these static-background results are strong, they support only the narrower
claim that COLMAP-only background reconstruction can work without LiDAR
initialization. They do not prove full autonomous-driving street-scene
reconstruction because vehicles are disabled.

## Recommended server commands

Run config checks first:

```bash
python scripts/check_closed_loop_config.py --config configs/experiments/a100_baseline_streetgs.yaml
python scripts/check_closed_loop_config.py --config configs/experiments/a100_da3_only.yaml
python scripts/check_closed_loop_config.py --config configs/experiments/a100_da3_periodic_group_softpatch.yaml
```

Then train:

```bash
python scripts/train.py --config configs/experiments/a100_baseline_streetgs.yaml
python scripts/train.py --config configs/experiments/a100_da3_only.yaml
python scripts/train.py --config configs/experiments/a100_da3_periodic_group_softpatch.yaml
```

Build paper evidence after training:

```bash
python scripts/build_paper_evidence_pack.py --output-root outputs/a100_main_experiments --paper-dir outputs/paper_evidence
python scripts/build_paper_result_visuals.py --paper-dir outputs/paper_evidence --out-dir outputs/paper_results
python scripts/build_paper_training_gallery.py --output-root outputs/a100_main_experiments --out-dir outputs/paper_results/training_gallery --copy-assets
```

## Paper framing decision

If B/C approach A while using no LiDAR training supervision, the clean claim is:

```text
The method removes LiDAR training supervision while retaining the original
StreetGS full-scene initialization protocol.
```

Do not claim strict no-LiDAR initialization for B/C, because vehicles currently
depend on the StreetGS-style LiDAR/object initialization path.

## Full-scene v2 output commands

For the next full-scene rerun, use new paper output directories to avoid
overwriting the previous round:

```bash
python scripts/build_paper_evidence_pack.py --output-root outputs/a100_main_experiments --paper-dir outputs/paper_evidence_full_scene_v2
python scripts/build_paper_result_visuals.py --paper-dir outputs/paper_evidence_full_scene_v2 --out-dir outputs/paper_results_full_scene_v2
python scripts/build_paper_training_gallery.py --output-root outputs/a100_main_experiments --out-dir outputs/paper_results_full_scene_v2/training_gallery --copy-assets
```

Required outputs:

1. Curves in `outputs/paper_results_full_scene_v2/plots/`: PSNR mean/median,
   eval L1, total training loss, training RGB L1 loss, DA3 structure loss when
   present, and LiDAR depth loss when present.
2. Fixed-view RGB/depth panels in
   `outputs/paper_results_full_scene_v2/training_gallery/`. Training now
   defaults to 15 fixed views, 5 cameras x 3 frames, saved every 1000
   iterations through `periodic_eval`.
3. Audit and metric tables in `outputs/paper_evidence_full_scene_v2/tables/`
   and compact paper tables in `outputs/paper_results_full_scene_v2/tables/`.

If exact hand-picked views are needed, set:

```yaml
train:
  periodic_eval_view_ids: [...]
```

Otherwise the loader chooses the first three training views per camera and
keeps that list fixed for the entire run.
