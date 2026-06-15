# Experiment Matrix Audit

## Definitions

1. No LiDAR training supervision:
   no LiDAR loss, no LiDAR selected pixels, and no LiDAR labels during training.
   This may still use a LiDAR pointcloud initializer unless explicitly forbidden.

2. No LiDAR initialization:
   the initial Gaussian pointcloud must not come from Waymo LiDAR. Use COLMAP,
   image-only, random, or another non-LiDAR initializer.

3. LiDAR-supervised reference:
   LiDAR loss or LiDAR risk can be used as an upper-bound reference.

## Formal no-LiDAR matrix

The formal no-LiDAR claims must use COLMAP/SfM initialization:

| Group | Config | Initialization | LiDAR training |
| --- | --- | --- | --- |
| A | `configs/experiments/a100_baseline_streetgs.yaml` | COLMAP, no LiDAR init | none |
| B | `configs/experiments/a100_da3_only.yaml` | COLMAP, no LiDAR init | none |
| C | `configs/experiments/a100_da3_periodic_group_softpatch.yaml` | COLMAP, no LiDAR init | none |

These configs set:

```yaml
data.use_colmap: true
data.filter_colmap: true
data.allow_lidar_initialization: false
data.require_no_lidar_initialization: true
model.nsg.include_obj: false
model.nsg.include_sky: false
```

`include_obj=false` is intentional for strict no-LiDAR initialization because
the current object Gaussian initializer is derived from Waymo LiDAR points in
tracked boxes. `include_sky=false` avoids requiring an additional sky
pointcloud file in the COLMAP-only initialization path.

## Debug/reference matrix

The completed `outputs/a100_short_5000` A/B/C runs are LiDAR-init engineering
debug runs. They are useful for stability, DA3, and feedback-controller
debugging, but must not be used as the main no-LiDAR paper conclusion.

| Role | Config |
| --- | --- |
| LiDAR-init debug A | `configs/short_5000/a100_baseline_streetgs_5000.yaml` |
| LiDAR-init debug B | `configs/short_5000/a100_da3_only_5000.yaml` |
| LiDAR-init debug C | `configs/short_5000/a100_da3_periodic_group_softpatch_5000.yaml` |
| LiDAR-init reference | `configs/experiments/a100_lidar_init_streetgs_reference.yaml` |
| LiDAR-supervised upper bound | `configs/experiments/a100_lidar_supervised_reference.yaml` |

## COLMAP short-run expansion

Run in this order before 30000-iteration formal training:

```bash
python scripts/check_colmap_environment.py --config configs/smoke/a100_baseline_streetgs_colmap_smoke.yaml
python scripts/train.py --config configs/smoke/a100_baseline_streetgs_colmap_smoke.yaml
python scripts/train.py --config configs/short_5000/a100_baseline_streetgs_colmap_5000.yaml
python scripts/train.py --config configs/short_5000/a100_da3_only_colmap_5000.yaml
python scripts/train.py --config configs/short_5000/a100_da3_periodic_group_softpatch_colmap_5000.yaml
```

On the current server, prefer:

```bash
export COLMAP_BIN=/data/conda_envs/gaussian_splatting/bin/colmap
```

## Fail-fast audit

`scripts/check_closed_loop_config.py` fails any config with
`data.require_no_lidar_initialization=true` unless it also has COLMAP enabled
and LiDAR initialization disabled.

Training writes:

```text
input_ply/initialization_manifest.json
```

with `uses_lidar_initialization`, `initialization_source`, `pointcloud_source`,
`colmap_binary`, `colmap_point_count`, and `lidar_point_count_used_for_init`.
