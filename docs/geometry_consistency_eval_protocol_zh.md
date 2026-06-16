# GeoFeedback-GS 几何一致性评估协议

## 目的

该评估模块用于在不重新训练、不修改训练流程的前提下，对 A/B/C/PV-C 四组已训练模型进行 final held-out test split 的几何质量评估。它补充 RGB PSNR/SSIM/LPIPS，重点检查 rendered depth 是否与 held-out LiDAR、DA3 相对深度、RGB/DA3 边缘和 object 区域一致。

脚本：

`scripts/evaluate_geometry_consistency.py`

默认评估：

- split: `test`
- depth: `expected_depth = raw_rendered_depth / (accumulation + eps)`
- LiDAR: 作为 held-out geometry reference，只用于评价，不代表训练使用 LiDAR。
- DA3: 默认关闭；使用 `--enable-da3` 时运行 DA3 并计算 scale-shift aligned 相对一致性。

## 当前能否基于已训练模型评估

可以。只要服务器仍保留：

1. 四组配置文件；
2. 四组 `model_path` 下的 30000 iter checkpoint / point cloud；
3. Waymo 数据目录，尤其 `data/waymo/002/lidar_depth/*.npy`；
4. held-out test split 设置仍为 `split_train=-1`, `split_test=4`；

就可以直接评估，不需要重新训练。

如果只下载了 `outputs/final_evaluation_test_only_v2` 的 RGB 指标和 jpg panels，本地通常不足以计算完整几何指标，因为缺少每个 held-out view 的数值 rendered depth、accumulation、LiDAR depth 和 object masks。因此建议在 A100 服务器上直接运行该脚本。

## 已实现指标

### 1. LiDAR 绝对深度指标

在 `data/waymo/002/lidar_depth/<image_name>.npy` 的有效像素上计算：

- `lidar_absrel`
- `lidar_rmse`
- `lidar_mae`
- `lidar_delta1`
- `lidar_valid_pixel_count`

该组指标是真实几何参考，但只在 sparse LiDAR projected valid pixels 上成立。

### 2. DA3 相对深度一致性

使用 per-view scale-shift alignment：

`D_da3_aligned = s * D_da3 + b`

然后计算：

- `da3_absrel_aligned`
- `da3_rmse_aligned`
- `da3_mae_aligned`
- `da3_spearman`
- `da3_order_accuracy`

注意：这不是绝对几何精度，只能表述为“与视觉几何先验的相对一致性”。

DA3 默认不运行。需要服务器具备 DA3 权重和依赖，并添加：

`--enable-da3`

### 3. Depth edge / RGB edge / DA3 edge 对齐

脚本从 rendered expected depth 提取 depth edge，从 GT RGB 提取 RGB edge；若启用 DA3，则从 aligned DA3 depth 提取 DA3 edge。

输出：

- `depth_edge_f1_rgb`
- `edge_precision_rgb`
- `edge_recall_rgb`
- `edge_chamfer_rgb`
- `depth_edge_f1_da3`
- `edge_precision_da3`
- `edge_recall_da3`
- `edge_chamfer_da3`

### 4. Object / background region metrics

脚本复用 `camera.guidance["obj_bound"]` 作为 object mask，分别输出：

- `full_image`
- `object_region`
- `background_region`

每个 scope 都独立统计 LiDAR、DA3 和 edge metrics。这样可以避免 full-image 指标掩盖车辆区域问题。

额外输出：

- `object_boundary_depth_jump_consistency`

该指标衡量 object mask 边界附近是否存在 rendered depth jump，用于诊断车辆边界是否被抹平。

### 5. Feedback local region metrics

对于 C 和 PV-C，脚本会尝试从 `<model_path>/feedback_controller/iter_*` 读取最新 risk/mask `.npy` 和 manifest/audit JSON，生成：

- `selected_region` scope
- `selected_pixel_count`
- `responsible_gaussian_group_count`
- selected region 上的 LiDAR/DA3/edge metrics

如果没有 before/after checkpoint 或中间 eval，不能把它解释为 feedback 的因果提升；只能称为 final-local diagnostic。

## 输出结构

默认输出：

```text
outputs/geometry_eval/
  a100_baseline_streetgs/
    per_view_geometry_metrics.csv
    summary_geometry_metrics.csv
    geometry_eval_manifest.json
    visualization_panels/test/*.jpg
  a100_da3_only/
  a100_da3_periodic_group_softpatch/
  a100_pv_da3_feedback_obj/
  compare_geometry_summary.csv
  compare_geometry_summary_test.csv
```

## `compare_geometry_summary.csv` 字段说明

基础字段：

- `experiment`: 实验配置名。
- `scope`: `full_image` / `object_region` / `background_region` / `selected_region`。
- `split`: 默认 `test`。
- `view_count`: 有效参与 summary 的 view 数。

每个 metric 都有：

- `<metric>_mean`
- `<metric>_median`
- `<metric>_std`
- `<metric>_min`
- `<metric>_max`
- `<metric>_valid_count`

关键 metric：

- `lidar_absrel`: LiDAR valid pixels 上的绝对相对误差，越低越好。
- `lidar_rmse`: LiDAR valid pixels 上的 RMSE，越低越好。
- `lidar_mae`: LiDAR valid pixels 上的 MAE，越低越好。
- `lidar_delta1`: 深度比例误差小于 1.25 的比例，越高越好。
- `da3_absrel_aligned`: scale-shift aligned 后与 DA3 的相对误差，越低表示越接近 DA3 相对几何。
- `da3_spearman`: rendered depth 与 DA3 depth 的 rank correlation，越高越好。
- `da3_order_accuracy`: 随机像素对前后顺序一致率，越高越好。
- `depth_edge_f1_rgb`: rendered depth edge 与 RGB edge 的 F1。
- `edge_chamfer_rgb`: depth edge 与 RGB edge 的 Chamfer distance，越低越好。
- `object_boundary_depth_jump_consistency`: object 边界附近 depth jump 支持比例，越高通常说明车辆边界更清楚。

## 推荐服务器命令

先跑不带 DA3 的 held-out LiDAR geometry evaluation：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_geometry_consistency.py \
  --configs \
    configs/experiments/a100_baseline_streetgs.yaml \
    configs/experiments/a100_da3_only.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch.yaml \
    configs/experiments/a100_pv_da3_feedback_obj.yaml \
  --output-root outputs/geometry_eval_test_only_v1 \
  --loaded-iter 30000 \
  --splits test \
  --max-panels-per-split 12
```

如果确认 DA3 权重和依赖可用，再跑 DA3 相对一致性：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_geometry_consistency.py \
  --configs \
    configs/experiments/a100_baseline_streetgs.yaml \
    configs/experiments/a100_da3_only.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch.yaml \
    configs/experiments/a100_pv_da3_feedback_obj.yaml \
  --output-root outputs/geometry_eval_test_only_da3_v1 \
  --loaded-iter 30000 \
  --splits test \
  --enable-da3 \
  --max-panels-per-split 12
```

快速 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_geometry_consistency.py \
  --configs configs/experiments/a100_pv_da3_feedback_obj.yaml \
  --output-root outputs/geometry_eval_smoke \
  --loaded-iter 30000 \
  --splits test \
  --max-views-per-split 5 \
  --max-panels-per-split 5 \
  --overwrite
```

## 结果解释边界

- LiDAR 指标：可以写成 held-out sparse LiDAR geometry evaluation。
- DA3 aligned 指标：只能写成 DA3-relative geometry consistency，不能写成真实深度精度。
- Edge 指标：是边界几何诊断，受 RGB texture edge、object mask 质量和 depth edge threshold 影响。
- selected region 指标：没有 before/after 时，只能作为 final-local diagnostic，不能宣称 feedback 因果提升。
