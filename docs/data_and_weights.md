# 数据与权重

## Waymo

本仓库不包含 Waymo 数据。请将预处理结果放到：

```text
data/waymo/<scene_id>/
```

LiDAR 必须保留 sparse valid mask。所有 LiDAR 指标只在 `valid_lidar=True` 的像素上计算。

## DA3

本仓库不包含 DA3 权重。推荐使用本地 snapshot：

```text
weights/da3/DA3-LARGE-1.1/
```

配置：

```yaml
geovit:
  enabled: true
  model_dir: weights/da3/DA3-LARGE-1.1
  local_files_only: true
```

DA3 depth 只作为结构先验，不作为 metric depth ground truth。

## Checkpoints

大型 checkpoint 不应提交到 GitHub。请放到：

```text
weights/streetgs/
```

或实验输出目录下的 `outputs/`。
