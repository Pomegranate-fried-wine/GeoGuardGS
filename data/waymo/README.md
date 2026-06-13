# Waymo 数据放置说明

本仓库不包含 Waymo Open Dataset 数据。

请在服务器上将预处理后的 Waymo 场景放到：

```text
data/waymo/<scene_id>/
```

每个 view 需要能够被底层 Street Gaussian 数据加载器读取，并且如果要计算 LiDAR-valid 指标，应保留 Waymo 稀疏 LiDAR 的 `dict(mask, value)` 表示。无效 LiDAR 像素不能当作 depth=0。

示例：

```text
data/waymo/002/
  images/
  cameras/
  lidar/
  ...
```

具体结构以当前 Street Gaussian / Waymo 预处理脚本为准。
