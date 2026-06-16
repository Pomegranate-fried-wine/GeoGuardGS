# 方法说明

## 1. Risk source

GeoFeedback-GS 支持两类风险来源：

- `da3_boundary`：DA3-unsupervised 主线，使用 DA3 depth edge、rendered edge mismatch、relative ranking 和 boundary-side structure。
- `lidar_error`：LiDAR-supervised reference，只作为 upper-bound 参考。

## 2. Live contribution

在 selected high-risk pixels 上调用 CUDA debug 接口，记录：

- stable Gaussian id；
- view-local id；
- alpha；
- transmittance；
- contribution weight `T * alpha`；
- depth；
- depth order；
- support count。

责任分数不再只依赖 screen-space overlap，而是基于真实 rasterizer contribution。

## 3. Boundary-aware group responsibility

DA3 分支不做单 Gaussian hard bad/good 判定，而是在 boundary-risk patch 内聚合 Gaussian group：

```text
score = T*alpha * DA3Risk * EdgeMismatch * SupportFactor
```

group 标签包括：

- `bad_boundary_mixing_group`
- `bad_edge_blurring_group`
- `bad_ranking_conflict_group`
- `good_boundary_support_group`
- `rgb_protect_group`
- `neutral_group`
- `low_evidence_group`

## 4. Periodic feedback

训练每隔 `interval` 轮触发一次：

```text
current model -> render diagnostic views -> build risk map
-> live CUDA contribution -> group responsibility
-> softpatch feedback -> continue training
```

## 5. Gaussian control

控制接口统一读取 `GroupEvidence`，不关心证据来自 LiDAR 还是 DA3。

当前允许：

- protect-only；
- opacity regularization loss；
- config-gated opacity decay；
- repair dry-run candidate tagging。

当前禁止：

- real prune；
- real shrink；
- real split。
