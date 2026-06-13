# 安全门说明

## 默认关闭

以下能力默认关闭：

- feedback controller；
- guided feedback；
- Gaussian control；
- opacity regularization；
- opacity decay；
- repair dry-run。

## DA3-unsupervised 安全规则

必须满足：

```yaml
train.guided_feedback.use_lidar_depth: false
optim.lambda_depth_lidar: 0.0
train.feedback_controller.risk_source: da3_boundary
```

日志和 manifest 应明确：

```text
LiDAR supervision disabled; LiDAR is used for evaluation only.
```

## Gaussian repair 安全规则

当前允许：

- protect-only；
- opacity regularization loss；
- opacity_decay_apply；
- prune/shrink/split dry-run candidate tagging。

当前禁止：

- real prune；
- real shrink；
- real split；
- real surface-align。

真实 prune/shrink/split 必须保持：

```python
raise RuntimeError("Real prune/shrink/split is disabled in this release.")
```

## opacity_decay_apply 条件

必须显式配置：

```yaml
train.gaussian_control.control_mode: opacity_decay_apply
train.gaussian_control.allow_parameter_modification: true
train.gaussian_control.allow_real_prune: false
train.gaussian_control.allow_real_split: false
train.gaussian_control.allow_real_shrink: false
```

并限制：

```yaml
max_decay_gaussians_per_trigger: 10
max_decay_ratio: 0.00005
```
