# 代码审查报告

审查对象为当前 `street_gaussians-main` 中与 GeoFeedback-GS 相关的核心模块。

## 已完成模块

- DA3 Bridge 与 DA3 structure loss。
- LiDAR-valid geometry evaluation。
- Pixel-level geometry error map。
- Screen-space Gaussian responsibility v0/v0.1。
- Multi-view responsibility aggregation v1。
- v1.5 border / thin / layer-conflict diagnostics。
- live selected-pixel CUDA contribution dump。
- DA3 boundary-aware Gaussian group responsibility。
- Periodic feedback controller。
- Unified GaussianControlManager。
- GaussianRepairOperator dry-run。
- Opacity regularization loss。
- Config-gated opacity_decay_apply。

## 当前真实启用能力

- `feedback_controller` 可在训练中周期性触发。
- `contribution_source=live_current_model` 可调用当前模型做 selected-pixel contribution。
- `gaussian_control.control_mode=opacity_regularization` 可作为 loss 项。
- `gaussian_control.control_mode=opacity_decay_apply` 可真实衰减 opacity，但必须显式允许参数修改。
- `repair_dryrun` 只输出候选表，不修改模型。

## 默认关闭能力

- guided feedback。
- feedback controller。
- Gaussian control。
- opacity regularization。
- opacity decay。
- repair dry-run。

## 仍然禁止能力

- real prune。
- real shrink。
- real split。
- real surface-align。
- DA3-unsupervised 使用 LiDAR training loss。

## 风险点

- 部分 copied module 仍保留 `lib.*` legacy import，开源版需要和 StreetGS 代码一起使用，或后续逐步重构 import。
- 当前已将 StreetGS 兼容源码复制到 `third_party/street_gaussian`，`scripts/train.py` 会作为 wrapper 调用该入口。
- `feedback_pipeline_stages.py` 中 DA3 dynamic risk 当前在缺少 DA3 depth 注入时会退化为 rendered-depth edge risk，需要在正式服务器实验中确认 DA3 cache/bridge 被正确接入。
- `run_counterfactual` 默认关闭，正式长训不要每 500 轮直接启用，以免成本过高。
- `opacity_decay_apply` 会真实修改 opacity，必须只在专门实验组开启。

## 已迁移第三方源码

- `third_party/street_gaussian`
- `third_party/diff_gaussian_rasterization`
- `third_party/simple_knn`
- `third_party/simple_waymo_open_dataset_reader`
- `third_party/depth_anything_3`
- `third_party/nvdiffrast`

迁移到 A100 后必须重新编译 CUDA/C++ 扩展，不应复用 Windows 本地编译产物。

## hardcoded path 审查

开源配置已改为相对路径：

- `data/waymo/<scene_id>/`
- `weights/da3/DA3-LARGE-1.1/`
- `outputs/a100_main_experiments/`

迁移前仍需检查 copied legacy scripts 中是否存在本地 `output/local_formal`、`A5000`、`p15` 等路径；这些脚本应作为研究历史入口，不作为 A100 主训练入口。

带本地历史默认路径的脚本已移动到：

```text
scripts/research_archive/
```

正式 A100 入口只使用 `scripts/` 根目录下的通用脚本。

## LiDAR supervision 风险

DA3-unsupervised 必须满足：

- `train.guided_feedback.use_lidar_depth=false`
- `optim.lambda_depth_lidar=0`
- `train.feedback_controller.risk_source=da3_boundary`

已提供 `scripts/check_closed_loop_config.py` 和 `scripts/validate_no_lidar_leakage.py`。

## real repair 误触发风险

`GaussianControlManager` 和 `GaussianRepairOperator` 均对 real prune/shrink/split 设置 runtime error。开源配置也将：

- `allow_real_prune=false`
- `allow_real_split=false`
- `allow_real_shrink=false`

作为默认。
