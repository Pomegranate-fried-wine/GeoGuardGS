# GeoGuardGS

**GeoGuardGS: Closed-Loop Geometry-Responsible Gaussian Control for Street Scene Reconstruction**

**GeoGuardGS：面向街景重建的闭环几何责任高斯控制框架**

GeoGuardGS 是一个面向自动驾驶街景 Gaussian 重建的研究型开源工程。它的目标不是简单把 DA3 depth prior 接入 Street Gaussian，也不是继续堆叠 boundary loss / depth loss，而是构建一条更可解释、更可审计的几何责任诊断与训练反馈路线：

```text
几何风险区域
-> 真实渲染贡献 T * alpha
-> 责任 Gaussian group
-> 周期性训练反馈
-> 保守 Gaussian 控制
-> 安全评估与审计
```

当前版本重点提供一个可迁移到 A100 服务器正式训练的完整工程包：包含 StreetGS 兼容训练入口、DA3 结构反馈、live CUDA selected-pixel contribution dump、周期性 feedback controller、统一 GaussianControlManager、GaussianRepairOperator、安全门检查、A100 实验配置、每 500 轮输出规范和结果收集脚本。

## 1. 项目定位

GeoGuardGS 面向的问题是：在动态街景 Gaussian 重建中，车辆边界、遮挡边界、树干、电线杆、远处薄结构和深度突变处往往出现局部几何不可信。只看全局 RGB 指标或全图平均 depth loss，无法回答：

- 哪些像素区域存在几何风险？
- 哪些 Gaussian 真实参与了这些风险像素的渲染？
- 这些 Gaussian 是边界支撑者、错误混叠者，还是低证据噪声？
- 诊断结果能否周期性反馈回训练，而不是只做离线可视化？

GeoGuardGS 因此把研究重点从“加一个 loss”推进到：

```text
risk map -> contribution capture -> group responsibility -> feedback/control
```

## 2. 方法概览

核心闭环如下：

```text
Street Gaussian training
-> DA3 / LiDAR risk source
-> selected high-risk pixels
-> live CUDA T*alpha contribution dump
-> boundary-aware Gaussian group responsibility
-> unified GroupEvidence
-> periodic feedback controller
-> group softpatch feedback
-> GaussianControlManager
-> GaussianRepairOperator
-> safe opacity decay / dry-run repair candidates
-> evaluation and audit
```

### 2.1 DA3-unsupervised 主线

DA3 在本项目中不是 metric depth ground truth。DA3 主线只使用：

- depth edge；
- boundary-risk；
- rendered / DA3 edge mismatch；
- relative depth ranking；
- boundary-side structure consistency。

DA3-unsupervised 训练阶段必须满足：

```yaml
train.guided_feedback.use_lidar_depth: false
optim.lambda_depth_lidar: 0.0
train.feedback_controller.risk_source: da3_boundary
```

LiDAR 只能用于 evaluation，不参与 DA3 主线训练监督、risk pixel selection 或 bad/good labeling。

### 2.2 LiDAR-supervised reference

LiDAR-supervised 分支是上界参考。它允许使用 sparse LiDAR valid pixels 做训练或风险区域选择，但不能作为 DA3-unsupervised 主方法的结论来源。

### 2.3 Live CUDA contribution

传统 screen-space overlap 只能说明某个 Gaussian 投影覆盖了高误差区域，不能证明它真实参与渲染。GeoGuardGS 使用 selected-pixel CUDA debug dump，在指定风险像素上记录：

- stable Gaussian id；
- view-local id；
- alpha；
- transmittance；
- contribution weight `T * alpha`；
- depth；
- depth order；
- support pixel count。

这让责任归因从“投影相关性”升级为“真实渲染贡献证据”。

### 2.4 Boundary-aware group responsibility

DA3 分支不对单个 Gaussian 做过度激进的 hard bad/good 判定，而是在 DA3 boundary-risk patch 内聚合 Gaussian group。group score 参考：

```text
score = T*alpha * DA3Risk * EdgeMismatch * SupportFactor
```

典型 group 标签：

- `bad_boundary_mixing_group`
- `bad_edge_blurring_group`
- `bad_ranking_conflict_group`
- `good_boundary_support_group`
- `rgb_protect_group`
- `neutral_group`
- `low_evidence_group`

### 2.5 Gaussian control

统一的 GaussianControlManager 读取 GroupEvidence，不区分证据来自 DA3 还是 LiDAR。当前支持：

- protect-only；
- opacity regularization loss；
- config-gated conservative opacity decay；
- prune / shrink / split dry-run candidate tagging。

当前不启用真实 prune / shrink / split。

## 3. 当前版本能力边界

已经补齐：

- StreetGS 兼容训练入口；
- DA3 Bridge / DA3 structure feedback；
- LiDAR-valid evaluation；
- live selected-pixel CUDA contribution dump；
- DA3 boundary-aware group responsibility；
- periodic feedback controller；
- group softpatch feedback；
- unified GroupEvidence；
- GaussianControlManager；
- opacity regularization；
- config-gated opacity_decay_apply；
- GaussianRepairOperator dry-run；
- A100 正式实验配置；
- 每 500 轮输出规范；
- config safety checker；
- LiDAR leakage checker；
- repair safety checker；
- migration package verifier；
- 第三方源码迁移目录。

仍然没有启用：

- real prune；
- real shrink；
- real split；
- real surface-align；
- DA3 多帧动态几何主线；
- SOTA 级最终指标结论。

也就是说，除了真实高斯结构操作 prune / shrink / split / surface-align 之外，当前闭环工程、迁移训练配置和安全检查已经基本补齐。

## 4. 目录结构

```text
GitHub_main/
  configs/
    base/
    experiments/
  docs/
  geoguardgs/
    contribution/
    feedback/
    gaussian_control/
    gaussian_repair/
    evaluation/
    training/
  scripts/
  scripts/research_archive/
  third_party/
    street_gaussian/
    diff_gaussian_rasterization/
    simple_knn/
    simple_waymo_open_dataset_reader/
    depth_anything_3/
    nvdiffrast/
  data/waymo/
  weights/da3/
  weights/streetgs/
  outputs/
```

`scripts/` 根目录是正式迁移训练入口。`scripts/research_archive/` 只保留历史研究脚本，可能包含 A5000 / p15 / local output 默认路径，不作为 A100 正式入口。

## 5. 安装

推荐在 A100 Linux 服务器上执行：

```bash
cd /path/to/GitHub_main
conda env create -f environment.yml
conda activate geoguardgs
pip install -r requirements.txt
```

然后重新编译服务器本地 CUDA/C++ 扩展：

```bash
bash scripts/install_server_extensions.sh
python scripts/check_imports.py
python scripts/verify_migration_package.py
```

必须重新编译的组件：

- `third_party/diff_gaussian_rasterization`
- `third_party/simple_knn`
- `third_party/nvdiffrast`，如果当前配置或依赖使用
- `third_party/simple_waymo_open_dataset_reader`，如果服务器需要该 reader

不要提交本地编译产物，例如 `.so`、`.pyd`、`.dll`、`build/`、`*.egg-info/`。

## 6. 数据与权重

本仓库不包含 Waymo 数据、DA3 权重和大型 checkpoint。

请手动准备：

```text
data/waymo/<scene_id>/
weights/da3/DA3-LARGE-1.1/
weights/streetgs/<optional_checkpoint>/
```

Waymo LiDAR 必须保留 sparse valid mask。所有 LiDAR 指标只能在 `valid_lidar=True` 像素上计算，不能把 invalid LiDAR 当作 depth 0。

## 7. 正式实验配置

当前提供：

- `configs/experiments/a100_baseline_streetgs.yaml`
- `configs/experiments/a100_da3_only.yaml`
- `configs/experiments/a100_da3_periodic_group_softpatch.yaml`
- `configs/experiments/a100_da3_periodic_group_softpatch_opacity_reg.yaml`
- `configs/experiments/a100_da3_periodic_group_softpatch_opacity_decay.yaml`
- `configs/experiments/a100_lidar_supervised_reference.yaml`
- `configs/experiments/a100_hybrid_reference.yaml`

每个配置默认遵守安全门。DA3 主线不使用 LiDAR training loss。真实 prune / shrink / split 全部关闭。

## 8. 启动实验

先做配置检查：

```bash
python scripts/check_closed_loop_config.py \
  --config configs/experiments/a100_da3_periodic_group_softpatch.yaml

python scripts/validate_no_lidar_leakage.py \
  configs/experiments/a100_da3_periodic_group_softpatch.yaml

python scripts/validate_repair_safety.py \
  configs/experiments/a100_da3_periodic_group_softpatch_opacity_decay.yaml
```

并行 dry-run：

```bash
python scripts/launch_a100_experiments.py \
  --configs \
    configs/experiments/a100_baseline_streetgs.yaml \
    configs/experiments/a100_da3_only.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch_opacity_reg.yaml \
    configs/experiments/a100_da3_periodic_group_softpatch_opacity_decay.yaml \
    configs/experiments/a100_lidar_supervised_reference.yaml \
  --gpus 0,1,2,3 \
  --output-root outputs/a100_main_experiments \
  --dry-run
```

正式运行去掉 `--dry-run`。

## 9. 每 500 轮输出

正式 A100 配置中，每 500 轮应保存：

- rendered RGB；
- GT RGB；
- RGB error；
- rendered depth；
- DA3 depth；
- LiDAR sparse overlay；
- depth error map；
- DA3 boundary-risk map；
- selected risk regions；
- contribution top-K overlay；
- group responsibility overlay；
- Gaussian control summary；
- repair safety manifest；
- RGB metrics；
- LiDAR-valid geometry metrics；
- DA3 structure metrics；
- responsibility/control metrics。

详细规范见 `docs/output_specification.md`。

## 10. 核心文档

- `docs/project_overview.md`：项目定位与技术路线。
- `docs/method.md`：方法设计。
- `docs/server_a100_experiment_guide.md`：A100 正式实验说明。
- `docs/output_specification.md`：输出与指标规范。
- `docs/safety_gates.md`：安全门。
- `docs/code_audit_report.md`：代码审查。
- `docs/data_and_weights.md`：数据与权重。
- `docs/references.md`：参考文献占位。

## 11. 快速自检

提交 GitHub 或迁移服务器前运行：

```bash
python scripts/verify_migration_package.py
python scripts/check_closed_loop_config.py --config configs/experiments/a100_da3_periodic_group_softpatch.yaml
python scripts/validate_no_lidar_leakage.py configs/experiments/a100_da3_periodic_group_softpatch.yaml
python scripts/validate_repair_safety.py configs/experiments/a100_da3_periodic_group_softpatch_opacity_decay.yaml
```

如果这些检查失败，不要启动正式训练。

## 12. License 与第三方声明

GeoGuardGS 主体代码使用 MIT License。第三方代码、数据集和模型权重可能有独立许可证。请在公开发布前核对：

- Street Gaussian；
- Depth Anything 3；
- diff-gaussian-rasterization；
- nvdiffrast；
- Waymo Open Dataset。

精确引用信息请在论文提交前核验，当前 `docs/references.md` 中保留了 TODO 占位。
