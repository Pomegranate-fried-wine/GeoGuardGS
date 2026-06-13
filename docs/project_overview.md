# GeoGuardGS 项目概览

项目名称：

```text
GeoGuardGS: Closed-Loop Geometry-Responsible Gaussian Control for Street Scene Reconstruction
```

中文名称：

```text
GeoGuardGS：面向街景重建的闭环几何责任高斯控制框架
```

## 定位

GeoGuardGS 不是 DA3 depth-prior、boundary-loss 或 loss-engineering 项目。DA3、LiDAR evaluation、geometry error map、responsibility score、candidate tags 和 region clustering 都是基础模块。

当前目标是把几何风险定位、真实 Gaussian contribution、责任 group 归因和周期性训练反馈连接成可审计的闭环系统。

## 技术路线

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

## 研究贡献

1. Geometry-risk-driven responsible Gaussian attribution。
2. 训练态 live selected-pixel CUDA contribution dump。
3. DA3 boundary-aware Gaussian group responsibility。
4. 统一 LiDAR-supervised reference 与 DA3-unsupervised main path。
5. Periodic closed-loop feedback controller。
6. Config-gated conservative Gaussian repair operator。
7. Manifest / audit / visualization / evaluation 输出规范。

## 当前真实启用能力

- DA3 structure feedback。
- LiDAR-valid evaluation。
- live CUDA selected-pixel contribution dump。
- DA3 boundary-aware group responsibility。
- periodic feedback controller。
- group softpatch feedback。
- GaussianControlManager。
- opacity regularization loss。
- config-gated `opacity_decay_apply`。
- prune/shrink/split dry-run candidate tagging。

## 当前禁止能力

- 真实 prune。
- 真实 shrink。
- 真实 split。
- surface-align 真实结构操作。
- DA3-unsupervised 使用 LiDAR 训练监督。
- 把 DA3 depth 当作 metric depth truth。
