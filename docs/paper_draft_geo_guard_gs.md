# Draft Manuscript: GeoFeedback-GS

Working title options:

1. **Auditing LiDAR dependency in object-aware street Gaussian reconstruction
   with DA3-guided feedback**
2. **Object-aware street Gaussian reconstruction with DA3 feedback under
   reduced LiDAR dependence**
3. **Pure-vision object-aware street Gaussian reconstruction with DA3-guided
   feedback**

## One-sentence argument

In object-aware autonomous-driving scene reconstruction, we show that
training-time LiDAR supervision can be replaced by DA3-guided structure and
feedback signals, and that a pure-vision object-aware variant can retain
competitive held-out RGB rendering quality on the evaluated Waymo scene, with
the boundary that current evidence is single-scene and does not yet establish
general geometry improvement.

## Terminology ledger

- **StreetGS**: the LiDAR-supervised object-aware Street Gaussians baseline.
- **DA3-only**: LiDAR-initialized training with DA3 structure guidance but no
  training-stage LiDAR supervision.
- **DA3+Feedback**: DA3-guided training with periodic group softpatch feedback.
- **PV-C**: pure-vision DA3+Feedback object-aware configuration.
- **LiDAR initialization**: using LiDAR point clouds to initialize background
  and/or object Gaussians.
- **LiDAR supervision**: using LiDAR depth or selected LiDAR pixels as a
  training loss or feedback source.
- **Held-out test split**: final evaluation split with 245 test views.
- **Object-region metrics**: metrics computed only on views/pixels with valid
  object masks; in the current evidence, 84 of 245 held-out views contain valid
  object-region rows.

## Abstract draft

Object-aware neural reconstruction for autonomous driving commonly benefits
from LiDAR geometry, but this dependence complicates claims of image-only or
weakly supervised scene understanding. We study this dependence in the setting
of Street Gaussian reconstruction by separating LiDAR initialization,
training-time LiDAR supervision, DA3-guided structure, and periodic feedback.
Across four formal configurations on a held-out Waymo split, we compare a
LiDAR-supervised StreetGS baseline, a LiDAR-initialized DA3-only variant, a
LiDAR-initialized DA3+Feedback variant, and a pure-vision DA3+Feedback variant
with COLMAP background initialization and random-box object initialization. The
pure-vision configuration achieved 25.63 PSNR and 0.8468 SSIM over 245 held-out
test views, comparable to the LiDAR-initialized StreetGS baseline in this
single-scene evaluation, while passing the available no-LiDAR initialization
and supervision audit. Periodic feedback was successfully triggered throughout
the feedback runs, but the current final RGB metrics do not show a consistent
advantage of feedback over DA3-only. These results support a bounded claim:
object-aware street Gaussian reconstruction can remain viable under reduced
LiDAR dependence, while broader claims about geometric improvement and
cross-scene robustness require additional evaluation.

## Introduction draft

Autonomous-driving reconstruction systems require both static scene fidelity
and object-aware modeling of vehicles and other dynamic actors. Street
Gaussian methods provide an efficient representation for this setting, but
their strongest configurations often rely on LiDAR for initialization,
supervision, or both. This reliance creates an ambiguity when such systems are
described as image-driven: the training loss may be free of LiDAR supervision
while the initial geometry still inherits LiDAR structure.

We address this ambiguity by treating LiDAR use as an experimental variable
rather than a single binary condition. The resulting question is not only
whether LiDAR supervision can be removed, but also whether object-aware
training remains viable when LiDAR initialization is removed. This distinction
is important for comparing methods that claim reduced sensor dependence, and it
is especially important for object branches, where vehicles may disappear if
object modeling is disabled or if initialization is too weak.

GeoFeedback-GS builds on StreetGS with DA3-guided structural signals and a periodic
feedback controller. DA3 provides image-derived depth, edge, and risk cues,
while the feedback controller periodically groups high-risk regions and
activates softpatch supervision. We evaluate this design through four formal
groups: a StreetGS baseline with LiDAR initialization and LiDAR supervision, a
LiDAR-initialized DA3-only variant without training-stage LiDAR supervision, a
LiDAR-initialized DA3+Feedback variant, and a pure-vision DA3+Feedback variant
using COLMAP background initialization and random-box object initialization.

The current evidence shows that the pure-vision object-aware variant can train
and preserve competitive held-out rendering quality in the tested Waymo scene.
It does not yet show that feedback consistently improves full-image metrics
over DA3-only, nor does it establish a final metric geometry advantage. We
therefore frame the contribution as an audited study of LiDAR dependency and a
working pure-vision object-aware DA3 feedback configuration, rather than as a
general superiority claim.

## Methods draft

### Experimental task

The task is object-aware street Gaussian reconstruction from Waymo camera
sequences. Each method produces RGB renderings and rendered depth maps for
held-out test views. Formal evaluation uses the held-out test split with 245
views.

### Baseline and experimental groups

The four formal groups isolate initialization and supervision choices:

- **A: StreetGS baseline** uses LiDAR point-cloud initialization and
  LiDAR-supervised training.
- **B: DA3-only** uses LiDAR initialization but removes training-stage LiDAR
  supervision, replacing it with DA3-guided structure signals.
- **C: DA3+Feedback** uses LiDAR initialization, removes training-stage LiDAR
  supervision, and adds periodic group softpatch feedback.
- **PV-C: Pure-vision DA3+Feedback** removes LiDAR initialization and LiDAR
  supervision. It uses COLMAP for background initialization and random-box
  object initialization while retaining object-aware modeling.

### DA3-guided structure and feedback

DA3-derived depth, edge, and risk maps provide image-based structural cues.
For feedback-enabled groups, a periodic controller identifies risk regions,
estimates responsible Gaussian groups, and activates softpatch masks. The
current implementation records feedback manifests and audit rows to verify
that the feedback loop does not use LiDAR-selected pixels or modify Gaussian
parameters outside the intended supervision path.

### Evaluation

Final quantitative evaluation is performed on the held-out test split at the
final checkpoint. Metrics include PSNR, SSIM, LPIPS, and L1 for full-image
rendering, with additional object-region and background-region CSV outputs.
Training-time PSNR/L1 curves are diagnostic and are not used as the main final
performance table. Fixed-view qualitative panels compare GT RGB, rendered RGB,
and rendered depth across all four groups at matched cameras and frames.

## Results draft

### Held-out rendering performance

On the 245-view held-out test split, PV-C achieved the highest full-image PSNR
among the four formal groups in the current run, with 25.63 PSNR and 0.8468
SSIM. The StreetGS baseline achieved 25.57 PSNR and 0.8456 SSIM. DA3-only and
DA3+Feedback achieved lower full-image PSNR values of 24.38 and 24.33,
respectively. These results indicate that the pure-vision object-aware
configuration is viable in this tested scene, but they do not support a claim
that DA3+Feedback improves full-image rendering over DA3-only.

### LiDAR dependency audit

The audit table separates LiDAR initialization from training-stage LiDAR
supervision. A, B, and C use LiDAR initialization. C and PV-C report no LiDAR
supervision or LiDAR-selected pixels in the available feedback/safety audit.
PV-C is the only formal group that satisfies the current no-LiDAR
initialization and no-LiDAR supervision condition.

### Feedback behavior

The feedback controller completed 59 valid triggers in C and 30 valid triggers
in PV-C. This verifies that the periodic feedback pipeline executed during
formal training. The current metrics, however, should be interpreted as
pipeline viability rather than proof that feedback improves final rendering
quality.

### Qualitative RGB and depth behavior

The generated same-view panels provide 90 fixed-view comparisons across six
iterations and 15 views. These panels show that all four groups retain
object-aware renderings at the selected views. PV-C exhibits competitive RGB
appearance in several inspected views, while its rendered depth can differ
substantially from LiDAR-initialized groups. This supports visual viability but
also motivates final quantitative depth or geometry evaluation before making
strong geometric claims.

## Discussion draft

The main outcome of this study is a clearer separation of LiDAR dependence in
object-aware street Gaussian reconstruction. Removing training-stage LiDAR
supervision is not equivalent to removing LiDAR from the entire pipeline,
because LiDAR initialization can still provide strong geometric priors. By
including PV-C, the current experiments establish a stricter pure-vision
condition in which background and object initialization avoid LiDAR.

The most encouraging result is that PV-C remained competitive with the
LiDAR-initialized StreetGS baseline in full-image held-out RGB metrics. This
suggests that COLMAP background initialization, random-box object
initialization, and DA3 feedback can provide a practical route toward
object-aware reconstruction under reduced sensor dependence. At the same time,
the results caution against overclaiming the role of feedback. The final RGB
metrics do not show C outperforming B, so feedback should currently be framed
as a functioning audited mechanism rather than a proven accuracy booster.

Several limitations remain. The evidence package covers a single formal scene
and split. Loss traces are missing from the downloaded package. Final depth or
geometry metrics are not yet included, so rendered depth panels should be
treated as qualitative. Object-region evaluation applies to the object-present
subset, not all 245 views. Addressing these limitations would strengthen the
paper from a bounded experimental report into a more general method claim.

## Conclusion draft

GeoFeedback-GS provides an audited experimental framework for studying LiDAR
dependency in object-aware street Gaussian reconstruction. In the current
formal evaluation, the pure-vision DA3+Feedback configuration achieved
competitive held-out RGB rendering while avoiding both LiDAR initialization and
LiDAR supervision. The result supports a bounded claim that pure-vision
object-aware reconstruction is feasible in the evaluated setting. Stronger
claims about feedback-driven improvement, geometry accuracy, and cross-scene
robustness require additional evidence.

## Proposed main figures

1. **Figure 1: Method and experiment design.** Schematic showing A/B/C/PV-C,
   LiDAR use boundaries, DA3 signals, and feedback loop.
2. **Figure 2: Held-out test quantitative results.** Main table or bar plot
   for PSNR, SSIM, LPIPS, and L1.
3. **Figure 3: Same-view RGB/depth qualitative comparison.** Select 4-6 views
   from `formal_rgb_depth_comparisons/iter_030000`.
4. **Figure 4: Training diagnostics.** Sampled/full-split PSNR and L1 curves,
   explicitly labeled as diagnostic.
5. **Figure 5: Audit evidence.** Initialization audit and feedback trigger
   timeline.

## Assumptions and missing inputs

- Target journal is not fixed; this draft uses a generic Nature-style argument.
- Loss curves need `train_loss_trace.csv`; currently absent.
- Final quantitative depth/geometry metrics are absent.
- Cross-scene evidence is absent.
- A formal related-work section needs citations selected by the author.
