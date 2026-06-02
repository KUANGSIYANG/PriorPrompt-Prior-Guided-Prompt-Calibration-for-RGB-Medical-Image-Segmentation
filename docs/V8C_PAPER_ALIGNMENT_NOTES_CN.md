# SafePrompt-BED / V8C 论文对照修订文档

本文只对照 `draft (2)(1).pdf` 与当前 `paper_release_v8c` 中已经进入 V8C/SafePrompt-BED 主线的代码、配置、指标和 3-seed 证据。本文不引入 BB2v2、NativePrompt、V11/V12/V13、IDRiD/DFUC 或其他未进入当前主线的研究分支。

当前建议：论文主线继续使用论文名 `SafePrompt-BED`，模块名使用 `GPC` 与 `BED`。代码里的 `SafeAnchor-GPC-v8c`、`BB2` 可以作为实现名或内部别名，不建议在论文正文中作为主方法名反复出现。

---

## 1. 总体结论

当前 PDF 的大方向已经和 V8C 主线基本一致：

```text
nnU-Net probability prior
-> Generalizable Prior Calibrator (GPC)
-> frozen Boundary Evidence Decomposition (BED) / MobileSAM bridge
-> single-pass final mask
```

但如果按 CCFA/强会议审稿标准看，还需要修正 5 个地方：

| 优先级 | PDF 当前问题 | 推荐动作 |
|---|---|---|
| 必改 | 仍是 single-seed 数字，没有写 3-seed mean/std | 把 Abstract、Results、Conclusion 中的主结果更新为 last-checkpoint 3-seed mean/std |
| 必改 | Implementation 写 Kvasir 100 epochs、PH2 30 epochs | 改成每个数据集独立训练 nnU-Net 至收敛并导出 frozen probability priors，不强调 epoch 不一致 |
| 必改 | Training Objective 写成 “six active terms” | 改成 “implementation computes six weighted terms, but under NoResidual only final/box/bin are GPC-dependent for prompt parameters” |
| 必改 | Ablation 说 PH2 ablation 未报告 | 删除这句，补 PH2 消融表，并解释 PH2 上 fixed policy 的小幅 Dice 波动 |
| 建议 | Abstract/Discussion 把 BED 说得略像大增益核心 | 改成 frozen boundary response basis，核心贡献是 GPC 学习 prompt-level policy |

修完这些后，论文叙事会更干净：不是“堆模块”，而是“冻结强 prior 与 frozen SAM bridge，只学习最小 prompt policy”。

---

## 2. 当前代码事实核对

### 2.1 方法名与模块名

论文名建议：

```text
SafePrompt-BED: Prior-Guided Prompt Calibration for RGB Medical Image Segmentation
```

论文模块名建议：

```text
GPC = Generalizable Prior Calibrator
BED = Boundary Evidence Decomposition
```

代码实现别名：

```text
SafeAnchor-GPC-v8c = release implementation name
BB2 = BED adapter implementation alias
```

论文正文中建议统一写 `SafePrompt-BED / GPC / BED`。如果必须说明实现，可以在 reproducibility appendix 中写：

```text
In the released code, BED is implemented by the protected BB2 adapter used by the SafeAnchor bridge.
```

### 2.2 V8C 主线配置

当前主线配置为：

```text
configs/branch_safeanchor_gpc_v8c_paper_release.json
```

关键事实：

```text
use_residual = false
use_temperature = false
use_learned_threshold = true
use_binary_shape_losses = false
use_binary_distance_loss = false
teacher_mode = none
bb2_channels = logit_grad_only
uses_candidate_gate = false
uses_area_gate = false
uses_point_prompt = false
post_safeanchor_module = false
```

所以论文中不要写：

```text
coarse residual refinement
temperature scaling
point prompt
candidate routing
second decode
post-processing refiner
binary shape auxiliary loss
distance auxiliary loss
teacher distillation
```

### 2.3 GPC 真实可学习自由度

GPC 输入是 6-channel prior-conditioned feature：

```text
Z_GPC = concat[
  clipped coarse logit,
  coarse probability,
  coarse uncertainty,
  coarse logit gradient,
  RGB luma,
  image gradient
] in R^{B x 6 x H x W}
```

GPC encoder 是卷积网络，不是纯 MLP：

```text
3x3 Conv
GroupNorm
SiLU
4 x LargeKernelResidualBlock
AdaptiveAvgPool
1x1 Conv
SiLU
1x1 Conv global head
```

V8C NoResidual 下，`Zc = Z0`。GPC 不修 coarse mask，只学习：

```text
alpha: dense prompt fusion coefficient
box_scale: soft-moment box scale
threshold: final binarization threshold
homoscedastic loss weights
```

这三个 prompt-level 自由度正好对应推理链中的三个必要决策：

```text
How much BED evidence should be mixed?
How much spatial context should SAM see?
Where should the final probability be binarized?
```

### 2.4 BED / BB2 使用方式

当前 V8C 主线 active BED evidence 是 `logit + gradient`。

代码兼容旧 checkpoint 时，BB2 adapter 仍可接受 4-channel maps，但 V8C 会通过 channel prune 保留：

```text
channel 0 = logit
channel 2 = gradient
```

并把 band/edge 清零。因此论文中建议写：

```text
BED uses logit-gradient evidence.
```

不要写：

```text
BED uses four active evidence maps.
```

也不要在主表中加入 `BB2 all maps` 消融，因为这已经不是 V8C 主线定义的一部分。

---

## 3. Abstract 需要怎么改

### 3.1 PDF 当前问题

PDF abstract 现在写的是 single-seed last-checkpoint 数字：

```text
On Kvasir-120, it achieves 0.8743 Dice, 15.0907 ASD, and 50.4763 HD95.
On PH2-60, it achieves 0.9386 Dice, 12.6456 ASD, and 40.0622 HD95.
```

这不再是最强证据。现在已经完成 3-seed last-checkpoint full training，应该用 mean ± std。

### 3.2 推荐替换

把 abstract 的结果句替换为：

```text
Experiments on Kvasir-SEG and PH2 show that SafePrompt-BED consistently improves the nnU-Net-only baseline. Under a fixed three-seed last-checkpoint protocol, SafePrompt-BED obtains 0.8746 +- 0.0003 Dice, 15.0576 +- 0.0379 ASD, and 50.3549 +- 0.1139 HD95 on Kvasir-120, and 0.9386 +- 0.0000 Dice, 12.6490 +- 0.0207 ASD, and 40.0727 +- 0.0190 HD95 on PH2-60.
```

如果担心 abstract 太长，可以用短版：

```text
Across three last-checkpoint runs, SafePrompt-BED obtains stable improvements over nnU-Net-only on both Kvasir-120 and PH2-60, with Dice standard deviations of only 0.0003 and 0.0000, respectively.
```

说明：内部结果文件以百分制记录，论文 Dice 用小数。这里的换算为：

```text
Kvasir Dice 87.4561 +- 0.0266 points -> 0.8746 +- 0.0003
PH2 Dice 93.8635 +- 0.0034 points -> 0.9386 +- 0.0000
```

---

## 4. Method 叙事需要怎么改

### 4.1 主线公式建议

论文中应把完整推理链写成下面这种极简形式：

```text
P0 = nnUNet(I)
Z0 = logit(P0)
(alpha, s_b, tau) = GPC_theta(I, Z0)
Zc = Z0
E = BED(Zc)
b = SoftMomentBox(sigmoid(Zc), s_b)
Q = alpha * Zc + (1 - alpha) * E
Y = MobileSAM(I, b, Q)
M_hat = 1[sigmoid(Y) > tau]
```

其中 `Zc = Z0` 是 NoResidual 的核心。它让论文归因非常干净：

```text
SafePrompt-BED does not refine the coarse prior pixel-wise. It preserves the nnU-Net prior as a semantic anchor and calibrates only the prompt-level degrees of freedom required by the frozen BED-MobileSAM bridge.
```

### 4.2 不建议再写的表达

不要写：

```text
coarse deformation
coarse refinement
residual prior correction
failure-aware pixel repair
```

更准确的写法是：

```text
prior-conditioned prompt calibration
identity-preserving prior anchoring
case-wise prompt policy
```

---

## 5. Section 3.7 Training Objective 推荐替换

### 5.1 PDF 当前问题

PDF 当前写法：

```text
The active objective contains six terms ...
We combine the six active terms by homoscedastic uncertainty weighting ...
```

从代码看，V8C 确实计算了 6 个 weighted terms：

```text
L_final
L_prior
L_boundary
L_box
L_anchor
L_bin
```

但在 NoResidual 下：

```text
Zc = Z0
L_anchor = SmoothL1(Zc, Z0) = 0
L_prior and L_boundary are constants with respect to GPC prompt parameters
```

所以如果直接写 “six active terms”，审稿人可能会误以为这 6 个 loss 都在训练一个 coarse refiner。更准确的是：代码计算 6 项，但对 GPC prompt policy 有直接监督作用的是 final / box / binarization；prior / boundary / anchor 是 identity-prior regularization and diagnostic terms，主要约束论文叙事与 loss weighting 的审计口径。

### 5.2 推荐替换英文

建议把 3.7 整段替换为：

```text
During the main training stage, only the GPC parameters and the homoscedastic loss weights are optimized. The upstream nnU-Net probability maps, the BED adapter, and MobileSAM remain frozen. The V8C configuration follows an identity-prior design, i.e., Zc = Z0, and therefore does not learn a pixel-level residual correction of the nnU-Net prior.

The implementation computes six weighted objective terms:

L_final = DiceBCE(Y, M),
L_prior = DiceBCE(Zc, M),
L_boundary = SmoothL1(||grad sigma(Zc)||, ||grad M||),
L_box = 1 - IoU(b, b_GT),
L_anchor = SmoothL1(Zc, Z0),
L_bin = DiceBCE(k (sigma(Y) - tau), M).

Under the NoResidual constraint, L_prior, L_boundary, and L_anchor act as identity-prior regularization and diagnostic terms: they verify that the upstream prior is preserved rather than learned as an additional coarse-mask refiner. The GPC-dependent prompt-policy supervision is provided by the final mask term, the soft-moment box term, and the differentiable learned-threshold binarization term. The six implemented terms are combined using homoscedastic uncertainty weighting,

L = sum_i exp(-s_i) L_i + s_i.

The reported mainline excludes teacher distillation, binary shape supervision, binary distance supervision, temperature scaling, point prompts, candidate routing, second decoding, and post-processing. Overall, SafePrompt-BED learns a compact case-wise prompt policy while keeping the upstream prior, BED adapter, and MobileSAM decoder fixed.
```

注意这里不要说 “only three losses are used”，因为代码确实有 6 个 loss slots。准确说法是：

```text
six implemented weighted terms, three prompt-policy dependent terms under NoResidual.
```

---

## 6. Section 4.3 Implementation Details 推荐替换

### 6.1 PDF 当前问题

PDF 当前写：

```text
The upstream nnU-Net models are trained for 100 epochs on Kvasir-SEG and 30 epochs on PH2...
```

这句最好删。原因不是它一定假，而是论文观感不好：审稿人会问为什么两个数据集 epoch 不一致，是否对 PH2 欠训练或调参。当前项目讨论中已经确定主线口径应是 “dataset-specific converged nnU-Net priors”。

### 6.2 推荐替换英文

建议替换为：

```text
For each dataset, an nnU-Net model is trained independently on the corresponding training split until convergence, and its foreground probability maps are exported as frozen upstream priors. These probability maps are not updated during SafePrompt-BED training or inference. Dataset-specific items are limited to the training images, labels, exported nnU-Net probabilities, the frozen BED/MobileSAM bridge checkpoint, and the GPC checkpoint; the module graph and inference path are identical for Kvasir-SEG and PH2.

During the main GPC training stage, only GPC and the homoscedastic loss weights are optimized. GPC is trained for 3 full epochs with batch size 1, and all reported SafePrompt-BED results use checkpoint_last. We do not use validation-best checkpoint selection, checkpoint averaging, checkpoint soup, or post-hoc morphological processing. To assess whether the last-checkpoint protocol is stable, we additionally train the complete GPC stage with three random seeds and report mean and standard deviation.
```

这段能同时回答：

```text
为什么 batch size = 1
为什么 last checkpoint
为什么不用 validation-best
为什么 Kvasir/PH2 epoch 不一致不应写成主叙事
为什么 3 epochs 不是调参峰值
```

---

## 7. Baselines and Protocol 需要更严谨的地方

PDF 当前把外部 promptable baselines 写成使用 ground-truth box prompt。这个可以保留，但必须清楚说这是 upper-bound prompt protocol，而不是和 SafePrompt-BED 完全同等自动设置。

建议增加一句：

```text
For promptable external baselines, the ground-truth box protocol should be interpreted as an oracle-prompt reference rather than a fully automatic setting. SafePrompt-BED, in contrast, constructs its box prompt automatically from the nnU-Net prior through the learned soft-moment box policy.
```

这样能避免审稿人质疑：

```text
为什么 baseline 用 GT box，而你的方法不用 GT box？
```

我们的叙事是：外部 SAM 类 baseline 是 oracle prompt reference；SafePrompt-BED 是 automatic prior-guided pipeline。

---

## 8. Results 建议更新为 3-seed

### 8.1 主结果表推荐写法

论文主结果表可以同时保留 single-seed 与 three-seed，但最干净的写法是主表用 mean ± std：

| Method | Kvasir-120 Dice | Kvasir-120 ASD | Kvasir-120 HD95 | PH2-60 Dice | PH2-60 ASD | PH2-60 HD95 |
|---|---:|---:|---:|---:|---:|---:|
| nnU-Net-only | 0.8614 | 17.4308 | 65.5075 | 0.9264 | 24.3656 | 78.9434 |
| SafePrompt-BED | 0.8746 +- 0.0003 | 15.0576 +- 0.0379 | 50.3549 +- 0.1139 | 0.9386 +- 0.0000 | 12.6490 +- 0.0207 | 40.0727 +- 0.0190 |

如果版面紧，可以把 std 放脚注：

```text
SafePrompt-BED is reported as mean +- std over three full 3-epoch runs using checkpoint_last.
```

### 8.2 结果段落推荐替换

```text
SafePrompt-BED consistently improves the nnU-Net-only baseline on both datasets. On Kvasir-120, it improves Dice from 0.8614 to 0.8746 and reduces HD95 from 65.5075 to 50.3549. On PH2-60, it improves Dice from 0.9264 to 0.9386 and reduces HD95 from 78.9434 to 40.0727. The improvement is stable across three full-training runs with checkpoint_last, with Dice standard deviations of 0.0003 on Kvasir-120 and 0.0000 on PH2-60.
```

---

## 9. Ablation Study 需要怎么改

### 9.1 PDF 当前问题

PDF 当前写：

```text
PH2 ablation is not reported because the complete evidence set is currently available only for Kvasir-120.
```

这句必须删除。现在 PH2 消融已经有完整结果。如果不删，会让证据链显得落后，而且和 release 文档不一致。

### 9.2 消融主表保留哪些

为了论文好看且 CCFA 级别干净，只保留 V8C 内部必要组件：

```text
nnU-Net-only
Fixed prompt policy
w/o learned box
w/o learned threshold
w/o BED response
SafePrompt-BED full
BED-only prompt sanity check
```

不要写进主消融表：

```text
BB2 all maps
BB2v2
NativePrompt
V11/V12/V13
area gate
candidate routing
point prompt
second decode
post-refiner
```

原因：这些不是当前 V8C 主线。写进去会让论文像实验日志，而不是一个干净方法。

### 9.3 Kvasir ablation 推荐表

论文表格用小数 Dice：

| Variant | Changed component | Dice | ASD | HD95 |
|---|---|---:|---:|---:|
| nnU-Net-only | remove SafePrompt-BED | 0.8614 | 17.4308 | 65.5075 |
| Fixed prompt policy | fixed alpha, threshold box, fixed threshold | 0.8697 | 15.1478 | 51.9354 |
| w/o learned box | use threshold box | 0.8710 | 15.1155 | 52.1832 |
| w/o learned threshold | use fixed threshold 0.6 | 0.8727 | 15.1514 | 50.9994 |
| w/o BED response | use coarse prompt only | 0.8734 | 15.2086 | 51.4225 |
| SafePrompt-BED | full model | 0.8743 | 15.0907 | 50.4763 |
| BED-only prompt | remove coarse semantic prompt | 0.8424 | 20.5159 | 62.3460 |

注意：如果主结果表改为 3-seed mean/std，消融表仍可以使用 fixed seed single-run checkpoint_last。建议表注写：

```text
Ablation variants are evaluated using the fixed mainline checkpoint protocol. The multi-seed stability of the full model is reported separately.
```

### 9.4 PH2 ablation 推荐表

| Variant | Changed component | Dice | ASD | HD95 |
|---|---|---:|---:|---:|
| nnU-Net-only | remove SafePrompt-BED | 0.9264 | 24.3656 | 78.9434 |
| Fixed prompt policy | fixed alpha, threshold box, fixed threshold | 0.9404 | 12.6929 | 40.1126 |
| w/o learned box | use threshold box | 0.9406 | 12.7853 | 41.3687 |
| w/o learned threshold | use fixed threshold 0.6 | 0.9370 | 12.5532 | 39.7670 |
| w/o BED response | use coarse prompt only | 0.9403 | 12.4907 | 40.3036 |
| SafePrompt-BED | full model | 0.9386 | 12.6456 | 40.0622 |
| BED-only prompt | remove coarse semantic prompt | 0.8534 | 30.5792 | 74.6701 |

### 9.5 PH2 消融该怎么解释

PH2 上不要硬说每个组件都单调提升 Dice。真实结果更细：

```text
Full model clearly improves nnU-Net-only.
Fixed prompt policy and w/o learned box have slightly higher Dice on PH2.
Full model remains balanced and avoids returning to hand-crafted prompt constants.
BED-only fails badly, so the semantic prior is essential.
```

推荐论文解释：

```text
The PH2 ablation shows a different but informative pattern. Because PH2 lesions are larger and morphologically more regular, several conservative prompt variants obtain slightly higher Dice than the fully learnable policy on this small split. However, all SafePrompt-BED variants substantially outperform nnU-Net-only in surface metrics, and the BED-only prompt fails severely. This supports our central design choice: nnU-Net should remain the semantic anchor, while GPC provides a compact learnable prompt policy for interacting with the frozen BED-MobileSAM bridge. We therefore keep the same learnable architecture for both datasets rather than selecting dataset-specific heuristic prompt constants.
```

这段很重要。它把 PH2 “不是每个小模块都涨分” 转化为科研可信度：我们没有按数据集挑手工策略，而是坚持同构架构和可学习策略。

---

## 10. 增加 Multi-seed Stability 小节

建议在 Experiments 或 Discussion 中新增小节：

```text
Stability under Last-Checkpoint Training.
```

推荐正文：

```text
Since the main GPC stage is intentionally short and uses checkpoint_last rather than validation-best selection, we evaluate the stability of the protocol with three random seeds. For each seed, Kvasir and PH2 are trained for the same full 3-epoch GPC schedule, and the last checkpoint is evaluated without checkpoint averaging. SafePrompt-BED obtains 0.8746 +- 0.0003 Dice on Kvasir-120 and 0.9386 +- 0.0000 Dice on PH2-60. The very small standard deviations indicate that the reported improvements are not caused by a favorable random seed or validation-based model selection.
```

推荐表格：

| Dataset | Seeds | Dice | ASD | HD95 |
|---|---:|---:|---:|---:|
| Kvasir-120 | 3 | 0.8746 +- 0.0003 | 15.0576 +- 0.0379 | 50.3549 +- 0.1139 |
| PH2-60 | 3 | 0.9386 +- 0.0000 | 12.6490 +- 0.0207 | 40.0727 +- 0.0190 |

对应 release 证据：

```text
scripts/run_gpc_multiseed_last_checkpoint.ps1
summarize_multiseed.py
outputs/multiseed/safeprompt_bed_multiseed_last_checkpoint.json
outputs/multiseed/safeprompt_bed_multiseed_last_checkpoint.csv
```

---

## 11. Discussion 需要更克制

### 11.1 不要过度声称 BED 一定正增益

Kvasir 中 `w/o BED response` 比 full 差，说明 BED 有帮助。

PH2 中 `w/o BED response` Dice/ASD 略好但 HD95 略差，说明 BED 贡献与数据形态相关。

所以论文中不要写：

```text
BED consistently improves all metrics on all datasets.
```

推荐写：

```text
BED is not intended to replace the semantic prior. It provides a frozen boundary-response basis whose usefulness depends on the image morphology and is controlled by the learned GPC prompt policy.
```

### 11.2 不要过度声称 learned box 在所有数据集都涨 Dice

Kvasir 中 learned box 更强。

PH2 中 threshold box Dice 略高，但 HD95 更差。

推荐写：

```text
The soft-moment box is most beneficial on Kvasir, where polyp boundaries and contextual ambiguity are more variable. On PH2, where lesions are larger and more regular, threshold boxes can be competitive. We nevertheless keep the learned soft-moment box because it removes a hand-crafted threshold-box dependency and provides the same differentiable geometry policy across datasets.
```

### 11.3 不要把 NoResidual 写成弱点

NoResidual 是当前主线优点：

```text
It prevents hidden coarse-mask overfitting.
It makes attribution clean.
It forces improvement to come from prompt calibration.
```

推荐写：

```text
The identity-prior constraint is a deliberate design choice rather than a limitation of capacity. It makes the method less likely to overfit small training splits through pixel-level mask repair and clarifies that the observed gains arise from prompt calibration.
```

---

## 12. 逐段替换清单

### 12.1 Abstract

原文要改的核心：

```text
On Kvasir-120, it achieves 0.8743 Dice...
On PH2-60, it achieves 0.9386 Dice...
```

改成：

```text
Across three last-checkpoint runs, SafePrompt-BED achieves 0.8746 +- 0.0003 Dice, 15.0576 +- 0.0379 ASD, and 50.3549 +- 0.1139 HD95 on Kvasir-120, and 0.9386 +- 0.0000 Dice, 12.6490 +- 0.0207 ASD, and 40.0727 +- 0.0190 HD95 on PH2-60.
```

### 12.2 Implementation Details

原文要删：

```text
trained for 100 epochs on Kvasir-SEG and 30 epochs on PH2
```

改成：

```text
For each dataset, nnU-Net is trained independently on the corresponding training split until convergence, and the exported foreground probability maps are kept frozen as upstream priors.
```

### 12.3 Training Objective

原文要改：

```text
The active objective contains six terms
```

改成：

```text
The implementation computes six weighted terms. Under the NoResidual constraint, the prior, boundary, and anchor terms serve as identity-prior regularization and diagnostics, while the final-mask, soft-box, and learned-threshold terms provide the GPC-dependent supervision for the prompt policy.
```

### 12.4 Ablation

原文要删：

```text
PH2 ablation is not reported because the complete evidence set is currently available only for Kvasir-120.
```

改成：

```text
We report the same component ablation on both Kvasir-120 and PH2-60. The two datasets use the same architecture, training protocol, and inference path, while dataset-specific weights are trained independently.
```

### 12.5 Discussion

建议补一句：

```text
The ablation results indicate that SafePrompt-BED should be interpreted as a compact prompt-policy calibration framework rather than as a collection of independently monotonic modules. Some components have dataset-dependent effects, but the full fixed architecture consistently improves the nnU-Net-only baseline under the same protocol.
```

---

## 13. Camera-ready 版本建议结构

推荐最终论文结构如下：

```text
1. Introduction
2. Related Work
3. Method
   3.1 Overview
   3.2 nnU-Net prior and identity-prior anchoring
   3.3 Generalizable Prior Calibrator
   3.4 Boundary Evidence Decomposition
   3.5 Dense prompt fusion and soft-moment box
   3.6 Single-pass MobileSAM decoding
   3.7 Training objective
4. Experiments
   4.1 Datasets
   4.2 Baselines and protocol
   4.3 Implementation details
   4.4 Metrics
   4.5 Main results
   4.6 Last-checkpoint multi-seed stability
5. Ablation Study
   5.1 Kvasir component ablation
   5.2 PH2 component ablation
   5.3 Cross-dataset ablation interpretation
6. Discussion
7. Conclusion
```

这个结构比当前 PDF 更强，因为它把 multi-seed 和 PH2 ablation 纳入主证据链，同时仍保持方法简洁。

---

## 14. 最终判断

只看 V8C/SafePrompt-BED 当前证据，论文已经具备强主线雏形：

```text
同一架构跨 Kvasir 和 PH2
各数据集独立训练自己的权重
不使用 validation-best 和 checkpoint soup
3-seed last-checkpoint 方差极低
相对 nnU-Net-only 在 Dice 和 surface metrics 上均有正收益
NoResidual 让归因干净
GPC 只学习必要 prompt-level 自由度
```

当前最该做的不是继续加模块，而是把 PDF 改到和代码证据一致。改完上述 5 个必改点后，论文叙事会从 “一个效果不错的工程系统” 上升为 “一个克制、可复现、归因清晰的 prompt calibration 方法”。

