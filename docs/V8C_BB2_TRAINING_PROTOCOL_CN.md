# SafeAnchor-GPC-v8c 的 BB2 训练审计与 CCFA 级主线决策

Date: 2026-05-27

本文专门回答一个问题：在当前 `SafeAnchor-GPC-v8c NoResidual` 主线里，BB2 还应不应该继续训练、如何训练才算科研规范、以及训练后是否有资格进入 CCFA 级主线。

结论先写在前面：

```text
BB2 训练已经被充分复查。
在当前数据规模和 V8C 推理路径下，继续训练 BB2 不是稳定泛化增益点。
最干净、最强泛化、最适合作为论文主线的选择仍然是：

converged nnUNet prior
  -> SafeAnchor-GPC-v8c NoResidual
  -> protected archived BB2/SAM bridge
  -> single MobileSAM decode
  -> learned final threshold
```

本轮新增的 `Protected-Bridge Distilled BB2` 是目前最科学的 BB2 训练尝试：它在最终 GPC 推理路径内训练，使用全量 Kvasir train880 / PH2 train140，使用 `checkpoint_last`，并加入训练期 final-response distillation 来保护旧主线函数行为。但结果仍然只带来 PH2 微增益、Kvasir 微损伤，因此不升主线。

---

## 1. BB2 在 V8C 主线中的真实角色

V8C 的完整路径是：

\[
I \xrightarrow{\text{nnUNet}} P_0,\quad
Z_0=\operatorname{logit}(P_0).
\]

当前 V8C 是 `NoResidual`：

\[
Z_c = Z_0.
\]

也就是说，GPC 不直接做像素级 coarse mask residual 修复。它学习的是连续 prompt 参数：

\[
\alpha_\eta(I,Z_0),\quad b_\eta(I,Z_0),\quad \tau_\eta(I,Z_0).
\]

BB2 负责把 coarse prior 的边界证据变成 SAM mask prompt response：

\[
B_{\phi_0}(X),\quad X=[Z_0,\operatorname{band}(Z_0),\|\nabla Z_0\|,\operatorname{morph}(Z_0)].
\]

在当前主线配置中，实际有效证据是 `logit + grad`：

\[
X_{\text{active}}=[Z_0,\|\nabla Z_0\|].
\]

主线 prompt 为：

\[
Q
=\alpha_\eta(I,Z_0)Z_0
+\bigl(1-\alpha_\eta(I,Z_0)\bigr)B_{\phi_0}(X_{\text{active}}).
\]

最终只解码一次：

\[
\hat S=\operatorname{SAM}_{\psi_0}(I,b_\eta,Q),
\quad
\hat Y=\mathbf 1[\sigma(\hat S)>\tau_\eta(I,Z_0)].
\]

因此，BB2 不是一个独立 segmentation head。它是 frozen SAM 解码前的 boundary prompt bridge。任何 BB2 训练如果脱离 GPC 的 \(\alpha,b,\tau\)，都会产生训练-推理错位。

---

## 2. 为什么单独重训 BB2 不够 CCFA

旧的独立 BB2 训练可以写成：

\[
\min_\phi
\mathcal L_{\text{seg}}
\mathcal L_{\text{boundary}},
\]

其中：

\[
\hat S_\phi=\operatorname{SAM}_{\psi}(I,b_0,B_\phi(X)).
\]

这个目标的问题是，它没有看见 V8C 推理时真正使用的：

```text
GPC soft-moment box
GPC alpha
GPC learned threshold
protected SAM state
single final decode path
```

于是 BB2 可能在自己的训练路径里变好，却在 V8C 路径里变坏。这正是 native BB2v2 的实际失败原因。

---

## 3. 已尝试 BB2 训练分支

### 3.1 Native BB2v2 decoder-finetune

设计：

```text
input: native two-channel [logit, grad]
adapter: lightweight multiscale ConvNeXt-style BB2v2
trainable: BB2v2 + MobileSAM prompt encoder + mask decoder
frozen: MobileSAM image encoder
training: Kvasir train880 / PH2 train140, full 3ep
selection: checkpoint_last
```

结果：

| Method | Kvasir val50 | Kvasir test70 | PH2 val30 | PH2 test30 |
|---|---:|---:|---:|---:|
| V8C protected mainline | 90.5115 / 8.4902 / 32.0095 | 85.2218 / 19.8053 / 63.6669 | 95.4095 / 9.8026 / 32.7226 | 92.3030 / 15.4886 / 47.4019 |
| BB2v2 decoder-finetune | 89.5674 / 9.2782 / 35.4764 | 84.6088 / 21.4589 / 78.0053 | 95.7358 / 12.7198 / 45.6099 | 92.8267 / 14.6847 / 43.5623 |

判定：

```text
Rejected.
```

原因：PH2 Dice 上升，但 surface 变差；Kvasir val/test 同时下降。这个分支学到了对 PH2 平滑闭合病灶更友好的 decoder bias，但不适合 Kvasir polyp 的复杂边界。

### 3.2 Native BB2v2 true frozen-SAM

设计：

```text
input: native two-channel [logit, grad]
trainable: only BB2v2
frozen: MobileSAM image encoder, prompt encoder, mask decoder
training: full 3ep
selection: checkpoint_last
```

结果：

| Method | Kvasir val50 | Kvasir test70 | PH2 val30 | PH2 test30 |
|---|---:|---:|---:|---:|
| BB2v2 true frozen-SAM | 85.3990 / 12.8905 / 42.9662 | 80.1578 / 26.3177 / 77.4308 | 88.4047 / 23.4610 / 67.9398 | 89.7358 / 20.9660 / 55.4142 |

判定：

```text
Rejected.
```

原因：随机初始化的 native two-channel adapter 无法有效控制 frozen MobileSAM 的 mask prompt pathway。只靠 BB2v2 adapter 学不到 protected bridge 已经拥有的 prompt-response 对齐。

### 3.3 Protected-Bridge Aligned BB2

设计：

```text
init: protected archived BB2/SAM bridge
trainable: BB2 adapter
frozen: nnUNet prior, GPC, MobileSAM image encoder, prompt encoder, mask decoder
training path: exact V8C final decode path
selection: checkpoint_last
```

目标函数：

\[
\hat S_\phi
=\operatorname{SAM}_{\psi_0}
\left(I,b_\eta,
\alpha_\eta Z_0+(1-\alpha_\eta)B_\phi(X_{\text{active}})
\right).
\]

\[
\mathcal L
=
\mathcal L_{\text{final}}(\hat S_\phi,Y)
+\mathcal L_{\text{bin}}(\hat S_\phi,\tau_\eta,Y)
+\mathcal L_{\text{boundary}}(\hat S_\phi,Y)
+\mathcal L_{\text{anchor}}(B_\phi,B_{\phi_0}).
\]

其中 anchor 对齐的是 protected bridge response，而不是 coarse logit：

\[
\mathcal L_{\text{anchor}}
=
\operatorname{SmoothL1}(B_\phi(X),B_{\phi_0}(X)).
\]

结果：

| Method | Kvasir val50 | Kvasir test70 | PH2 val30 | PH2 test30 |
|---|---:|---:|---:|---:|
| Protected-bridge aligned BB2 | 90.4937 / 8.5241 / 31.9615 | 85.1936 / 19.8757 / 63.8861 | 95.4326 / 9.7436 / 32.5336 | 92.3180 / 15.4047 / 47.2693 |

判定：

```text
Not promoted.
```

原因：PH2 四个指标方向整体微好，但 Kvasir Dice/ASD 微伤。CCFA 主线不能为了 PH2 的 +0.015 到 +0.023 Dice 换 Kvasir 的稳定性。

### 3.4 Protected-Bridge Distilled BB2

本轮新增分支：

```text
config:
  configs/branch_safeanchor_gpc_v8c_bb2_bridge_distill.json

train:
  scripts/run_train_bb2_bridge_distill_kvasir.ps1
  scripts/run_train_bb2_bridge_distill_ph2.ps1

eval:
  scripts/run_eval_bb2_bridge_distill_all.ps1
```

它比 aligned BB2 多一个训练期 final-response distillation：

\[
\hat S_{\phi_0}
=\operatorname{SAM}_{\psi_0}
\left(I,b_\eta,
\alpha_\eta Z_0+(1-\alpha_\eta)B_{\phi_0}(X_{\text{active}})
\right).
\]

\[
\mathcal L_{\text{teacher-final}}
=
\operatorname{SmoothL1}
\left(
\sigma(\hat S_\phi),
\sigma(\hat S_{\phi_0})
\right).
\]

总损失为：

\[
\mathcal L
=
\sum_i
\left(
\exp(-s_i)\mathcal L_i+s_i
\right),
\]

其中 \(s_i\) 是可学习 loss log variance。也就是说，我们没有手工指定每个 loss 的固定权重，而是用 homoscedastic uncertainty 自动学习权重。

训练曲线摘要：

| Dataset | epoch | final | binarize | boundary | anchor | teacher-final | prompt delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kvasir | 1 | 0.4506 | 0.1301 | 0.00619 | 0.04703 | 0.000005 | 0.1122 |
| Kvasir | 2 | 0.4504 | 0.1297 | 0.00619 | 0.01991 | 0.000003 | 0.0703 |
| Kvasir | 3 | 0.4503 | 0.1296 | 0.00619 | 0.01387 | 0.000002 | 0.0581 |
| PH2 | 1 | 0.1581 | 0.1495 | 0.00175 | 0.03129 | 0.000015 | 0.1524 |
| PH2 | 2 | 0.1579 | 0.1492 | 0.00175 | 0.01627 | 0.000012 | 0.1037 |
| PH2 | 3 | 0.1579 | 0.1490 | 0.00175 | 0.01382 | 0.000010 | 0.0963 |

结果：

| Method | Kvasir val50 | Kvasir test70 | PH2 val30 | PH2 test30 |
|---|---:|---:|---:|---:|
| V8C protected mainline | 90.5115 / 8.4902 / 32.0095 | 85.2218 / 19.8053 / 63.6669 | 95.4095 / 9.8026 / 32.7226 | 92.3030 / 15.4886 / 47.4019 |
| Bridge-distilled BB2 | 90.4881 / 8.4925 / 32.0067 | 85.2073 / 19.9242 / 64.0066 | 95.4282 / 9.7481 / 32.5945 | 92.3155 / 15.4129 / 47.3377 |

相对 protected V8C：

| Split | Dice delta | ASD delta | HD95 delta |
|---|---:|---:|---:|
| Kvasir val50 | -0.0234 | +0.0023 | -0.0028 |
| Kvasir test70 | -0.0145 | +0.1189 | +0.3397 |
| PH2 val30 | +0.0187 | -0.0545 | -0.1281 |
| PH2 test30 | +0.0125 | -0.0757 | -0.0642 |

判定：

```text
Not promoted.
```

原因：distillation 让 BB2 训练更稳，但没有突破主线。它证明了一件重要的事：当前 protected BB2/SAM bridge 已经处在一个非常强的局部泛化点；继续训练 BB2 只会在 Kvasir 与 PH2 之间做小幅 trade-off。

---

## 4. 从训练曲线看问题在哪里

Kvasir 的 full 3ep 曲线有一个明显现象：

```text
final loss 几乎不动：0.4506 -> 0.4503
boundary loss 几乎不动：0.00619 -> 0.00619
prompt delta 下降：0.1122 -> 0.0581
teacher-final 接近 0
```

这说明模型主要在学“回到 protected bridge”，而不是找到新的可泛化改进方向。

PH2 也是类似：

```text
final loss 几乎不动：0.1581 -> 0.1579
boundary loss 几乎不动：0.00175 -> 0.00175
prompt delta 下降：0.1524 -> 0.0963
teacher-final 很小
```

PH2 指标稍微提升，是因为病灶形态更平滑，轻微 prompt response 调整就可能改善 Dice/ASD；但这不是跨域稳定规律。

---

## 5. 架构级原因

当前系统的主要可学习性已经由 GPC 承担：

```text
GPC learns alpha:
  decide how much to trust coarse prior vs BB2 response

GPC learns soft box:
  decide how much context to expose to SAM

GPC learns threshold:
  decide final binarization per case
```

BB2 在这个系统里的最佳角色不是继续变强，而是保持稳定的 boundary prompt basis。

如果继续训练 BB2，就会出现三种风险：

```text
1. prompt response drift:
   BB2 output moves away from the protected SAM prompt manifold.

2. domain shape bias:
   PH2 smoother lesions benefit, Kvasir irregular polyps get hurt.

3. objective redundancy:
   GPC already learns alpha/box/threshold, so BB2 fine-tuning mostly duplicates or disturbs GPC policy.
```

因此，“强化可学习性”不能等价于“所有模块都继续训练”。对 CCFA 级主线来说，更高级的选择是：

```text
freeze strong prompt basis,
learn small continuous calibration policy,
reject branches that improve only one dataset.
```

---

## 6. BB2 训练的最终主线决策

当前主线应保留：

```text
SafeAnchor-GPC-v8c NoResidual
+ protected archived BB2/SAM bridge
+ logit/grad active evidence
+ single decode
+ learned alpha / box / threshold
```

当前不应升主线：

```text
native BB2v2 decoder-finetune
native BB2v2 true frozen-SAM
protected-bridge aligned BB2
protected-bridge distilled BB2
```

这不是因为 BB2 训练没有被优化，而是因为它已经被按更严格的方式优化过，仍然没有给出跨 Kvasir + PH2 的稳定优势。

---

## 7. 论文叙事建议

可以这样写 BB2 部分：

```text
The boundary bridge is kept as a protected prompt basis rather than a fully re-optimized decoder. We investigated several increasingly aligned BB2 training strategies, including native two-channel retraining, decoder finetuning, final-path aligned finetuning, and protected final-response distillation. Although these variants can improve PH2 slightly, they introduce Kvasir degradation or surface instability. This suggests that the archived BB2/SAM bridge already provides a stable boundary prompt manifold, while the learnable GPC module should handle case-level adaptation through alpha, soft box, and threshold calibration. Therefore, the final mainline freezes BB2/SAM and learns only the calibration policy.
```

中文叙事：

```text
我们没有把 BB2 训练失败简单归因于容量不足，而是系统比较了 native two-channel 重训、decoder 微调、最终路径对齐微调和 protected final-response 蒸馏。结果表明，BB2 可训练性确实能带来 PH2 小幅收益，但会损伤 Kvasir 或 surface 稳定性。因此，BB2 在最终方法中被定义为稳定的边界提示基，而不是继续优化的主适配器；跨 case 的可学习性由 GPC 的 alpha、soft box 和 threshold 承担。
```

这比“我们又加了一个 BB2 loss”更像 CCFA 级叙事：每个模块都有边界、证据和存在理由。

---

## 8. 下一步

如果目标是泛化优秀，而不是只追 Kvasir val50 91+，下一步不应继续改 BB2。建议顺序：

```text
1. 固定 V8C protected mainline，跑 multi-seed full 3ep checkpoint_last mean/std。
2. 做 Kvasir test70 和 PH2 test30 的 failure visualization，特别看 HD95 长尾。
3. 如果还要提升，优先查 coarse prior 质量和 nnUNet probability calibration，而不是训练 BB2。
4. BB2 只保留为 frozen prompt basis；任何新 BB2 分支必须同时超过 Kvasir test70 和 PH2 test30 才能重开主线讨论。
```

最终主线判断：

```text
从 BB2 训练角度，当前 V8C protected mainline 已经足够干净。
更 CCFA 的下一步是证明它稳定，而不是继续把 BB2 训练成另一个不稳定适配器。
```

---

## 9. 参考思想

本轮设计主要借鉴以下方向，但只保留和当前代码路径一致的部分：

```text
Segment Anything:
  prompt encoder / mask decoder 结构说明为什么 BB2 应被视为 prompt bridge，而不是独立分割器。

Medical SAM Adapter / SAMed:
  医学域适配通常更适合轻量 prompt/decoder adaptation，但本项目证据显示 decoder finetune 在跨域上不稳。

Boundary Loss / Generalized Dice:
  医学分割边界与类别不平衡需要进入训练目标，但 loss 只有在最终推理路径对齐时才有意义。

DeepLab / ASPP / ConvNeXt:
  多尺度卷积能增加边界感受野，因此被用于 BB2v2 诊断分支；但容量提升没有解决跨域泛化。

Homoscedastic uncertainty weighting:
  用可学习 loss log variance 取代固定 loss coefficient，避免把手工 loss 权重写成方法贡献。
```

相关文献：

- Segment Anything: https://arxiv.org/abs/2304.02643
- Medical SAM Adapter: https://arxiv.org/abs/2304.12620
- SAMed: https://arxiv.org/abs/2304.13785
- Boundary Loss: https://arxiv.org/abs/1812.07032
- Generalized Dice Loss: https://arxiv.org/abs/1707.03237
- DeepLabv3: https://arxiv.org/abs/1706.05587
- ConvNeXt: https://arxiv.org/abs/2201.03545
- Multi-task uncertainty weighting: https://arxiv.org/abs/1705.07115
