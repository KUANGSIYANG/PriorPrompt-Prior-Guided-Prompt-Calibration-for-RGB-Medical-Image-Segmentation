# SafeAnchor-GPC-v8c 代码级架构审稿底稿

Date: 2026-05-29

本文只描述 `E:\PA\safeanchor_repro_clean\paper_release_v8c` 当前 release 中真实可运行的 V8C 主线和 IDRiD/DFUC 扩展协议。目标不是再发明一个分支，而是把当前架构按 CCF-A/医学分割论文规范拆清楚：每个 active 模块为什么存在，张量形状是什么，训练和推理哪里可学习，哪里冻结，哪些 head/slot 只是代码兼容而不是论文贡献。

当前主线名：

```text
SafeAnchor-GPC-v8c Teacher-Free NoResidual NoTemperature LearnedThreshold
```

一句话架构：

```text
dataset-specific nnUNet probability
  -> GPC prior-conditioned prompt calibration
  -> frozen dataset-specific BB2/SAM boundary bridge
  -> one MobileSAM decode
  -> learned final threshold
  -> binary lesion mask
```

---

## 1. Source of Truth

### 1.1 主线配置

代码依据：

```text
configs/branch_safeanchor_gpc_v8c_paper_release.json
configs/branch_safeanchor_gpc_v8c_teacher_free_nores_notemp_learnthr_unified.json
configs/branch_safeanchor_gpc_v8c_idrid_dfuc_extension.json
```

关键字段：

```json
{
  "single_path_per_case": true,
  "safeanchor_final_mask": true,
  "teacher_mode": "none",
  "bb2_channels": "logit_grad_only",
  "uses_point_prompt": false,
  "uses_thresholded_box_in_model": false,
  "uses_fixed_alpha_at_inference": false,
  "uses_fixed_pred_threshold_at_inference": false,
  "gpc_cfg": {
    "hidden_channels": 24,
    "depth": 4,
    "use_residual": false,
    "use_distill_loss": false,
    "use_temperature": false,
    "use_learned_threshold": true,
    "global_delta_scale": 1.0,
    "use_learned_global_scale": true,
    "use_global_tanh": true,
    "threshold_init": 0.6,
    "threshold_sharpness_init": 12.0
  }
}
```

论文中必须坚持的边界：

```text
V8C is not a pixel residual refiner.
V8C is not a post-processing cascade.
V8C is not teacher-distilled in the current mainline.
V8C does not use point prompts, candidate gates, area gates, validation-best selection, or checkpoint soup.
V8C learns prompt-level calibration parameters for a frozen SafeAnchor/MobileSAM bridge.
```

### 1.2 代码入口

| 任务 | 入口 | 关键代码 |
|---|---|---|
| nnUNet 训练 | `train.py --stage nnunet` | `train_nnunet.py` |
| nnUNet probability 导出 | `export_nnunet_probs.py` | `predict_from_raw_data(..., save_probabilities=True)` |
| BB2/SAM bridge 训练 | `train.py --stage bb2` | `train_bb2.py` |
| GPC 训练 | `train.py --stage gpc` | `train_gpc_prior.py` |
| V8C 推理 | `run_safeanchor_gpc.py` | one GPC + one BB2 + one MobileSAM decode |
| 数据加载 | `safeanchor/gpc_data.py` | `PriorDataset`, `collate_pad`, `safe_logit` |
| GPC 模型 | `safeanchor/generalizable_prior_calibrator.py` | `GeneralizablePriorCalibrator`, `gpc_loss` |
| BB2 bridge | `safeanchor/bb2_adapter.py` | `BB2PromptAdapter`, `BB2PromptAdapterV2`, `load_bb2_adapter` |
| SAM wrapper | `safeanchor/mobile_sam_wrapper.py` | `FrozenMobileSAM` |

代码审计行号证据：

```text
GPC cfg/model/loss: safeanchor/generalizable_prior_calibrator.py:15-383
GPC data path: safeanchor/gpc_data.py:27-156
BB2 map construction: safeanchor/coarse_prior_calibrator.py:61-88
BB2 channel pruning: safeanchor/bb2_channel_prune.py:7-41
BB2 adapter: safeanchor/bb2_adapter.py:12-327
MobileSAM wrapper: safeanchor/mobile_sam_wrapper.py:34-343
GPC training loop: train_gpc_prior.py:97-338
BB2 training loop: train_bb2.py:139-250
V8C inference loop: run_safeanchor_gpc.py:138-229
nnUNet train/export: train_nnunet.py:15-175, export_nnunet_probs.py:10-96
```

---

## 2. Dataset and Weight Protocol

### 2.1 数据集各自训练各自权重

当前协议要求每个数据集拥有自己的：

```text
nnUNet coarse model
nnUNet probability maps
BB2/SAM bridge checkpoint
GPC checkpoint
```

唯一共享的是 MobileSAM base checkpoint：

```text
assets/checkpoints/mobile_sam.pt
```

这是论文归因的关键。V8C 的跨数据集说服力来自“同一架构、同一训练入口、同一推理入口、独立数据集权重”，而不是把 Kvasir/PH2 权重复用到 IDRiD/DFUC。

### 2.2 当前主线数据

Kvasir/PH2 主线：

```text
Kvasir train880 / val50 / test70
PH2 train140 / val30 / test30
```

外部 RGB 病灶扩展：

```text
IDRiD train/test = 54/27
DFUC/FUSeg train/test = 1010/200
```

当前本地状态审计：

```text
IDRiD train: ids=54, images=54, labels=54, probs=0
IDRiD test:  ids=27, images=27, labels=27, probs=0
DFUC train:  ids=0, images=0, labels=0, probs=0
DFUC test:   ids=0, images=0, labels=0, probs=0
IDRiD/DFUC BB2 checkpoints: not present yet
```

解释：IDRiD 原始 segmentation 已规范化成 binary lesion split，但还没有完成 dataset-specific nnUNet 训练与 probability 导出。DFUC/FUSeg 原始数据还未进入 release 目录。

### 2.3 nnUNet 训练细节

`train_nnunet.py` 使用 nnU-Net v2 的官方训练入口：

```text
config_name = 2d
fold = 0
trainer = nnUNetTrainer
plans = nnUNetPlans
device = cuda by protocol
input = RGB NaturalImage2DIO
label = binary foreground lesion/polyp/ulcer
```

数据集 ID：

```text
902 KvasirSEG2D
903 PH2
904 IDRiDLesion
905 DFUCUlcer
```

`prepare_raw_dataset()` 只把当前 dataset 的 train split 写入 nnUNet raw dataset：

```text
imagesTr/{case_id}_0000.png
labelsTr/{case_id}.png
dataset.json with R/G/B channel names
```

`export_nnunet_probs.py` 对 train/test 或 val/test split 调用：

```text
predict_from_raw_data(..., save_probabilities=True, checkpoint_name="checkpoint_final.pth", device=cuda)
```

导出的 `.npz` 中 `probabilities` 被 `load_prob_map()` 读取为：

```text
probabilities: 2 x 1 x H x W
foreground P0 = probabilities[1, 0]
```

科研叙事理由：nnU-Net 是当前医学分割中强基线/自配置 backbone，V8C 不从零做语义分割，而是在强 coarse prior 上学习 prompt calibration。这样增益可以干净归因于 prior-to-SAM calibration，而不是换了 backbone。

---

## 3. Tensor Shapes

### 3.1 Dataset 输出

`PriorDataset.__getitem__` 输出：

```text
image:     3 x H x W, float32, [0,1]
gt:        1 x H x W, float32, binary
logit_raw: 1 x H x W, float32
case_id:   string
```

其中：

\[
Z_0=\log \frac{\operatorname{clip}(P_0,10^{-4},1-10^{-4})}{1-\operatorname{clip}(P_0,10^{-4},1-10^{-4})}.
\]

训练时 `max_train_side=768`。若长边超过 768：

```text
image: bilinear resize
gt: nearest resize
prob: bilinear resize
```

`collate_pad()` 将 batch 内样本 pad 到同一形状：

```text
image batch:     B x 3 x Hmax x Wmax
gt batch:        B x 1 x Hmax x Wmax
logit_raw batch: B x 1 x Hmax x Wmax
```

当前训练默认 `batch_size=1`，所以多数情况下没有跨样本 padding 干扰。

### 3.2 GPC 输入特征

给定：

```text
I:  B x 3 x H x W
Z0: B x 1 x H x W
```

GPC 构造 6 通道 feature：

```text
clip(Z0, -8, 8) / 8: B x 1 x H x W
sigmoid(Z0):         B x 1 x H x W
uncertainty 4p(1-p): B x 1 x H x W
grad_logit:          B x 1 x H x W
luma:                B x 1 x H x W
grad_luma:           B x 1 x H x W
```

拼接后：

```text
Phi(I,Z0): B x 6 x H x W
```

必要性：

| 通道 | 理由 |
|---|---|
| clipped logit | 保留 nnUNet confidence 的有符号 margin，避免极端 logit 主导 |
| probability | 给出 foreground mass，用于更自然的空间统计 |
| uncertainty | 标出 coarse prior 边界和不确定区域 |
| logit gradient | 提供 prior 自身边界证据 |
| image luma | 提供 RGB 图像低层强度结构 |
| luma gradient | 提供原图边界证据，弥补 coarse prior 漏边 |

这些特征都来自已有输入，没有额外模型、没有手写 case rule。

---

## 4. GPC Architecture

### 4.1 实测形状与参数量

用当前 `branch_safeanchor_gpc_v8c_paper_release.json` dry-run：

```text
input image: B=2, 3 x 128 x 160
input logit: B=2, 1 x 128 x 160
GPC params: 16,482
GPC trainable params: 16,482
```

输出：

```text
logit_cal:           2 x 1 x 128 x 160
residual:            2 x 1 x 128 x 160
alpha:               2 x 1 x 1 x 1
boxes:               2 x 4
box_scale:           2
temperature:         2 x 1 x 1 x 1
threshold:           2 x 1 x 1 x 1
threshold_sharpness: 1
```

当前有效自由度只有：

```text
alpha
box_scale
threshold
```

`residual` 输出恒为 0，因为 `use_residual=false`。`temperature` 恒为 1，因为 `use_temperature=false`。

### 4.2 Stem

代码：

```text
Conv2d(6 -> 24, kernel=3, padding=1, bias=False)
GroupNorm(groups=8, channels=24)
SiLU(inplace=True)
```

形状：

```text
B x 6 x H x W -> B x 24 x H x W
```

必要性：

```text
3x3 Conv: 融合局部 prior/image cue。
GroupNorm: batch_size=1 时比 BatchNorm 稳定。
SiLU: 平滑非单调激活，适合小网络的连续 calibration。
```

### 4.3 Encoder Blocks

当前 `depth=4`，每个 `LargeKernelResidualBlock`：

```text
depthwise Conv2d(24 -> 24, kernel=7, padding=3, groups=24, bias=False)
GroupNorm(groups=8, channels=24)
pointwise Conv2d(24 -> 48, kernel=1)
GELU
pointwise Conv2d(48 -> 24, kernel=1)
residual add
```

形状保持：

```text
B x 24 x H x W -> B x 24 x H x W
```

数学形式：

\[
H_{l+1}=H_l+\operatorname{PW}_2(\operatorname{GELU}(\operatorname{PW}_1(\operatorname{GN}(\operatorname{DWConv}_{7\times7}(H_l))))).
\]

必要性：

```text
7x7 depthwise conv: 低参数大感受野，捕捉 coarse boundary neighborhood。
1x1 expand/project: 跨通道 mixing，补偿 depthwise conv 只做逐通道空间卷积。
GELU: 平滑门控式非线性，和 ConvNeXt-like block 对齐。
Residual add: 保持轻量校准网络不破坏 stem 表征，训练更稳。
```

### 4.4 Global Head

代码：

```text
AdaptiveAvgPool2d(1)
Conv2d(24 -> 24, kernel=1)
SiLU(inplace=True)
Conv2d(24 -> 4, kernel=1)
```

形状：

```text
B x 24 x H x W -> B x 4 x 1 x 1 -> global_delta: B x 4
```

当前 active 使用：

```text
global_delta[0] -> alpha
global_delta[1] -> box_scale
global_delta[3] -> threshold
```

当前 inactive compatibility slot：

```text
global_delta[2] -> temperature slot, but use_temperature=false, so ignored
```

审稿边界：论文图里不要把 temperature 作为 V8C 贡献。当前代码保留 4 输出是为了同一类支持历史分支/现有 checkpoint shape；从“纯 V8C-only 架构洁癖”角度，未来若重训可把 no-temperature learned-threshold 版本改成 3 输出，但这会破坏现有 checkpoint 兼容，不适合直接改当前 release 结果。

### 4.5 Residual Head

代码里有：

```text
residual_head = Conv2d(24 -> 1, kernel=1)
```

但当前配置：

```text
use_residual = false
```

所以：

\[
\Delta Z=0,\qquad Z_c=Z_0.
\]

审稿边界：`residual_head` 是共享类的历史兼容 head，不是当前 V8C active module。论文不能写“GPC 学会 pixel residual 修复 coarse mask”。准确说法是：

```text
V8C preserves the nnUNet prior and learns prompt-level calibration.
```

### 4.6 Alpha, Box Scale, Threshold

GPC global output 先经过：

\[
g=\lambda\tanh(\operatorname{GlobalHead}(H)),
\]

其中当前 `use_learned_global_scale=true`，\(\lambda=\operatorname{softplus}(s_g)\) 可学习。

Alpha：

\[
\alpha=\sigma(a_0+g_0),\qquad \alpha\in[0,1].
\]

Box scale：

\[
s_b=\operatorname{softplus}(b_0+g_1),\qquad s_b\ge 0.5.
\]

Threshold：

\[
\tau=\sigma(t_0+g_3),\qquad \tau\in[0,1].
\]

Threshold sharpness：

\[
\kappa=\operatorname{softplus}(k_0),\qquad \kappa\ge1.
\]

必要性：

```text
alpha: dense prompt 中 coarse prior 与 BB2 boundary response 的 case-wise mixing。
box_scale: soft-moment box 的尺度，决定 SAM box prompt 的上下文范围。
threshold: final probability 到 binary mask 的 case-wise decision。
```

这三个自由度正好对应 SafeAnchor/MobileSAM 推理链中最关键、最少量的可学习决策位置。

---

## 5. Soft-Moment Box

当前 V8C 不在模型内部用 fixed threshold box。它使用 `soft_moment_box(sigmoid(Zc), box_scale)`：

\[
m=\sum_x P_c(x),
\quad
\mu_x=\frac{\sum_x P_c(x)x}{m},
\quad
\mu_y=\frac{\sum_x P_c(x)y}{m}.
\]

\[
\sigma_x=\sqrt{\frac{\sum_x P_c(x)(x-\mu_x)^2}{m}},
\quad
\sigma_y=\sqrt{\frac{\sum_x P_c(x)(y-\mu_y)^2}{m}}.
\]

\[
b=[\mu_x-s_b\sigma_x,\mu_y-s_b\sigma_y,\mu_x+s_b\sigma_x,\mu_y+s_b\sigma_y],
\]

再 clamp 到 `[0,1]`。

形状：

```text
sigmoid(Zc): B x 1 x H x W
box_scale:   B
boxes:       B x 4 normalized xyxy
```

必要性：soft moment box 是连续概率几何，不依赖手写 threshold，因此比 `P0>0.4` 的 hard box 更符合可学习 calibration 叙事。IDRiD 的多小病灶会挑战这一点，因为 global box 可能覆盖更多背景；这正是 IDRiD 作为压力测试的价值。

---

## 6. BB2/SAM Boundary Bridge

### 6.1 BB2 evidence maps

`build_bb2_maps_from_logit(Zc)` 输出：

```text
channel 0: clipped logit
channel 1: soft uncertainty band around p=0.5
channel 2: Sobel gradient magnitude of clipped logit
channel 3: smooth morphology edge surrogate
```

形状：

```text
B x 4 x H x W
```

当前主线使用：

```text
bb2_channel_mode = logit_grad
```

这会保留 channel 0 和 channel 2，其余通道置零。由于历史 RepVGG checkpoint 需要 4 通道输入，`logit_grad` 是“4 通道兼容包装 + 2 个有效证据通道”。如果使用 `logit_grad2`，则实际张量会变成 `B x 2 x H x W`，适合 BB2v2 分支，但不是当前 protected V8C 主线。

必要性：

```text
logit: 保留 semantic anchor 的有符号强度。
gradient: 提供 boundary response 的核心证据。
band/morph: 当前 logit_grad 模式下不是 active method evidence，不应在论文中夸大。
```

### 6.2 当前 protected BB2 adapter

本地 checkpoint 审计：

```text
kvasir_bb2_checkpoint_best.pth: cfg in_channels=4, base_channels=32, depth=4, sam_state_dict exists
ph2_bb2_checkpoint_best.pth:   cfg in_channels=4, base_channels=32, depth=4, arch=repvgg, sam_state_dict exists
idrid_bb2_checkpoint_last.pth: missing
dfuc_bb2_checkpoint_last.pth:  missing
```

RepVGG block：

```text
3x3 Conv + BatchNorm
1x1 Conv + BatchNorm
optional identity BatchNorm
sum + ReLU
```

BB2 backbone：

```text
4 x RepVGGBlock
1x1 Conv head -> Zbb2
```

实测新建 RepVGG adapter 参数量：

```text
BB2 repvgg params: 42,404
output: B x 1 x H x W
alpha parameter shape: scalar
```

GPC 推理不使用 BB2 checkpoint 内部 alpha，而是只调用：

```text
Zbb2 = adapter.run_bb2_adapter(maps, Zc)
```

然后用 GPC 的 alpha 融合：

\[
Q=\alpha Z_c+(1-\alpha)Z_{\text{bb2}}.
\]

必要性：BB2 是 frozen boundary response basis。它不是 final mask，也不是另一个 candidate selector；它只提供一个可由 GPC 选择强度的边界提示。

### 6.3 BB2 training for IDRiD/DFUC

扩展脚本：

```powershell
train.py --stage bb2 --dataset idrid --arch repvgg --bb2_channel_mode logit_grad --train_sam_prompt_decoder
train.py --stage bb2 --dataset dfuc  --arch repvgg --bb2_channel_mode logit_grad --train_sam_prompt_decoder
```

训练目标：

```text
seg = Dice + BCE on SAM decoded logits
boundary = SmoothL1(grad(sigmoid(logits)), grad(gt))
total = uncertainty-weighted sum(seg, boundary)
```

冻结/可学：

```text
nnUNet probabilities: frozen input
SAM image encoder: frozen
BB2 adapter: trainable
SAM prompt encoder and mask decoder: trainable when --train_sam_prompt_decoder is set
loss_log_vars: trainable
```

扩展训练完成后，checkpoint 被复制到：

```text
assets/checkpoints/idrid_bb2_checkpoint_last.pth
assets/checkpoints/dfuc_bb2_checkpoint_last.pth
```

这样 IDRiD/DFUC 的 bridge 与 Kvasir/PH2 一样是 dataset-specific artifact。

---

## 7. MobileSAM Wrapper

### 7.1 Preprocess

`FrozenMobileSAM.encode_image()`：

```text
input image: B x 3 x H x W, [0,1] or [0,255]
scale longest side to 1024
normalize by SAM pixel_mean/pixel_std
pad to 1024 x 1024
image encoder -> image_embeddings
```

常见 vit_t shape：

```text
image_embeddings: B x 256 x 64 x 64
```

### 7.2 Prompt encoding

Box prompt：

```text
boxes_norm: B x 4 normalized xyxy
boxes_resized: B x 4 pixel xyxy after longest-side resize
```

Dense mask prompt：

```text
Q: B x 1 x H x W
resize to SAM input_size
pad to 1024 x 1024
downsample to 256 x 256
mask_input: B x 1 x 256 x 256
```

如果 mask prompt 已经是 `[0,1]`，wrapper 会转 logit；当前 V8C 传入的是 logit-like prompt `Q`，因此不会被二次 logit。

### 7.3 Decode

```text
prompt_encoder(points=None, boxes=boxes, masks=mask_input)
mask_decoder(multimask_output=False)
postprocess_masks(...)
```

输出：

```text
final_logits Y: B x 1 x H x W
iou_pred:      B x 1
```

推理只有一次 MobileSAM decode。没有 candidate gate，没有多 prompt bank，没有 oracle selector。

---

## 8. GPC Training Protocol

### 8.1 Trainable and frozen

`train_gpc_prior.py` 中 optimizer 只包含：

```text
GeneralizablePriorCalibrator.parameters()
```

因此 GPC 训练更新：

```text
GPC stem / blocks / global head
GPC scalar params alpha_logit, box_scale_log, threshold_logit, threshold_sharpness_log
GPC loss_log_vars
```

不更新：

```text
nnUNet
BB2 adapter checkpoint
MobileSAM image encoder
MobileSAM prompt encoder
MobileSAM mask decoder
```

代码级小注：`load_bb2_adapter()` 当前会 `model.eval()`，且 BB2 参数不在 optimizer 中，所以权重不会更新；若以后长时间重训，为了显式减少 autograd 参数梯度内存，可以把 loaded BB2 params 设为 `requires_grad=False`。这不会改变方法行为，但会让“frozen bridge”在工程上更字面。

### 8.2 Forward path in training

每个 batch：

```text
1. image, gt, logit_base = dataset batch
2. out = GPC(image, logit_base)
3. maps = build_bb2_maps_from_logit(out["logit_cal"])
4. maps = prune_bb2_channels_torch(maps, "logit_grad")
5. bb2_logits = run_bb2_logits(adapter, maps, out["logit_cal"])
6. prompt = out["alpha"] * out["logit_cal"] + (1-out["alpha"]) * bb2_logits
7. image_embeddings = FrozenMobileSAM.encode_image(image)
8. logits = FrozenMobileSAM.decode_from_embeddings(boxes=out["boxes"], mask_prompt=prompt)
9. loss = gpc_loss(...)
10. optimizer updates GPC only
```

### 8.3 Loss

当前 V8C active losses：

```text
final:    Dice+BCE(final SAM logits, gt)
prior:    Dice+BCE(logit_cal, gt)
boundary: SmoothL1(grad(sigmoid(logit_cal)), grad(gt))
box:      1 - IoU(soft_moment_box, gt_box)
anchor:   SmoothL1(logit_cal, logit_base.detach())
binarize: Dice+BCE(kappa*(sigmoid(final_logits)-tau), gt)
```

当前 inactive losses：

```text
distill: disabled, because use_distill_loss=false and teacher_mode=none
binary_boundary: disabled, because use_binary_shape_losses=false
binary_area: disabled, because use_binary_shape_losses=false
binary_distance: disabled, because use_binary_distance_loss=false
```

Total loss uses learned homoscedastic uncertainty weighting:

\[
\mathcal{L}=\sum_i \exp(-s_i)\mathcal{L}_i+s_i.
\]

必要性：

| Loss | 为什么存在 |
|---|---|
| final | 直接监督最终输出，不让 GPC 只优化中间 prompt |
| prior | 约束 calibrated prior 仍保留 segmentation semantics |
| boundary | 给 surface reliability 明确信号 |
| box | 让 soft-moment box 与 GT support 对齐 |
| anchor | 在 NoResidual 下应为 0，保留是防止未来打开 residual 时漂移 |
| binarize | 让 learned threshold 通过可微 soft binary path 被训练 |

审稿边界：当前 `anchor` 在 NoResidual 下基本是冗余保护项，不是贡献点；可以在方法中作为 regularization 写轻，不要夸大。

### 8.4 Optimizer and checkpoints

默认训练参数：

```text
epochs = 3
batch_size = 1
lr = 5e-5
weight_decay = 1e-4
optimizer = AdamW
clip_grad_norm = 1.0
checkpoint policy = checkpoint_last
```

主线已完成 GPC：

```text
Kvasir: outputs/gpc_v8c_learnthr_unified_kvasir_train880_3ep_v1/checkpoint_last.pth
PH2:    outputs/gpc_v8c_learnthr_unified_ph2_train140_3ep_v1/checkpoint_last.pth
```

训练日志证据：

```text
Kvasir: 880 cases x 3 epochs = 2640 steps
PH2:    140 cases x 3 epochs = 420 steps
```

Kvasir learned quantities：

```text
alpha:     0.5270 -> 0.7489
box_scale: 2.4215 -> 2.2119
threshold: 0.5993 -> 0.5768
residual:  0.0
```

PH2 logged window：

```text
alpha:     0.5178
box_scale: 2.4204
threshold: 0.5880
residual:  0.0
```

---

## 9. Inference Protocol

### 9.1 Single path per case

`run_safeanchor_gpc.py`：

```text
1. load GPC checkpoint
2. load dataset-specific BB2/SAM checkpoint
3. load MobileSAM base + optional dataset-specific sam_state_dict
4. read image, label, nnUNet probability
5. GPC predicts alpha, box, threshold
6. build BB2 logit+grad evidence
7. BB2 produces boundary logits
8. prompt = alpha*Zc + (1-alpha)*Zbb2
9. one MobileSAM decode
10. pred = sigmoid(final_logits) > learned threshold
```

No active inference branch:

```text
box_mode = gpc_soft
alpha_mode = gpc
residual_mode = gpc, but current residual is zero
temperature_mode = gpc, but current temperature is one
pred_mode = auto -> gpc_threshold
prompt_mode = mix
bb2_channel_mode = config/default logit_grad
```

Final mask:

\[
\hat{M}(x)=\mathbf{1}\{\sigma(Y(x))>\tau\}.
\]

### 9.2 What not to claim

不要写：

```text
V8C learns a pixel residual coarse-mask refiner.
V8C uses a temperature-calibrated SAM logit in the current mainline.
V8C uses point prompts or candidate routing.
V8C selects checkpoint by validation/test.
V8C reuses Kvasir/PH2 weights for IDRiD/DFUC.
```

准确写法：

```text
V8C keeps the upstream nnUNet prior as a semantic anchor and learns the minimum prompt-level degrees of freedom needed by a frozen SafeAnchor/MobileSAM bridge: dense-prompt mixture, continuous box scale, and final threshold.
```

---

## 10. Module Necessity Audit

| Component | Active in V8C? | Keep/Cut decision | Reason |
|---|---|---|---|
| Dataset-specific nnUNet | Yes | Keep | Provides strong medical coarse prior; clean baseline anchor |
| `safe_logit(P0)` | Yes | Keep | Stable signed prior representation |
| 6-channel GPC feature | Yes | Keep | Minimal semantic/uncertainty/boundary/image cues |
| GPC 3x3 stem | Yes | Keep | Local fusion before lightweight encoder |
| GPC 7x7 DWConv blocks | Yes | Keep | Large local context at low parameter cost |
| GroupNorm | Yes | Keep | Robust with batch_size=1 |
| SiLU/GELU | Yes | Keep | Smooth nonlinear calibration; aligned with modern ConvNet blocks |
| Global alpha head | Yes | Keep | Controls dense prompt mixing |
| Global box-scale head | Yes | Keep | Controls SAM box prompt support |
| Global threshold head | Yes | Keep | Learns final binary policy |
| Residual head | No in current config | Do not claim; keep only for checkpoint/code compatibility | `use_residual=false`, `residual_abs=0` |
| Temperature slot | No in current config | Do not claim; future pure v8c can prune after retrain | `use_temperature=false`, output remains for compatibility |
| BB2 logit+gradient evidence | Yes | Keep | Boundary response basis without extra image backbone |
| BB2 band/morph channels | Not active under logit_grad | Do not claim as method evidence | Zeroed for current protocol |
| MobileSAM image encoder | Yes, frozen | Keep | Promptable segmentation foundation; final mask decoded by SAM |
| MobileSAM prompt/mask decoder | Yes, frozen during GPC | Keep | Converts learned prompts into final mask |
| Learned loss weights | Yes | Keep | Avoids hand-tuned loss lambdas |
| Distillation loss | No | Cut from V8C story | `teacher_mode=none`, `use_distill_loss=false` |
| Binary shape/distance losses | No | Cut from V8C story | Disabled in current config |

Architecture cleanliness verdict:

```text
The paper method is clean if and only if the method figure exposes only active V8C degrees of freedom:
alpha, box_scale, threshold.
Residual head and temperature slot should be described as implementation compatibility, not method contributions.
```

---

## 11. Literature Positioning

V8C 对应文献不是“随便堆模块”，而是把已有成熟思想放在正确位置：

| V8C part | Literature basis | How to use in narrative |
|---|---|---|
| nnUNet prior | nnU-Net self-configuring biomedical segmentation | Upstream strong medical prior, not our main contribution |
| MobileSAM/SAM decode | Promptable segmentation foundation models | We learn prompts, not a new decoder |
| ConvNeXt-lite GPC | Large-kernel depthwise conv + pointwise mixing | Lightweight local context for prior calibration |
| RepVGG BB2 | 3x3 + 1x1 + identity conv bias | Simple convolutional boundary response basis |
| GroupNorm | Batch-size-robust normalization | Necessary because medical training often uses batch size 1 |
| GELU/SiLU | Smooth activations | Good for continuous calibration heads |
| Dice+BCE | Medical foreground imbalance | Direct segmentation supervision |
| Boundary loss idea | Surface-aware segmentation | Boundary/surface reliability without inference post-processing |
| Uncertainty-weighted loss | Learned multi-loss balancing | Avoid hand-tuned lambda soup |
| DeepLab/dilated context | Large receptive field motivation | Background only; V8C does not use ASPP |

Primary references used for positioning:

```text
nnU-Net: https://www.nature.com/articles/s41592-020-01008-z
Segment Anything: https://arxiv.org/abs/2304.02643
MobileSAM: https://arxiv.org/abs/2306.14289
ConvNeXt: https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html
RepVGG: https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html
GroupNorm: https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html
GELU: https://arxiv.org/abs/1606.08415
Swish/SiLU: https://arxiv.org/abs/1710.05941
Uncertainty-weighted multi-task loss: https://arxiv.org/abs/1705.07115
Boundary loss: https://arxiv.org/abs/1812.07032
V-Net/Dice objective: https://arxiv.org/abs/1606.04797
DeepLab/dilated context background: https://arxiv.org/abs/1706.05587
IDRiD data/challenge: https://idrid.grand-challenge.org/
FUSeg challenge: https://fusc.grand-challenge.org/FUSeg-2021/
DFUC challenge: https://dfu-challenge.github.io/
ISIC 2018 challenge context: https://challenge.isic-archive.com/data/
```

---

## 12. Paper Narrative

### 12.1 Problem

Given an RGB medical image \(I\) and a dataset-specific nnUNet probability prior \(P_0\), direct thresholding of \(P_0\) can retain far-boundary and surface errors. Directly feeding \(P_0\) to SAM is also not automatically optimal because the box prompt support, dense prompt mixture, and final binarization threshold are dataset- and case-dependent.

### 12.2 Hypothesis

```text
The upstream nnUNet prior is already a strong semantic anchor.
The most valuable learnable part is not another pixel-level refiner, but a small prior-conditioned policy that calibrates how this prior is used by a frozen promptable decoder.
```

### 12.3 Method

\[
P_0=f_{\text{nnU}}(I),
\quad
Z_0=\operatorname{logit}(P_0).
\]

\[
(\alpha,s_b,\tau)=\psi_\theta(I,Z_0).
\]

\[
Z_c=Z_0.
\]

\[
b=\operatorname{SoftMomentBox}(\sigma(Z_c),s_b).
\]

\[
Z_{\text{bb2}}=f_{\text{BB2}}([Z_c,\nabla Z_c]).
\]

\[
Q=\alpha Z_c+(1-\alpha)Z_{\text{bb2}}.
\]

\[
Y=f_{\text{MobileSAM}}(I,b,Q).
\]

\[
\hat{M}=\mathbf{1}\{\sigma(Y)>\tau\}.
\]

This is the full V8C mathematical story.

### 12.4 Contribution Claim

Strong claim:

```text
SafeAnchor-GPC-v8c turns a strong nnUNet probability prior into a calibrated prompt policy for a frozen MobileSAM bridge, using only three learned prompt-level degrees of freedom: dense-prompt mixture, continuous box scale, and final threshold.
```

Avoided overclaim:

```text
We do not claim a new segmentation backbone, a new SAM decoder, or a pixel-level residual correction network.
```

---

## 13. IDRiD/DFUC Extension Narrative

Why IDRiD:

```text
RGB fundus lesion segmentation, ISBI challenge lineage, small/disconnected lesions, strong stress test for global soft-moment box.
```

Why DFUC/FUSeg:

```text
RGB clinical wound/ulcer segmentation, MICCAI challenge lineage, non-polyp non-skin domain, more continuous lesions than IDRiD.
```

Two-split protocol:

```text
IDRiD: train 54, test 27
FUSeg/DFUC: train 1010, test 200
```

No validation checkpoint selection:

```text
train split -> train nnUNet -> export train/test probs -> train BB2 -> train GPC -> evaluate test
```

Every dataset uses its own:

```text
Dataset904/905 nnUNet model
data/{dataset}/{train,test}/probs
assets/checkpoints/{dataset}_bb2_checkpoint_last.pth
outputs/gpc_v8c_{dataset}_train_3ep/checkpoint_last.pth
```

科研价值：IDRiD 如果 Dice 不显著但 HD95/ASD 改善，仍可说明 V8C 在多小目标场景下提升 surface reliability。DFUC/FUSeg 更贴合连续病灶，理论上应更能体现 soft-moment box + dense prompt calibration 的稳定性。

---

## 14. Next Clean Steps

不要优先加新模块。按论文强度排序，下一步应该是：

```text
1. 完成 IDRiD dataset-specific nnUNet GPU training。
2. 导出 IDRiD train/test probability maps。
3. 训练 IDRiD 自己的 BB2/SAM bridge。
4. 训练 IDRiD 自己的 GPC checkpoint。
5. 只在 IDRiD test27 报告，不用 test 选 checkpoint。
6. 再接 DFUC/FUSeg 原始数据，重复同一协议。
7. 对 Kvasir/PH2/IDRiD/DFUC 做统一表格：split-level + weighted combined where applicable。
```

如果要做代码洁癖版重训，而不是复用当前 release checkpoint：

```text
1. 建一个 pure-v8c class，删除 residual_head。
2. 在 no-temperature learned-threshold setting 下把 global_head output 从 4 改成 3。
3. 显式 freeze loaded BB2 adapter params。
4. 重新训练 Kvasir/PH2/IDRiD/DFUC，保证结果可比。
```

这属于下一轮 clean-room retrain，不应直接混入当前 release 结果。

