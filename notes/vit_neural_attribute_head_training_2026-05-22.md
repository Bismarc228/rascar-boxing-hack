# ViT neural attribute-head tuning, 2026-05-22

Scope: leakage-safe leave-one-fight OOF neural heads on the committed evaluator
`tools/evaluate_vit_neural_attribute_head.py`. No Kaggle upload was made.

## Evaluator changes

Validated the evaluator fold logic against the fixed-row evaluator:

- labels are matched only from OOF prediction rows to GT within `--label-window`;
- training rows are `(groups != held_out_fight) & label_matched`;
- validation predictions are emitted for the held-out fight only.

Small evaluator changes made:

- `--pca-components <= 0` now means "no PCA"; features are scaled then passed
  directly to the MLP head.
- the script now logs `training_device=... device_name=... cuda_visible=...`.

GPU validation from the final runs:

```text
training_device=cuda:0 device_name=NVIDIA RTX 6000 Ada Generation cuda_visible=1
```

Because every CUDA command used `CUDA_VISIBLE_DEVICES=1`, `cuda:0` is physical
GPU 1. Low `nvidia-smi` utilization is expected here: the dataset has only 1051
labeled rows and the MLP batches are tiny, so CPU fold/scaler work dominates
wall time even though model tensors are on CUDA.

## Baseline

Base OOF rows:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_rgb_effectiveness_margin02_oof.csv
```

Base score:

```text
macro=0.407232
time=0.537088
fighter=0.544665
punch_type=0.203789
effectiveness=0.278732
hand=0.526763
target=0.485253
fp_penalty=0.073644
n=1226
labeled_rows=1051
```

Label counts:

```text
punch_type=cross:378,hook:378,jab:191,uppercut:104
effectiveness=blocked:117,landed:613,miss:321
hand=left:539,right:512
target=body:230,head:821
fighter=blue:575,red:476
```

## Punch type: CLIP

The neural CLIP punch-type head did not beat the existing fixed-row logreg
point. PCA was useful as regularization here; direct 4608-dim MLP heads were
worse.

Best neural CLIP punch-type, 3-seed CE:

```text
features=clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_20260522.npz
pca_components=64
hidden_dim=96
class_weight_power=0.3
seeds=17,29,43
margin=0.26
macro=0.407818
delta_vs_base=+0.000586
score_punch_type=0.209649
wins=7
changed=157
```

Raw neural diagnostic for that run:

```text
baseline punch_type accuracy=0.397716 balanced_acc=0.310421 macro_f1=0.305457 weighted_f1=0.383277
raw_model punch_type accuracy=0.402474 balanced_acc=0.306558 macro_f1=0.297680 weighted_f1=0.383208
cross    precision=0.451531 recall=0.468254 f1=0.459740 support=378
hook     precision=0.442553 recall=0.550265 f1=0.490566 support=378
jab      precision=0.244898 recall=0.188482 f1=0.213018 support=191
uppercut precision=0.047619 recall=0.019231 f1=0.027397 support=104
```

Best direct no-PCA CLIP scout was only:

```text
hidden_dim=96
class_weight_power=0.7
seed=17
margin=0.16
macro=0.407695
score_punch_type=0.208424
```

Conclusion: keep the existing logreg CLIP punch-type margin-0.28 row artifact:

```text
data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_ptype_m028_rows_20260522.csv
macro=0.408457
score_punch_type=0.216039
changed_vs_base=472 punch_type rows
```

## Effectiveness: VideoMAE direct MLP

This is the useful neural result. A direct MLP head over the pooled VideoMAE
features beat the prior VideoMAE logreg effectiveness point.

Best effectiveness-only run on the original base:

```text
features=videomae_attackerdef_glovetarget_clip16s2_current_oof_cuda_preproc_20260522.npz
pca_components=0
hidden_dim=192
dropout=0.25
loss=ce
label_smoothing=0.04
class_weight_power=0.7
seeds=17,29,43
margin=0.3
macro=0.408571
delta_vs_base=+0.001339
score_effectiveness=0.295470
wins=9
changed=159 effectiveness rows
```

Per-class raw-model diagnostics:

```text
baseline effectiveness accuracy=0.544244 balanced_acc=0.459302 macro_f1=0.445687 weighted_f1=0.545601
raw_model effectiveness accuracy=0.522360 balanced_acc=0.457201 macro_f1=0.447165 weighted_f1=0.533586
blocked precision=0.241611 recall=0.307692 f1=0.270677 support=117
landed  precision=0.690979 recall=0.587276 f1=0.634921 support=613
miss    precision=0.401575 recall=0.476636 f1=0.435897 support=321
```

Comparison checks:

```text
prior VideoMAE logreg effectiveness margin 0.9: macro=0.407991, effectiveness=0.288228
direct VideoMAE MLP effectiveness margin 0.3: macro=0.408571, effectiveness=0.295470
```

On top of existing CLIP punch-type margin-0.28 rows:

```text
baseline_with_clip_ptype macro=0.408457
best_stack_margin=0.3
macro=0.409796
delta_vs_clip_ptype=+0.001339
delta_vs_original_base=+0.002564
score_punch_type=0.216039
score_effectiveness=0.295470
time=0.537088
fighter=0.544665
hand=0.526763
target=0.485253
fp_penalty=0.073644
wins_vs_clip_ptype=9
wins_vs_original_base=8
changed_vs_clip_ptype=159 effectiveness rows
changed_vs_original_base=569 rows total: 472 punch_type, 159 effectiveness
```

This beats the prior logreg stack noted by the main agent:

```text
prior stack: CLIP punch_type m0.28 + VideoMAE logreg effectiveness m0.8 => macro=0.409580
new stack:   CLIP punch_type m0.28 + VideoMAE MLP effectiveness m0.3    => macro=0.409796
delta=+0.000216
```

Generated row artifacts:

```text
data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_current_oof_cuda_preproc_neural_eff_direct_h192_w07_ls004_m03_rows_20260522.csv
data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_direct_h192_w07_ls004_m03_rows_20260522.csv
```

## Composition with other component gates

The neural effectiveness stack composes with the previously positive
pose-rival fighter residual and the tiny VideoMAE target gate:

```text
original fixed-row source macro                         0.407232
CLIP punch_type m0.28 + VideoMAE MLP effectiveness m0.3 0.409796
+ pose-rival fighter flip                               0.410071
+ VideoMAE target m0.95                                 0.410125
wins_vs_original                                        8/13
```

Final component deltas versus the original fixed-row source:

```text
score_time          +0.000000
score_fighter       +0.001377
score_punch_type    +0.012250
score_effectiveness +0.016739
score_hand          +0.000000
score_target        +0.000895
fp_penalty          +0.000000
```

The target increment is only `+0.000054` macro over the neural+fighter stack,
so it should remain a private-risk micro-signal rather than a standalone
candidate.

## Negative checks

Direct VideoMAE MLP refinements on the CLIP punch-type baseline:

```text
default CE, smoothing 0.04, dropout 0.25: best macro=0.409796 at margin=0.3
dropout 0.10: best macro=0.409500 at margin=0.35
dropout 0.40: best macro=0.409722 at margin=0.3
CE, smoothing 0.00: best macro=0.409726 at margin=0.25
focal gamma 1.5: best macro=0.409585 at margin=0.2
VideoMAE+CLIP concatenated direct features: best macro=0.409698 at margin=0.4
```

Conclusion: keep weighted CE with mild label smoothing. Focal did not help.
Concatenating CLIP features into the effectiveness MLP did not help.

## Key commands

Smoke validation:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 .venv/bin/python tools/evaluate_vit_neural_attribute_head.py \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_rgb_effectiveness_margin02_oof.csv \
  --feature-caches data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_20260522.npz \
  --heads punch_type \
  --margins 0.0 \
  --pca-components 64 \
  --hidden-dim 96 \
  --epochs 2 \
  --patience 1 \
  --seeds 17 \
  --device cuda
```

Best neural CLIP punch-type check:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 .venv/bin/python tools/evaluate_vit_neural_attribute_head.py \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_rgb_effectiveness_margin02_oof.csv \
  --feature-caches data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_20260522.npz \
  --heads punch_type \
  --margins 0.0,0.2,0.24,0.26,0.28,0.3,0.32,0.36,0.4,0.5 \
  --pca-components 64 \
  --hidden-dim 96 \
  --class-weight-power 0.3 \
  --loss ce \
  --label-smoothing 0.04 \
  --epochs 220 \
  --patience 28 \
  --seeds 17,29,43 \
  --device cuda
```

Best direct VideoMAE effectiveness run and row write:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 .venv/bin/python tools/evaluate_vit_neural_attribute_head.py \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_rgb_effectiveness_margin02_oof.csv \
  --feature-caches data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_current_oof_cuda_preproc_20260522.npz \
  --heads effectiveness \
  --margins 0.0,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6 \
  --pca-components 0 \
  --hidden-dim 192 \
  --dropout 0.25 \
  --class-weight-power 0.7 \
  --loss ce \
  --label-smoothing 0.04 \
  --epochs 220 \
  --patience 28 \
  --seeds 17,29,43 \
  --device cuda \
  --write-oof-rows data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_current_oof_cuda_preproc_neural_eff_direct_h192_w07_ls004_m03_rows_20260522.csv \
  --write-head effectiveness \
  --write-margin 0.3
```

Best stack row write:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 .venv/bin/python tools/evaluate_vit_neural_attribute_head.py \
  --predictions data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_ptype_m028_rows_20260522.csv \
  --feature-caches data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_current_oof_cuda_preproc_20260522.npz \
  --heads effectiveness \
  --margins 0.0,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6 \
  --pca-components 0 \
  --hidden-dim 192 \
  --dropout 0.25 \
  --class-weight-power 0.7 \
  --loss ce \
  --label-smoothing 0.04 \
  --epochs 220 \
  --patience 28 \
  --seeds 17,29,43 \
  --device cuda \
  --write-oof-rows data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_direct_h192_w07_ls004_m03_rows_20260522.csv \
  --write-head effectiveness \
  --write-margin 0.3
```

Validation:

```bash
python3 -m py_compile tools/evaluate_vit_neural_attribute_head.py
git diff --check
```
