# VideoMAE effectiveness head tuning, 2026-05-22

Scope: effectiveness-only head tuning over frozen cached VideoMAE features:

```text
data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_current_oof_cuda_preproc_20260522.npz
features shape=(1226, 1, 1536)
```

All primary runs used group-safe leave-one-fight OOF through the same
`fight_group(...)`/`matched_attribute_labels(...)` path as the existing ViT
evaluator. No Kaggle upload was run. All CUDA commands explicitly used
`CUDA_VISIBLE_DEVICES=1` and all primary results used the single fixed seed
`--seeds 17`.

I did not pursue lightly-unfreezing VideoMAE. The existing cache has `T=1`,
the request's main target was frozen-representation head tuning, and the head
search found a material local OOF gain without adding a long clip fine-tune
job or new train-on-all inference surface.

## Best result

Primary stack baseline:

```text
data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_ptype_m028_rows_20260522.csv
macro=0.408457
score_punch_type=0.216039
score_effectiveness=0.278732
```

Best single fixed-seed result:

```text
model=depth2 residual MLP
features=VideoMAE center_mean_std, no PCA
hidden_dim=192
dropout=0.25
class_weight_power=0.8
loss=focal
focal_gamma=1.5
temperature=1.0
seed=17
margin=0.10
macro=0.411357
delta_vs_clip_ptype_base=+0.002900
score_effectiveness=0.314986
wins_vs_clip_ptype_base=10/13
changed_effectiveness_rows=421
```

This beats the historical current-best depth1/depth2 3-seed notes, but the
primary comparison is now single fixed seed per coordination update:

| run | seed policy | margin | macro | score_effectiveness | delta vs CLIP ptype base |
|---|---:|---:|---:|---:|---:|
| CLIP punch-type base | n/a | n/a | 0.408457 | 0.278732 | +0.000000 |
| historical depth1 CE MLP | 17,29,43 | 0.30 | 0.409796 | 0.295470 | +0.001339 |
| historical depth2 residual CE | 17,29,43 | 0.12 | 0.409799 | 0.295516 | +0.001342 |
| depth1 CE MLP recheck | 17 | 0.28 | 0.410176 | 0.300220 | +0.001719 |
| depth2 residual CE | 17 | 0.10 | 0.410915 | 0.309456 | +0.002458 |
| depth2 residual focal | 17 | 0.10 | 0.411357 | 0.314986 | +0.002900 |

Best dense margin check:

```text
margin=0.000 macro=0.408917 effectiveness=0.284491 changed=594
margin=0.010 macro=0.409143 effectiveness=0.287306 changed=569
margin=0.020 macro=0.409478 effectiveness=0.291499 changed=553
margin=0.050 macro=0.409927 effectiveness=0.297105 changed=495
margin=0.080 macro=0.410225 effectiveness=0.300835 changed=458
margin=0.085 macro=0.410491 effectiveness=0.304166 changed=452
margin=0.090 macro=0.410967 effectiveness=0.310114 changed=438
margin=0.095 macro=0.411223 effectiveness=0.313315 changed=429
margin=0.100 macro=0.411357 effectiveness=0.314986 changed=421
margin=0.105 macro=0.411254 effectiveness=0.313701 changed=410
margin=0.110 macro=0.411170 effectiveness=0.312645 changed=405
margin=0.115 macro=0.410932 effectiveness=0.309674 changed=397
margin=0.120 macro=0.410215 effectiveness=0.300710 changed=390
```

`margin=0` was explicitly checked after the main run because the dataset is
small. It was positive versus the CLIP punch-type base, but much worse than
`margin=0.10`: it over-applied the head and changed 594 effectiveness rows.

Generated OOF rows:

```text
data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_d2res_h192_cw08_focal15_m010_seed17_rows_20260522.csv
```

## Diagnostics

Raw model diagnostics for the best run, before margin gating:

```text
baseline effectiveness accuracy=0.544244 balanced_acc=0.459302 macro_f1=0.445687 weighted_f1=0.545601
raw_model effectiveness accuracy=0.517602 balanced_acc=0.441799 macro_f1=0.436377 weighted_f1=0.526030

raw_model blocked precision=0.242424 recall=0.273504 f1=0.257028 support=117
raw_model landed  precision=0.675277 recall=0.597064 f1=0.633766 support=613
raw_model miss    precision=0.387268 recall=0.454829 f1=0.418338 support=321

raw_model confusion labels: blocked,landed,miss
true blocked: 32,47,38
true landed:  54,366,193
true miss:    46,129,146
```

The raw classifier is not better as a direct replacement; the gain comes from
probability-margin gating against the existing fixed-row effectiveness column.

Per-video deltas at margin 0.10 versus the CLIP punch-type base:

```text
agn_003 macro_delta=+0.004376 effectiveness_delta=+0.054696
agn_004 macro_delta=-0.003489 effectiveness_delta=-0.043619
agn_010 macro_delta=+0.002555 effectiveness_delta=+0.031941
agn_023 macro_delta=+0.004261 effectiveness_delta=+0.053266
agn_024 macro_delta=+0.002361 effectiveness_delta=+0.029516
agn_025 macro_delta=+0.000740 effectiveness_delta=+0.009245
agn_056 macro_delta=+0.005436 effectiveness_delta=+0.067955
agn_057 macro_delta=-0.003026 effectiveness_delta=-0.037826
agn_058 macro_delta=+0.005600 effectiveness_delta=+0.070000
agn_069 macro_delta=+0.004740 effectiveness_delta=+0.059249
agn_070 macro_delta=+0.011966 effectiveness_delta=+0.149573
agn_071 macro_delta=-0.000906 effectiveness_delta=-0.011331
agn_072 macro_delta=+0.003092 effectiveness_delta=+0.038647
```

## Negative checks

Single fixed-seed runs around the best shape:

| run | best margin | macro | score_effectiveness | note |
|---|---:|---:|---:|---|
| depth2 residual CE, cw=0.7 | 0.22 | 0.410675 | 0.306459 | below cw=0.8 |
| capped CE weights, min=0.6 max=1.6 | 0.28 | 0.409908 | 0.296877 | cap hurt |
| effective-number weights, beta=0.99 | 0.50 | 0.408653 | 0.281190 | weak |
| macro-F1 early stopping | 0.45 | 0.407808 | 0.270622 | weak |
| focal gamma 1.0 | 0.12 | 0.411036 | 0.310976 | good, below gamma 1.5 |
| focal gamma 2.0 | 0.10 | 0.410601 | 0.305539 | below gamma 1.5 |
| focal gamma 1.5, temp=1.25 | 0.08 | 0.411329 | 0.314637 | near tie, below temp=1.0 |
| focal gamma 1.5, center-only pooling | 0.22 | 0.409569 | 0.292638 | T=1 pooling simplification did not help |
| focal gamma 1.5, dropout=0.15 | 0.22 | 0.410140 | 0.299767 | below dropout 0.25 |
| focal gamma 1.5, dropout=0.35 | 0.18 | 0.410565 | 0.305086 | below dropout 0.25 |
| focal gamma 1.5, hidden=128 | 0.08 | 0.409747 | 0.294861 | below hidden 192 |
| focal gamma 1.5, hidden=256 | 0.35 | 0.409434 | 0.290953 | below hidden 192 |

## Commands run

Smoke:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 .venv/bin/python tools/evaluate_videomae_effectiveness_head.py --predictions data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_ptype_m028_rows_20260522.csv --feature-caches data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_current_oof_cuda_preproc_20260522.npz --margins 0.3 --pca-components 0 --feature-pooling center --hidden-dim 64 --mlp-depth 1 --dropout 0.25 --class-weight-power 0.7 --label-smoothing 0.04 --epochs 2 --patience 1 --seeds 17 --device cuda
```

Depth1 CE recheck:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 .venv/bin/python tools/evaluate_videomae_effectiveness_head.py --predictions data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_ptype_m028_rows_20260522.csv --feature-caches data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_current_oof_cuda_preproc_20260522.npz --margins 0.05,0.08,0.1,0.12,0.15,0.18,0.2,0.22,0.25,0.28,0.3,0.35,0.4,0.45,0.5 --pca-components 0 --feature-pooling center_mean_std --hidden-dim 192 --mlp-depth 1 --dropout 0.25 --lr 3e-4 --weight-decay 1e-3 --class-weight-power 0.7 --label-smoothing 0.04 --loss ce --temperature 1.0 --epochs 220 --patience 28 --seeds 17 --device cuda --report-video-deltas > data/processed/vit_features/effectiveness_head_d1_h192_cw07_ls004_seed17_20260522.log 2>&1
```

Best CE comparison:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 .venv/bin/python tools/evaluate_videomae_effectiveness_head.py --predictions data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_ptype_m028_rows_20260522.csv --feature-caches data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_current_oof_cuda_preproc_20260522.npz --margins 0.05,0.08,0.1,0.12,0.15,0.18,0.2,0.22,0.25,0.28,0.3,0.35,0.4,0.45,0.5 --pca-components 0 --feature-pooling center_mean_std --hidden-dim 192 --mlp-depth 2 --residual-head --dropout 0.25 --lr 3e-4 --weight-decay 1e-3 --class-weight-power 0.8 --label-smoothing 0.04 --loss ce --temperature 1.0 --epochs 220 --patience 28 --seeds 17 --device cuda --report-video-deltas > data/processed/vit_features/effectiveness_head_d2res_h192_cw08_ls004_seed17_20260522.log 2>&1
```

Best focal run and row write:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 .venv/bin/python tools/evaluate_videomae_effectiveness_head.py --predictions data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_ptype_m028_rows_20260522.csv --feature-caches data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_current_oof_cuda_preproc_20260522.npz --margins 0.085,0.09,0.095,0.1,0.105,0.11,0.115,0.12 --pca-components 0 --feature-pooling center_mean_std --hidden-dim 192 --mlp-depth 2 --residual-head --dropout 0.25 --lr 3e-4 --weight-decay 1e-3 --class-weight-power 0.8 --loss focal --focal-gamma 1.5 --temperature 1.0 --epochs 220 --patience 28 --seeds 17 --device cuda --report-video-deltas --write-oof-rows data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_d2res_h192_cw08_focal15_m010_seed17_rows_20260522.csv --write-margin 0.1 > data/processed/vit_features/effectiveness_head_d2res_h192_cw08_focal15_seed17_refine_write_20260522.log 2>&1
```

Other sweep logs are in:

```text
data/processed/vit_features/effectiveness_head_d2res_h192_cw08_focal15_margin0_seed17_20260522.log
data/processed/vit_features/effectiveness_head_d2res_h192_cw07_ls004_seed17_20260522.log
data/processed/vit_features/effectiveness_head_d2res_h192_cw08_cap06_16_ls004_seed17_20260522.log
data/processed/vit_features/effectiveness_head_d2res_h192_effnum099_p08_ls004_seed17_20260522.log
data/processed/vit_features/effectiveness_head_d2res_h192_cw08_ls004_macrostop_seed17_20260522.log
data/processed/vit_features/effectiveness_head_d2res_h192_cw08_focal10_seed17_20260522.log
data/processed/vit_features/effectiveness_head_d2res_h192_cw08_focal15_seed17_20260522.log
data/processed/vit_features/effectiveness_head_d2res_h192_cw08_focal20_seed17_20260522.log
data/processed/vit_features/effectiveness_head_d2res_h192_cw08_focal15_temp125_seed17_20260522.log
data/processed/vit_features/effectiveness_head_d2res_center_h192_cw08_focal15_seed17_20260522.log
data/processed/vit_features/effectiveness_head_d2res_h192_do015_cw08_focal15_seed17_20260522.log
data/processed/vit_features/effectiveness_head_d2res_h192_do035_cw08_focal15_seed17_20260522.log
data/processed/vit_features/effectiveness_head_d2res_h128_cw08_focal15_seed17_20260522.log
data/processed/vit_features/effectiveness_head_d2res_h256_cw08_focal15_seed17_20260522.log
```
