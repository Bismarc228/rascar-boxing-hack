# Combined fixed-row clear gate, 2026-05-22

Goal: test `clear` / FP reduction after the current transformer component
stack, using leakage-safe leave-one-fight gates over pose-context and cached
CLIP/VideoMAE features. This is local OOF research only. No Kaggle upload was
made.

## Source Stack

Rows before clear gating:

```text
data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_direct_h192_w07_ls004_m03_fighter_target_m095_rows_20260522.csv
```

Component source:

```text
CLIP punch_type margin 0.28
VideoMAE direct MLP effectiveness margin 0.3
pose-rival fighter flip hgb window 4 same_hand threshold 0.7
VideoMAE target margin 0.95
```

Baseline score:

```text
macro              0.410125
score_time         0.537088
score_fighter      0.546042
score_punch_type   0.216039
score_effectiveness 0.295470
score_hand         0.526763
score_target       0.486148
fp_penalty         0.073644
n_rows             1226
```

Oracle unmatched-FP drop:

```text
macro       0.483591
delta       +0.073466
fp_penalty  0.000000
n_rows      1113
wins        13/13
```

The oracle confirms very large `clear` headroom, but the learned gate must avoid
dropping matched evidence.

## Tool

Added:

```text
tools/evaluate_combined_fixed_row_gate.py
```

It compares `pose`, `vit`, and `pose_vit` feature modes on the same fixed rows
and fight-group OOF folds. It intentionally has its own cached-feature loader
instead of importing from the neural-head evaluator, because the neural file is
actively used for independent worker experiments.

Main sweep:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python tools/evaluate_combined_fixed_row_gate.py \
  --predictions data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_direct_h192_w07_ls004_m03_fighter_target_m095_rows_20260522.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --feature-caches \
    data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_20260522.npz \
    data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_current_oof_cuda_preproc_20260522.npz \
  --feature-modes pose,vit,pose_vit \
  --models hgb,logreg \
  --label-positive scorable \
  --thresholds 0.02,0.04,0.06,0.08,0.10,0.12,0.15,0.18,0.22,0.26,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.80,0.90 \
  --top-k 30
```

## Single-Model Results

Initial best max-macro single-model point:

```text
feature_mode vit
model logreg
threshold 0.12
macro 0.410125 -> 0.410945
delta +0.000820
wins 9/13
n_rows 1171
n_dropped 55
fp_penalty 0.073644 -> 0.059393  delta -0.014250
score_time 0.537088 -> 0.522804  delta -0.014284
score_fighter 0.546042 -> 0.532136  delta -0.013906
score_punch_type 0.216039 -> 0.206994  delta -0.009045
score_effectiveness 0.295470 -> 0.284553  delta -0.010918
score_hand 0.526763 -> 0.513051  delta -0.013713
score_target 0.486148 -> 0.471032  delta -0.015116
```

Drop audit for `vit/logreg/t0.12`:

```text
fp dropped           20
tp_scorable dropped  32
tp_time_only dropped 3
```

This is the top local macro, but it is a blunt gate: the gain comes from
`fp_penalty`, while every matched-row component regresses.

Softer point:

```text
feature_mode vit
model logreg
threshold 0.02
macro 0.410125 -> 0.410902
delta +0.000777
wins 6/13
n_rows 1204
n_dropped 22
fp_penalty 0.073644 -> 0.067161  delta -0.006482
score_time 0.537088 -> 0.530782  delta -0.006306
score_fighter 0.546042 -> 0.541951  delta -0.004091
score_punch_type 0.216039 -> 0.211563  delta -0.004475
score_effectiveness 0.295470 -> 0.288240  delta -0.007230
score_hand 0.526763 -> 0.521295  delta -0.005468
score_target 0.486148 -> 0.479813  delta -0.006335
```

Drop audit for `vit/logreg/t0.02`:

```text
fp dropped           11
tp_scorable dropped  11
tp_time_only dropped 0
```

This is almost the same aggregate score with lower damage. If this gate is used
as a component source, prefer `t0.02` unless a private-risk stack explicitly
wants max local macro.

Other checked points:

```text
pose hgb threshold 0.35/0.40
macro 0.410759
delta +0.000634
n_dropped 7

pose_vit hgb threshold 0.45
macro 0.410543
delta +0.000418
n_dropped 1
```

The simple concatenated `pose_vit` HGB did not improve over standalone ViT
logreg or pose HGB. The likely issue is tiny OOF data with a very high-dimensional
visual block, not lack of signal.

## Probability-Ensemble Gate

A follow-up tested independent OOF probabilities from:

```text
left:  pose hgb
right: vit logreg
mode:  drop_if_both_low
```

This keeps a row if either model is confident enough, and drops only rows that
both models dislike. It is more selective than the single ViT-logreg threshold.

First checked ensemble:

```text
pose_hgb threshold       0.90
vit_logreg threshold     0.08
mode                     drop_if_both_low
macro                    0.410125 -> 0.414485
delta_vs_stack           +0.004360
delta_vs_original        +0.007253
wins_vs_stack            7/13
wins_vs_original         10/13
n_rows                   1204
n_dropped                22
fp_penalty               0.073644 -> 0.065861  delta -0.007782
score_time               0.537088 -> 0.533465  delta -0.003623
score_fighter            0.546042 -> 0.542121  delta -0.003922
score_punch_type         0.216039 -> 0.212849  delta -0.003190
score_effectiveness      0.295470 -> 0.292170  delta -0.003300
score_hand               0.526763 -> 0.525590  delta -0.001173
score_target             0.486148 -> 0.483262  delta -0.002886
```

Drop audit:

```text
fp dropped           12
tp_scorable dropped  9
tp_time_only dropped 1
```

Per-video deltas versus the source stack:

```text
agn_003 +0.024507
agn_004 -0.003493
agn_010 +0.000000
agn_023 +0.000000
agn_024 +0.000000
agn_025 +0.006343
agn_056 +0.000000
agn_057 -0.003631
agn_058 +0.004755
agn_069 +0.000859
agn_070 +0.005746
agn_071 +0.005765
agn_072 +0.015831
```

Command for the OOF row artifact:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python tools/evaluate_combined_fixed_row_gate.py \
  --predictions data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_direct_h192_w07_ls004_m03_fighter_target_m095_rows_20260522.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --feature-caches \
    data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_20260522.npz \
    data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_current_oof_cuda_preproc_20260522.npz \
  --feature-modes pose,vit \
  --models hgb,logreg \
  --ensemble-pair pose:hgb,vit:logreg \
  --ensemble-modes drop_if_both_low \
  --label-positive scorable \
  --thresholds 0.02,0.08,0.9 \
  --top-k 20 \
  --write-oof-rows data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb09_vitlogreg008_rows_20260522.csv \
  --write-ensemble-mode drop_if_both_low \
  --write-left-threshold 0.9 \
  --write-right-threshold 0.08
```

Fine threshold sweep around the first ensemble found a stronger max-local
point:

```text
pose_hgb threshold       0.94
vit_logreg threshold     0.12
mode                     drop_if_both_low
macro                    0.410125 -> 0.415261
delta_vs_stack           +0.005136
delta_vs_original        +0.008029
wins_vs_stack            8/13
n_rows                   1188
n_dropped                38
fp_penalty               0.073644 -> 0.061715  delta -0.011929
score_time               0.537088 -> 0.530111  delta -0.006977
score_fighter            0.546042 -> 0.537953  delta -0.008089
score_punch_type         0.216039 -> 0.211716  delta -0.004323
score_effectiveness      0.295470 -> 0.289848  delta -0.005623
score_hand               0.526763 -> 0.521370  delta -0.005393
score_target             0.486148 -> 0.478143  delta -0.008005
```

Generated fine-sweep row artifact:

```text
data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv
```

Generated row artifacts:

```text
data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_vit_logreg_clear_t012_rows_20260522.csv
data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_vit_logreg_clear_t002_rows_20260522.csv
data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb09_vitlogreg008_rows_20260522.csv
data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv
```

## Conclusion

`clear` remains the largest unsolved component by oracle headroom. The current
learned gates are positive locally, but not clean: they trade FP reduction for
some matched-evidence loss. The probability-ensemble `drop_if_both_low` rule is
the best current clear source because it drops fewer TP per FP than the blunt
single ViT-logreg gate. The max-local fine-sweep point reaches `0.415261` OOF,
while the softer first ensemble reaches `0.414485` with fewer dropped rows. This
is now the strongest local fixed-row component stack, but still private-risk
because the improvement is a trade against dropped matched evidence. Next work
should make the gate more selective, for example by adding per-video thresholds
or a second-stage "recover dropped TP" model.
