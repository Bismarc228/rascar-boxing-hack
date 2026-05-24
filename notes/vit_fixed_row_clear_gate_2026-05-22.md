# ViT fixed-row clear gate, 2026-05-22

Goal: test whether cached CLIP/VideoMAE event features can improve the
`clear` / `fp_penalty` component by dropping likely false-positive fixed rows.
This is local OOF research only. No Kaggle upload was made.

## Setup

Base rows:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_rgb_effectiveness_margin02_oof.csv
```

Feature caches:

```text
data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_20260522.npz
data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_current_oof_cuda_preproc_20260522.npz
```

Command shape:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 .venv/bin/python tools/evaluate_vit_fixed_row_gate.py \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_rgb_effectiveness_margin02_oof.csv \
  --feature-caches data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_20260522.npz \
                   data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_current_oof_cuda_preproc_20260522.npz \
  --model hgb \
  --label-positive scorable
```

Labels from local matching:

```text
tp_scorable 1061
tp_time_only 52
fp 113
```

Baseline:

```text
macro 0.407232
score_time 0.537088
score_fighter 0.544665
score_punch_type 0.203789
score_effectiveness 0.278732
score_hand 0.526763
score_target 0.485253
fp_penalty 0.073644
n_rows 1226
```

Oracle unmatched-FP drop:

```text
macro 0.480698
delta +0.073466
fp_penalty 0.000000
n_rows 1113
wins 13/13
```

The oracle confirms large `clear`/FP headroom, but not that the ViT features can
find it.

## Results

HGB, positive label = `tp_scorable`:

```text
best threshold 0.60
macro 0.407798
delta +0.000567
score_time 0.537088 -> 0.536677
score_fighter 0.544665 -> 0.544225
fp_penalty 0.073644 -> 0.072731
wins 2/13
n_kept 1223
n_dropped 3
```

OOF probability separation is weak:

```text
p_keep_tp_scorable mean=0.9273 p10=0.8580 p50=0.9433 p90=0.9758
p_keep_tp_time_only mean=0.8934 p10=0.8094 p50=0.9093 p90=0.9649
p_keep_fp mean=0.9011 p10=0.8241 p50=0.9245 p90=0.9748
```

Logreg, positive label = `tp_scorable`:

```text
best threshold 0.10
macro 0.407686
delta +0.000454
score_time 0.537088 -> 0.523229
fp_penalty 0.073644 -> 0.060554
wins 8/13
n_kept 1175
n_dropped 51
```

This reduces FP penalty more, but loses too much time/matched evidence.

HGB, positive label = any matched row (`tp_scorable` or `tp_time_only`):

```text
best nontrivial threshold 0.55
macro 0.406708
delta -0.000523
n_dropped 1
```

## Interpretation

- Standalone CLIP+VideoMAE fixed-row clear gate is not a strong FP detector.
- The visual features assign high keep probabilities to many FP rows, so the
  positive thresholds either drop almost nothing or start deleting true matched
  evidence.
- The huge oracle gap says `clear/fp_penalty` is still worth attacking, but the
  next version needs pose/count/timing context features plus visual features,
  not transformer features alone.
- No test artifact or Kaggle upload is justified by this result.
