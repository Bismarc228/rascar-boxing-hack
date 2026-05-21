# RGB Fixed-Row Attribute Witness - 2026-05-21

Goal: test whether cached RGB clip features can improve event attributes while
keeping the current best rows, frames, fighters, and counts fixed.

No Kaggle upload and no test CSV generation.

## Tool

Added:

```text
tools/evaluate_rgb_fixed_row_attribute_model.py
```

The tool:

1. Reconstructs the same yolo26x sequence candidate pool used by the full RGB
   bridge cache.
2. Loads cached ViT-B/16 clip features.
3. Matches fixed prediction rows back to cached candidates.
4. Trains OOF fight-group attribute heads from RGB mean/std features plus current
   row context.
5. Applies only selected attribute columns; timing, count, fighter, and clear
   stay fixed.

## Main Run

Input rows:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_appearance_oof.csv
baseline=0.405200
```

RGB cache:

```text
data/processed/rgb_features/seq_bridge_contact_vitb16_union_clip4_stride2_pool1800_ab.npz
features=(22691, 4, 768)
```

Command:

```text
.venv/bin/python tools/evaluate_rgb_fixed_row_attribute_model.py \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_appearance_oof.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --rgb-feature-cache data/processed/rgb_features/seq_bridge_contact_vitb16_union_clip4_stride2_pool1800_ab.npz \
  --max-candidates-per-video 1800 \
  --candidate-match-window 6 \
  --label-window 12 \
  --pca-components 64 \
  --logreg-c 0.35 \
  --effectiveness-margins 0.0,0.1,0.2,0.3 \
  --write-variant effectiveness \
  --write-effectiveness-margin 0.2 \
  --write-oof-rows data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_rgb_effectiveness_margin02_oof.csv
```

Cache alignment:

```text
rgb_feature_rows=22691
pred_rows=1226
matched=1226
match_rate=1.0000
x_shape=(1226, 1547)
```

Result:

```text
variant        score     delta      wins  n_changed
punch_type     0.401541  -0.003659  3     782
effectiveness  0.406345  +0.001145  10    566
ptype_eff      0.402687  -0.002514  4     963
hand_target    0.385520  -0.019680  0     821
all_attrs      0.383006  -0.022194  0     1137
eff_margin_0.1 0.406567  +0.001366  10    453
eff_margin_0.2 0.407232  +0.002031  10    367
eff_margin_0.3 0.406304  +0.001103  8     271
```

Scorer verification:

```text
.venv/bin/python tools/score_predictions.py \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_rgb_effectiveness_margin02_oof.csv \
  --video-keys-from-predictions

macro_score=0.407232
```

The saved OOF rows are:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_rgb_effectiveness_margin02_oof.csv
```

## Stability Checks

Two small preselected controls:

```text
pca=64, C=0.15: margin_0.2=0.407232, ungated=0.406345
pca=32, C=0.35: margin_0.2=0.405830, margin_0.3=0.406302, ungated=0.405445
```

The effect is real enough to save, but not strong enough to justify a standalone
upload. The confidence margin reduces churn and improves the main setting, but
the best margin is still somewhat sensitive to RGB head capacity.

## Other Attribute Margins

The evaluator was extended to print confidence-margin variants for other
attribute columns. With the main `pca=64, C=0.35` setting:

```text
punch_type_margin_0.3  0.404246  delta=-0.000955  changed=406
hand_margin_0.3        0.401078  delta=-0.004122  changed=225
target_margin_0.3      0.402905  delta=-0.002295  changed=160
```

Decision: only `effectiveness` is currently useful. Do not combine RGB
`punch_type`, `hand`, or `target` replacements with the current best source.

## Decision

This is the new best local OOF row source at `0.407232`, but it is an attribute
micro-signal only:

- timing, count, fighter, hand, target, and punch type are unchanged for the
  saved `effectiveness` variant;
- 367 validation rows change `effectiveness`;
- no test generator or test CSV was created;
- no Kaggle upload should be made from this branch without explicit approval and
  a validated test-side generation plan.
