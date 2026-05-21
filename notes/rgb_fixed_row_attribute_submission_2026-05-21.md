# RGB Fixed-Row Attribute Test Artifact - 2026-05-21

Goal: make the RGB effectiveness signal reproducible on test rows without using
the tracklet-appearance micro-source that has no test generator.

No Kaggle upload.

## Validation Source

Input rows:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_gate_exchange_side_oof.csv
baseline=0.404915
```

This is the nearest current source with an already validated test analog.

RGB cache:

```text
data/processed/rgb_features/seq_bridge_contact_vitb16_union_clip4_stride2_pool1800_ab.npz
features=(22691, 4, 768)
```

OOF result:

```text
effectiveness_margin_0.2
score=0.406939
delta=+0.002024
wins=10
changed=369
```

Saved OOF rows:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_gate_exchange_side_rgb_effectiveness_margin02_oof.csv
```

Scorer verification:

```text
.venv/bin/python tools/score_predictions.py \
  --ground-truth data/raw/train/punches.csv \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_gate_exchange_side_rgb_effectiveness_margin02_oof.csv \
  --video-keys-from-predictions

macro_score=0.406939
```

## Test Maker

Added:

```text
tools/make_rgb_fixed_row_attribute_submission.py
```

The maker:

1. Reconstructs the yolo26x candidate pool for train and test.
2. Loads or extracts matching ViT-B/16 clip features.
3. Trains a full-data RGB attribute model on matched train rows.
4. Applies only a confidence-margin replacement for one fixed-row attribute.
5. Optionally protects named test videos such as `agn_038`.

Test RGB cache produced on GPU1:

```text
CUDA_VISIBLE_DEVICES=1
data/processed/rgb_features/seq_bridge_contact_vitb16_union_clip4_stride2_pool1800_test.npz
features=(16200, 4, 768)
clip_offsets=[-4, -2, 0, 2]
crop_modes=["union"]
model_name=vit_base_patch16_clip_224.openai
```

## Test Artifacts

Full audio-exchange source:

```text
input=submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_audio_exchange_side_OFFLINE_CANDIDATE.csv
output=submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_audio_exchange_side_rgb_eff_m02_OFFLINE_CANDIDATE.csv
test_clear_rows=714
matched_test=712
changed=195
Validation passed.
```

Changed effectiveness labels by video:

```text
agn_037:23
agn_038:36
agn_039:14
agn_047:36
agn_048:16
agn_049:13
agn_062:21
agn_063:19
agn_064:17
```

Protected root-splice source with `agn_038` unchanged:

```text
input=submissions/hybrid_root_seqrepeat_exchange_attr_motion_audio_exchange_side_noagn038_OFFLINE_CANDIDATE.csv
output=submissions/hybrid_root_seqrepeat_exchange_attr_motion_audio_exchange_side_rgb_eff_m02_noagn038_OFFLINE_CANDIDATE.csv
protected=agn_038
test_clear_rows=712
matched_test=709
changed=165
protected_changed=32
Validation passed.
```

Changed effectiveness labels by video:

```text
agn_037:18
agn_039:27
agn_047:36
agn_048:18
agn_049:16
agn_062:21
agn_063:19
agn_064:10
```

The protected artifact changes 478 ids versus the public root anchor, mostly
because it inherits the sequence/audio/exchange source on non-`agn_038` videos.

## Decision

Keep these artifacts for ensemble/private-risk bookkeeping. The validation lift
is real on the test-capable source (`0.404915 -> 0.406939`), but it is
attribute-only and does not by itself solve the known public-transfer risk of
full sequence-derived sources. Do not upload without explicit approval.
