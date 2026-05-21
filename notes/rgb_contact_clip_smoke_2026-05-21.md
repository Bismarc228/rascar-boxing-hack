# RGB Contact Clip Smoke - 2026-05-21

Goal: test a genuinely temporal RGB/contact signal on the current best fixed
rows, instead of repeating the killed still-image CLIP/ResNet filters.

No Kaggle upload.

## Tool

Added:

```text
tools/evaluate_rgb_contact_clip.py
```

The tool extracts frozen RGB clip embeddings as `N x T x D`, trains a small
temporal GRU head in leave-one-fight OOF folds, and evaluates fixed-row
keep/drop plus frame-offset correction. GPU work is run with
`CUDA_VISIBLE_DEVICES=1`; feature extraction uses cached `.npz` files.

The first implementation used OpenCV random seek per event frame and took about
132 seconds for the first validation video. It was changed to sequential decode
within each video; the same first video then took about 12 seconds.

## Current Source

Baseline OOF:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_gate_oof.csv
baseline score=0.401483
fighter=0.552581
time=0.545894
fp=0.084963
n=1275
```

## Union Crop Smoke

Command shape:

```text
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/evaluate_rgb_contact_clip.py \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_gate_oof.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --feature-cache data/processed/rgb_features/contact_vitb16_union_clip8_stride2_current_best.npz \
  --model-name vit_base_patch16_clip_224.openai \
  --clip-len 8 \
  --frame-stride 2 \
  --crop-modes union \
  --decode-mode sequential \
  --epochs 24 \
  --hidden 192
```

Cache:

```text
data/processed/rgb_features/contact_vitb16_union_clip8_stride2_current_best.npz
shape=(1275, 8, 768)
```

Best non-noop result:

```text
score=0.385960
delta=-0.015523
n_rows=1196
dropped=79
threshold=0.05
max_shift=0
shift_scale=0.0
```

Offset-only check with `threshold=0.0` also regressed:

```text
no-op=0.401483
best shifted=0.398441
delta=-0.003042
max_shift=2
shift_scale=0.25
```

## Attacker/Opponent Crop Smoke

Command shape:

```text
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/evaluate_rgb_contact_clip.py \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_gate_oof.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --feature-cache data/processed/rgb_features/contact_vitb16_attacker_opponent_clip8_stride2_current_best.npz \
  --model-name vit_base_patch16_clip_224.openai \
  --clip-len 8 \
  --frame-stride 2 \
  --crop-modes attacker_opponent \
  --decode-mode sequential \
  --epochs 24 \
  --hidden 192
```

Cache:

```text
data/processed/rgb_features/contact_vitb16_attacker_opponent_clip8_stride2_current_best.npz
shape=(1275, 8, 768)
```

Best non-noop results:

```text
drop-only: score=0.398062, delta=-0.003422, n_rows=1228, dropped=47, threshold=0.05
offset-only: score=0.398528, delta=-0.002955, max_shift=2, shift_scale=0.25
```

## Decision

Kill frozen ViT-B fixed-row contact as a direct keep/drop or offset-correction
branch. It does not beat the current motion-gated source, and both crop modes
regress versus the no-op baseline.

The tool is still useful for the next RGB direction, but only if the target
changes: candidate-level contact over a wider pose pool, or a supervised
fine-tuned/video model with enough positives and negatives. Do not spend
submissions on this fixed-row frozen-embedding branch.
