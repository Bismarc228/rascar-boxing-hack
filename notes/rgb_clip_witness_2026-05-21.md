# RGB CLIP Witness - 2026-05-21

Scope: fixed-row clear/drop witness on the current strongest OOF row source.
No Kaggle upload.

## Change

`tools/evaluate_rgb_event_filter.py` now uses each timm model's `default_cfg`
normalization instead of hard-coded ImageNet mean/std. This is needed for CLIP
and similar pretrained ViT models.

## Experiment

Command:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/evaluate_rgb_event_filter.py \
  --predictions data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_hgb_p024_oof.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --feature-cache data/processed/rgb_features/clip_vitb16_union_offsets_m8_0_p8_hybrid_rival_exchange_p024.npz \
  --model-name vit_base_patch16_clip_224.openai \
  --device auto \
  --image-size 224 \
  --batch-size 64 \
  --frame-offsets=-8,0,8
```

Environment:

- `CUDA_VISIBLE_DEVICES=1`
- visible CUDA device count: `1`
- model: `vit_base_patch16_clip_224.openai`
- crop: union of both fighter bboxes
- frames: `t-8,t,t+8`

## Result

Baseline row source:

```text
score=0.390962
time=0.512045
fp=0.056774
n=1185
pos_rate=0.9190
```

Best CLIP drop threshold:

```text
threshold=0.05
score=0.388244
delta=-0.002719
n_rows=1176
dropped=9
```

The filter keeps assigning high probabilities to negatives; for example
`neg_mean_p` is high across videos and often close to or above `pos_mean_p`.

## Decision

- Frozen CLIP union-crop embeddings are killed as a direct fixed-row
  clear/drop witness.
- Do not extract test CLIP features or generate a CLIP-filtered submission.
- RGB is not fully dead, but the next RGB attempt must use a stronger event
  representation than union-crop still embeddings, such as short clip/contact
  modeling or explicit attacker/opponent crops.

## Structured Crop Follow-Up

Added `--crop-mode` to `tools/evaluate_rgb_event_filter.py`:

- `union`
- `full`
- `attacker`
- `opponent`
- `attacker_opponent`

Follow-up command:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/evaluate_rgb_event_filter.py \
  --predictions data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_hgb_p024_oof.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --feature-cache data/processed/rgb_features/clip_vitb16_attacker_opponent_offsets_m8_0_p8_hybrid_rival_exchange_p024.npz \
  --model-name vit_base_patch16_clip_224.openai \
  --device auto \
  --image-size 224 \
  --batch-size 64 \
  --frame-offsets=-8,0,8 \
  --crop-mode attacker_opponent
```

Result:

```text
baseline=0.390962
best_structured_clip=0.389116
delta=-0.001846
threshold=0.05
dropped=11
```

Decision update:

- Structured attacker/opponent still-image CLIP is also killed as a direct
  fixed-row clear/drop witness.
- The remaining RGB path must use temporal contact modeling, not frozen
  per-frame CLIP embeddings.
