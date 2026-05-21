# Sequence RGB Contact Bridge - 2026-05-21

Goal: connect the live candidate-level RGB/contact signal to the stronger
sequence-TCN candidate scorer, instead of using RGB on raw pose scores or
fixed rows.

No Kaggle upload.

## Implementation

`tools/evaluate_pose_sequence_spotter.py` now has optional RGB-contact args:

```text
--rgb-contact-feature-cache
--rgb-contact-blend-alphas
--rgb-contact-model-name
--rgb-contact-clip-len
--rgb-contact-frame-stride
--rgb-contact-crop-modes
--rgb-contact-decode-mode
--rgb-contact-epochs
--rgb-contact-hidden
```

When `--rgb-contact-feature-cache` is provided, the sequence spotter:

1. Uses the same `packs[key].candidates` pre-NMS candidate pool as the sequence
   model.
2. Extracts or loads RGB clip features for those candidates.
3. Fits the temporal RGB contact head in the same leave-one-fight OOF grouping.
4. Blends RGB contact probability into sequence-TCN candidate scores before
   final threshold/NMS/count selection.

With no `--rgb-contact-feature-cache`, the old behavior is unchanged.

## Sanity Smoke

Command shape:

```text
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/evaluate_pose_sequence_spotter.py \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --pool-min-score 0.0 \
  --pool-nms-frames 2 \
  --max-candidates-per-video 250 \
  --epochs 1 \
  --chunks-per-epoch 100 \
  --hidden 32 \
  --layers 2 \
  --thresholds 0.4 \
  --nms-frames 10 \
  --cross-nms-frames 2 \
  --count-modes root_count \
  --count-multipliers 0.88 \
  --pose-priors 0.0 \
  --quiet \
  --rgb-contact-feature-cache data/processed/rgb_features/candidate_contact_vitb16_union_clip4_stride2_pool250.npz \
  --rgb-contact-clip-len 4 \
  --rgb-contact-frame-stride 2 \
  --rgb-contact-crop-modes union \
  --rgb-contact-epochs 1 \
  --rgb-contact-hidden 64 \
  --rgb-contact-blend-alphas 0.0,1.0
```

Sanity result:

```text
rgb_alpha=1.0 score=0.312313
rgb_alpha=0.0 score=0.309016
```

This only verifies that the optional bridge runs and the RGB blend can affect
sequence candidate scoring. It is not a competitive validation result because
the sequence model was intentionally undertrained (`1` epoch, `100` chunks).

## Next Useful Run

If spending more compute on this branch, use the repeated sequence-TCN recipe
that previously produced the strong local source, but add the cached RGB bridge:

```text
--max-candidates-per-video 400 or 1800
--epochs 8
--chunks-per-epoch 1600
--hidden 96
--layers 6
--seeds 41,42,43
--rgb-contact-blend-alphas 0.0,0.25,0.5,1.0
```

Stop unless the full OOF result approaches or beats the current best local
source (`0.401483`). Do not create a test CSV or upload from the sanity smoke.

## Medium Run

Command shape:

```text
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/evaluate_pose_sequence_spotter.py \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --witness-tracks-dirs data/processed/pose_tracks/val_yolo11s_conf035,data/processed/pose_tracks/val_yolo26l_conf035 \
  --max-candidates-per-video 400 \
  --seeds 41,42 \
  --epochs 6 \
  --chunks-per-epoch 1200 \
  --hidden 96 \
  --layers 6 \
  --thresholds 0.5,0.6,0.7 \
  --nms-frames 10 \
  --cross-nms-frames 2 \
  --snap-windows 4 \
  --count-modes root_count \
  --count-multipliers 0.88 \
  --pose-priors 0.0,0.2 \
  --rgb-contact-feature-cache data/processed/rgb_features/seq_bridge_contact_vitb16_union_clip4_stride2_pool400.npz \
  --rgb-contact-clip-len 4 \
  --rgb-contact-frame-stride 2 \
  --rgb-contact-crop-modes union \
  --rgb-contact-epochs 12 \
  --rgb-contact-hidden 128 \
  --rgb-contact-blend-alphas 0.0,0.25,0.5,1.0 \
  --write-best-rows data/processed/validation_rows/seq_rgb_contact_bridge_cap400_medium_oof.csv
```

Feature cache:

```text
data/processed/rgb_features/seq_bridge_contact_vitb16_union_clip4_stride2_pool400.npz
shape=(5200, 4, 768)
```

Best result:

```text
score=0.388112
pose_prior=0.2
rgb_alpha=0.25
threshold=0.7
nms=10
cross_nms=2
snap_window=4
count_mode=root_count
count_multiplier=0.88
n=1254
```

The corresponding rows were written to:

```text
data/processed/validation_rows/seq_rgb_contact_bridge_cap400_medium_oof.csv
```

Verification:

```text
python3 tools/score_predictions.py \
  --predictions data/processed/validation_rows/seq_rgb_contact_bridge_cap400_medium_oof.csv \
  --video-keys-from-predictions

macro_score=0.388112
```

Within this medium run, RGB helps the sequence scorer: the best `rgb_alpha=0.0`
configuration in the printed grid was `0.376648`, while `rgb_alpha=0.25`
reached `0.388112`. The absolute score is still below the current best local
source (`0.401483`) and below the postprocessed sequence-repeat branch, so this
does not justify generating a test CSV.

## Medium Decision

Do not promote the cap-400 RGB bridge branch. The bridge is technically working
and RGB provides useful ranking signal inside a weaker sequence setup, but the
OOF score is not close enough to the current best to spend more submissions or
create a test artifact.
