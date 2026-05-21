# Sequence Audio Contact Bridge - 2026-05-21

Goal: move the positive audio signal from fixed selected rows into the sequence
candidate scorer before NMS/count selection. This tests audio as a learned
candidate-level auxiliary prior, not audio-only detection or hard snapping.

No Kaggle upload.

## Tool Update

`tools/evaluate_pose_sequence_spotter.py` now has optional audio-contact args:

```text
--audio-contact
--audio-contact-sample-rate
--audio-contact-label-window
--audio-contact-feature-windows
--audio-contact-offsets
--audio-contact-epochs
--audio-contact-hidden
--audio-contact-blend-alphas
```

When enabled, the script extracts ffmpeg-based per-frame audio features, builds
candidate-level audio feature vectors, fits an OOF contact head by fight group,
and blends audio probability into sequence candidate scores before final NMS.

With no `--audio-contact`, existing behavior is unchanged except result rows now
include `audio_alpha=0.0`.

## Smoke Run

Cap-250, one-epoch infrastructure smoke:

```text
best audio_alpha=0.0 score=0.331280
best audio_alpha=1.0 score=0.326928
```

The hook runs, but the tiny undertrained setup regresses with audio.

## Medium Run

Command:

```text
CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/evaluate_pose_sequence_spotter.py \
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
  --audio-contact \
  --audio-contact-epochs 8 \
  --audio-contact-hidden 96 \
  --audio-contact-blend-alphas 0.0,0.25,0.5,1.0 \
  --threads 16 \
  --quiet \
  --write-best-rows data/processed/validation_rows/seq_audio_contact_bridge_cap400_medium_oof.csv
```

Audio labels:

```text
n=5200
pos=2302
pos_rate=0.4427
offset_mae=2.920
features=(5200, 1, 137)
```

Best result:

```text
score=0.379792
pose_prior=0.2
audio_alpha=0.25
threshold=0.7
nms=10
cross_nms=2
snap_window=4
count_mode=root_count
count_multiplier=0.88
n=1279
```

Matched no-audio control in the same grid:

```text
audio_alpha=0.0
score=0.376648
```

So audio helps this weaker sequence setup by `+0.003144`, but absolute score is
still below:

```text
current best audio-gate identity source: 0.404915
fixed-row audio gate:                  0.404479
previous RGB bridge medium:            0.388112
```

Saved OOF rows:

```text
data/processed/validation_rows/seq_audio_contact_bridge_cap400_medium_oof.csv
```

Scorer verification:

```text
macro_score=0.379792
```

## Decision

Keep the audio-contact bridge infrastructure, but do not promote this cap-400
run. Audio has real candidate-level signal inside a weak sequence setting, but
it does not beat existing fixed-row audio/pose gates or RGB bridge rows.
