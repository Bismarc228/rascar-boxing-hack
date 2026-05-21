# Tracklet Appearance Fighter Identity - 2026-05-21

Goal: test a fixed-row fighter-identity correction that uses per-video
tracklet appearance and equipment/color descriptors without changing timing,
count, hand, target, or attributes.

No Kaggle upload.

## Tool

Added:

```text
tools/evaluate_tracklet_appearance_fighter_model.py
```

The evaluator builds leave-one-fight OOF flip probabilities from:

- pose track ids and role purity;
- per-frame and per-track red/blue role score margins;
- bbox position/size features;
- per-video crop descriptors from upper/lower body HSV/LAB/chroma/color
  fractions;
- per-video role prototypes from high-purity tracklets.

It applies only `fighter` flips. The video crop extraction uses `decord`
sparse batch reads; the first OpenCV random-seek version was killed as too
slow.

## Audio-Gate Source

Input:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_gate_oof.csv
```

Baseline:

```text
score=0.404479
fighter=0.541524
time=0.537206
fp=0.073644
n=1226
```

Best OOF:

```text
model=hgb
threshold=0.6
changed=2

score=0.404679
delta=+0.000199
fighter=0.542521
time=0.537206
fp=0.073644
wins=1
```

Changed rows:

```text
agn_003 frame=2421 blue->red
agn_056 frame=1672 blue->red
```

OOF rows:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_tracklet_appearance_oof.csv
```

Scorer verification:

```text
macro_score=0.404679
```

## Current Best Source

Input:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_gate_exchange_side_oof.csv
```

Baseline:

```text
score=0.404915
fighter=0.543241
time=0.537088
fp=0.073644
n=1226
```

Best OOF:

```text
model=hgb
threshold=0.6
changed=1

score=0.405200
delta=+0.000285
fighter=0.544665
time=0.537088
fp=0.073644
wins=1
```

Changed row:

```text
agn_003 frame=2421 blue->red right/head
```

OOF rows:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_appearance_oof.csv
```

Scorer verification:

```text
macro_score=0.405200
```

## Decision

Save this as a separate fighter-identity micro-signal for future ensemble work.
It is now the best local OOF row source, but the gain is only `+0.000285` over
the previous best and comes from one validation flip, so it is not a standalone
Kaggle upload trigger. No test CSV was generated from this branch.

## Source-Oracle Check

Within the audio-family source pool:

```text
audio_gate=0.404479
audio_exchange=0.404915
audio_tracklet=0.405200
source_oracle=0.405380
source_oracle_choices=agn_003:audio_tracklet,agn_004/025/057/072:audio_exchange
```

Against the broader saved source pool, `audio_tracklet` is the best pairwise
audio-family source versus `seq_motion`:

```text
pairwise audio_tracklet=0.409500
pairwise audio_exchange=0.409215
pairwise audio_gate=0.409162
```

However, the full source oracle still prefers `old_attr` on `agn_003`, where
the tracklet branch is uniquely helpful inside the audio-family pool. That means
this source is worth keeping as a micro-signal, but it does not unlock the
larger source-selection problem.
