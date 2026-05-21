# Fixed-Row Audio Gate - 2026-05-21

Goal: test audio as a weak auxiliary feature around already selected pose/video
events. This does not create audio-only rows and does not shift punch frames.

No Kaggle upload.

## Tools

- `tools/evaluate_fixed_row_audio_gate.py`
  - Builds per-row audio features from ffmpeg-decoded mono 16 kHz audio:
    onset, RMS, high-band RMS, spectral flux, energy delta, local peaks, and
    nearest-peak features.
  - Compares `audio`, `pose`, and `pose_audio` fixed-row keep/drop gates with
    OOF fight-group splits.
- `tools/make_fixed_row_audio_gate_submission.py`
  - Trains the fixed-row audio gate on validation rows and applies it to an
    existing test submission by changing only `clear=true` rows to `clear=false`.

## Validation Source

Input:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_gate_oof.csv
```

Baseline:

```text
score=0.401483
fighter=0.552581
time=0.545894
fp=0.084963
n=1275
```

Oracle unmatched-FP drop headroom:

```text
score=0.486269
n=1141
```

## OOF Results

Same fixed rows, same HGB keep/drop learner:

| Feature mode | Best score | Delta | Wins | Kept | Dropped | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `audio` | `0.401483` | `+0.000000` | `0` | `1275` | `0` | Audio alone is dead as a row gate. |
| `pose` | `0.401999` | `+0.000516` | `4` | `1260` | `15` | Pose/context-only control. |
| `pose_audio` | `0.404479` | `+0.002996` | `8` | `1226` | `49` | First useful additive audio signal. |

The `pose_audio` HGB result is not a single-threshold accident:

```text
threshold=0.42 score=0.403742
threshold=0.45 score=0.403907
threshold=0.48 score=0.403827
threshold=0.50 score=0.404479
threshold=0.52 score=0.403063
```

Logistic regression is poor and over-drops rows:

```text
best logreg score=0.392961
```

Saved OOF rows:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_gate_oof.csv
```

Scorer verification:

```text
python3 tools/score_predictions.py \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_gate_oof.csv \
  --video-keys-from-predictions

macro_score=0.404479
```

## Test Artifact

Applied to the full current local source:

```text
input=submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_gate_OFFLINE_CANDIDATE.csv
output=submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_audio_gate_OFFLINE_CANDIDATE.csv
```

Result:

```text
clear_rows=743
dropped=29
threshold=0.5
output_clear_rows=714
Validation passed.
```

Drops by test video:

```text
agn_037 52 -> 47
agn_038 97 -> 93
agn_039 55 -> 50
agn_047 121 -> 116
agn_048 44 -> 41
agn_049 59 -> 58
agn_062 120 -> 117
agn_063 121 -> 119
agn_064 74 -> 73
```

## Decision

Keep as a useful independent audio/pose gate and future ensemble source. Do not
upload automatically: the test artifact inherits full sequence-source public
transfer risk, and `agn_038` remains public-sensitive. This is stronger than the
previous audio-only and hard-snap failures, but not yet a public-safe submit.
