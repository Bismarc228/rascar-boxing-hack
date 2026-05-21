# Audio-Gate Exchange-Side Identity - 2026-05-21

Goal: retest the high-precision opposite-fighter flip model after the fixed-row
audio gate changed the row set. Timing, counts, and `clear` stay fixed.

No Kaggle upload.

## Input

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

Matched-fighter oracle:

```text
score=0.419402
fighter=0.616734
matched_scorable=1060
fighter_wrong=128
wrong_rate=0.1208
```

Simple color/role flip rules still fail; most are no-op or regress.

Diagnostic written by `tools/analyze_fighter_identity_errors.py`:

```text
data/processed/diagnostics/fighter_identity_errors_seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_gate_oof.csv
```

## Learned Exchange-Side Model

Best OOF result:

```text
model=logreg
window=2
match_mode=same_hand
update_mode=fighter_only
threshold=0.9
changed=22

score=0.404915
delta=+0.000436
fighter=0.543241
time=0.537088
fp=0.073644
wins=4
```

Saved OOF rows:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_gate_exchange_side_oof.csv
```

Scorer verification:

```text
macro_score=0.404915
```

## Source Oracle

Adding this micro-source to the audio/source pool:

```text
audio_gate=0.404479
audio_exchange=0.404915
source_oracle=0.422744
```

The previous audio-gate source oracle was `0.422619`, so the net oracle lift is
only `+0.000125`.

`audio_exchange` wins:

```text
agn_025, agn_057, agn_072
```

## Decision

Keep as an identity micro-signal for future ensemble experiments. Do not create
a Kaggle upload from this branch alone: the lift is only `+0.000436` over the
audio gate and does not meet the promotion bar.

## Test Artifacts

Added:

```text
tools/make_exchange_side_fighter_model_submission.py
```

Full audio-gate source:

```text
input=submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_audio_gate_OFFLINE_CANDIDATE.csv
output=submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_audio_exchange_side_OFFLINE_CANDIDATE.csv
changed=11
Validation passed.
```

Conservative root-splice source with `agn_038` protected:

```text
input=submissions/hybrid_root_seqrepeat_exchange_attr_motion_audio_gate_noagn038_OFFLINE_CANDIDATE.csv
output=submissions/hybrid_root_seqrepeat_exchange_attr_motion_audio_exchange_side_noagn038_OFFLINE_CANDIDATE.csv
protected=agn_038
changed=11
Validation passed.
```

Changed videos for the protected artifact:

```text
agn_037:1
agn_047:3
agn_048:1
agn_049:2
agn_062:2
agn_063:2
```

These artifacts are saved for ensemble/private-risk bookkeeping only.
