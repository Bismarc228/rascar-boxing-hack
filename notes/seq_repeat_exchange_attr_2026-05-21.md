# Sequence Repeat + Exchange + Attributes - 2026-05-21

Goal: repeat the fixed `seq_tcn_snap4_rootcount088` configuration without a
postprocess sweep, then test whether the new sequence rows combine with the
positive exchange gate and fixed-row attribute model. No Kaggle upload.

## OOF Repeat

Command shape:

```text
tools/evaluate_pose_sequence_spotter.py
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035
  --witness-tracks-dirs data/processed/pose_tracks/val_yolo11s_conf035,data/processed/pose_tracks/val_yolo26l_conf035
  --seeds 41,42,43
  --epochs 8
  --chunks-per-epoch 1600
  --hidden 96
  --layers 6
  --thresholds 0.6
  --nms-frames 10
  --cross-nms-frames 2
  --snap-windows 4
  --count-modes root_count
  --count-multipliers 0.88
  --pose-priors 0.2
```

Result:

```text
seq_repeat=0.391357
time=0.550534
fp_penalty=0.092638
n_rows=1299
```

Rows:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_smoke.csv
```

This is better than the old `seq_tcn_snap4_rootcount088_oof.csv` (`0.379682`)
but still below the previous best fixed-row attribute source (`0.396329`).

## Postprocess Stack

Exchange keep/drop gate on repeat rows:

```text
seq_repeat_exchange=0.393593
threshold=0.28
n_rows=1282
dropped=17
```

Rows:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_hgb_p024_oof.csv
```

Fixed-row attributes on repeat rows:

```text
seq_repeat_attr_all=0.398004
```

Rows:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_attr_all_oof.csv
```

Fixed-row attributes after exchange gate:

```text
seq_repeat_exchange_attr_all=0.400288
time=0.546891
fp_penalty=0.087252
```

Rows:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_all_oof.csv
```

This is the best local OOF source so far, but the gain is still local-only and
the full-sequence public transfer risk remains high.

## Source Ensemble Headroom

With `seq_repeat_attr` and old attribute sources:

```text
source_oracle=0.417687
pairwise base+seq_repeat_attr=0.412286
pairwise base+seq_repeat_exchange=0.408317
pairwise base+seq_repeat=0.407292
```

Per-video oracle selected `seq_repeat_attr` on `agn_010`, `agn_024`,
`agn_025`, and `agn_057`; old attribute/effectiveness sources still won
several other videos.

After including `seq_repeat_exchange_attr` itself in the fight-level diagnostic
table, the source oracle reaches `0.418678`, but current learned/rule policies
still do not beat the best single source:

```text
single seq_repeat_exchange_attr=0.400288
ridge policy=0.394629
mean_root policy=0.389437
mean_global policy=0.385617
mean_round policy=0.383276
mean_root_round policy=0.382539
hgb policy=0.381433
```

The extra oracle headroom is real, but the current fight/round feature policies
do not recover it.

## Test Artifacts

Generated and validated:

```text
submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_OFFLINE_CANDIDATE.csv
submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_hgb_p028_OFFLINE_CANDIDATE.csv
submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_all_OFFLINE_CANDIDATE.csv
```

Final clear counts:

```text
agn_037:52
agn_038:99
agn_039:55
agn_047:123
agn_048:44
agn_049:60
agn_062:123
agn_063:122
agn_064:74
total=752
```

Compared with root `submission.csv`, the final CSV changes `767` rows. It is a
full sequence-style private-risk artifact, not a public-safe candidate.

Frame-consistency gating against the public anchor was also checked. On
validation, replacing only videos that pass the existing base/override proximity
gate scores below the full new source:

```text
base=0.365177
override seq_repeat_exchange_attr=0.400288
gated hybrid=0.398894
replace_keys=agn_004,agn_023,agn_024,agn_025,agn_056,agn_057,agn_058,agn_069,agn_070,agn_071
```

The analogous test artifact is valid:

```text
submissions/hybrid_root_seq_repeat_exchange_attr_gate_OFFLINE_CANDIDATE.csv
replace_keys=agn_038,agn_047,agn_062,agn_063
total_clear=757
```

However, because the validation gate underperforms the full new source and the
test replacement touches multiple public-risk videos, this remains diagnostic.

## Decision

- Keep as the current best local OOF branch and a strong ensemble source.
- Do not upload automatically: public evidence already punished full sequence
  replacement, and this artifact rewrites many public-anchor rows.
- If uploads resume, compare this only against a video-gated splice strategy,
  not as a blind full replacement.
