# Fighter Identity Rival Micro On Current Best - 2026-05-21

Goal: retest the opposite-fighter pose rival idea on the current best local
fixed-row source, without repeating the killed global HSV/color/deep-crop
identity calibrations and without uploading to Kaggle.

## Inputs

Current best OOF source:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_gate_oof.csv
```

Pose tracks:

```text
data/processed/pose_tracks/val_yolo26x_conf035
data/processed/pose_tracks/test_yolo26x_conf035
```

## Fixed-Row Identity Audit

Command:

```text
python3 tools/analyze_fighter_identity_errors.py \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_gate_oof.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --output data/processed/diagnostics/fighter_identity_errors_seq_repeat_exchange_attr_motion_gate.csv \
  --top-k 30
```

Result:

```text
baseline: score=0.401483,fighter=0.552581,time=0.545894,fp=0.084963,n=1275
oracle_matched_fighter: score=0.416256,fighter=0.627044,time=0.545801,fp=0.084963
matched_scorable=1080 fighter_wrong=127 wrong_rate=0.1176
```

Simple role/color rules still do not improve the source. Best rule scores were
flat only when changing no rows; all non-empty global color/role flips regressed.

## Rival Sweep

Command:

```text
python3 tools/evaluate_fighter_rival_flip.py \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_gate_oof.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --top-k 40
```

Best local rule:

```text
score=0.401575
delta=+0.000092
fighter=0.553224
time=0.545894
fp=0.084963
wins=2
n_changed=20
window=2
match_mode=same_hand_target
ratio=2.0
min_rival_score=0.5
update_mode=fighter_only
```

The raw rival diagnostic confirms why the rule must stay narrow:

```text
window_0 would_fix=84 would_break=478
window_2 would_fix=124 would_break=749
window_4 would_fix=141 would_break=860
window_8 would_fix=151 would_break=942
```

## Saved Artifacts

OOF micro-source:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_rival_micro_oof.csv
macro_score=0.401575
```

Full sequence test artifact:

```text
submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_rival_micro_OFFLINE_CANDIDATE.csv
changed=16 clear_rows=743 total_rows=1594
Validation passed.
```

Conservative no-`agn_038` root splice artifact:

```text
submissions/hybrid_root_seqrepeat_exchange_attr_motion_rival_micro_noagn038_OFFLINE_CANDIDATE.csv
changed=13 clear_rows=733 total_rows=1594
Validation passed.
```

## Decision

Keep this as an independent fighter-identity micro-signal for a later ensemble
or source arbitration table. Do not upload it directly: the local gain is only
`+0.000092`, and the rule still inherits the sequence public-transfer risk.
