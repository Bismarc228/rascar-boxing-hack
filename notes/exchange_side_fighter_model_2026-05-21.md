# Exchange-Side Fighter Model - 2026-05-21

Goal: test whether a learned high-precision opposite-fighter flip model can
recover more of the fighter-identity oracle headroom than the narrow hand-coded
rival rule, without using color/HSV/deep-crop identity.

No Kaggle upload.

## Tool

Added:

```text
tools/evaluate_exchange_side_fighter_model.py
```

The tool keeps frame/count/clear fixed and tests only local row updates:

```text
keep vs flip fighter
```

It builds OOF predictions by fight group using features from the selected row,
nearby self and opposite-fighter pose candidates, score ratios, candidate
feature deltas, and local density/window stats. Labels come from metric-style
matched rows: positive means the fixed row is time-scorable but has the wrong
fighter.

## Input

Current best local source:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_gate_oof.csv
baseline score=0.401483
fighter=0.552581
time=0.545894
fp=0.084963
n=1275
```

Pose tracks:

```text
data/processed/pose_tracks/val_yolo26x_conf035
```

## Smoke Command

```text
.venv/bin/python tools/evaluate_exchange_side_fighter_model.py \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_gate_oof.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --windows 0,2 \
  --match-modes same_hand_target,same_hand \
  --update-modes fighter_only \
  --models hgb,logreg \
  --thresholds 0.70,0.80,0.90,0.95,0.98 \
  --top-k 30
```

Best result:

```text
score=0.401720
delta=+0.000236
fighter=0.553299
time=0.545776
fp=0.084963
wins=4
n_changed=21
model=logreg
window=2
match_mode=same_hand
update_mode=fighter_only
threshold=0.90
```

Best HGB result:

```text
score=0.401660
delta=+0.000177
n_changed=1
model=hgb
window=2
match_mode=same_hand_target
threshold=0.80
```

## Decision

This is a real but tiny identity signal. It slightly improves over the previous
manual rival micro (`0.401575`), but it is nowhere near the promotion gate
(`>=0.4045`) and does not meaningfully close the matched-fighter oracle
headroom (`0.416256`).

Keep the tool for diagnostics and possible future stacking, but do not generate
a test CSV or upload. Broadening the flip rule hurts quickly, so this branch
should not be threshold-swept further without a new feature source.
