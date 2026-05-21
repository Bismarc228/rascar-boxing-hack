# Count Controller Audit - 2026-05-21

Goal: test whether a learned per-video/fight count controller can improve event
selection beyond simple root/round count heuristics.

No Kaggle upload and no test CSV generation.

## Tool

Existing evaluator:

```text
tools/evaluate_count_controller.py
```

It builds pose-candidate density/count features, predicts per-video clear counts
with fight-group OOF splits, then re-runs fixed candidate selection with those
predicted counts.

## Direct yolo26x Count

Command family:

```text
.venv/bin/python tools/evaluate_count_controller.py \
  --primary-tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --mode direct_count \
  --base-threshold 0.65 \
  --nms-frames 8 \
  --cross-nms-frames 4 \
  --context-feature same_count \
  --context-window 10 \
  --context-alpha -0.2 \
  --pool-min-score 0.0 \
  --pool-nms-frames 2 \
  --max-candidates-per-source-video 4000
```

Results:

```text
best_policy root_round_count=1.0  score=0.374359  n=1151
ridge residual_count              score=0.374335  n=1153
hgb residual_count                score=0.374335  n=1153
mean residual_count               score=0.373619  n=1144
```

The ridge/HGB controllers select nearly the same final row count as the default
threshold/root-count policies and do not beat the best static count policy.

## yolo26x/yolo26l Agreement Count

Command:

```text
.venv/bin/python tools/evaluate_count_controller.py \
  --primary-tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --secondary-tracks-dir data/processed/pose_tracks/val_yolo26l_conf035 \
  --mode agreement_count \
  --model ridge \
  --target residual_count \
  --agreement-window 4 \
  --agreement-alpha 0.2 \
  --primary-weight 1.0 \
  --secondary-weight 0.8
```

Result:

```text
best_policy root_rate=0.72  score=0.355101  fp=0.099506  n=1267
count_controller           score=0.339953  fp=0.171750  n=1637
oracle_count               score=0.348047  fp=0.163507  n=1647
```

The agreement pool is too dense for this controller; even oracle counts stay far
below the direct yolo26x count policies because the candidate pool itself raises
FP too much.

## Decision

Kill learned count-control as a current growth path. It does not explain the
sequence/public-transfer gap and does not recover the fight-level source oracle
headroom. Future fight-level work should focus on source selection or event
quality, not per-video count regression.
