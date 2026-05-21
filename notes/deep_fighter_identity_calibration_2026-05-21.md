# Deep Fighter Identity Calibration - 2026-05-21

Goal: test whether per-video deep crop embeddings can improve fighter identity
by remapping track clusters to `red` / `blue`. This is an independent identity
branch and does not change timing/count selection.

No Kaggle upload.

## Run

```text
CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/evaluate_deep_fighter_identity_calibration.py \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --threshold 0.85 \
  --nms-frames 10 \
  --cross-nms-frames 4 \
  --count-mode root_rate \
  --count-multiplier 0.88 \
  --context-feature same_sum \
  --context-window 4 \
  --context-alpha -0.2 \
  --sample-stride 48 \
  --candidate-neighbor-frames 0,4,8 \
  --min-track-obs 3 \
  --purity-threshold 0.70 \
  --model-name resnet50.a1_in1k \
  --image-size 224 \
  --batch-size 128 \
  --device auto \
  --quiet
```

## Per-Video Purity

Several videos have high embedding-cluster purity, but many of the difficult
ones fall back:

```text
agn_003 purity=0.523 fallback=1
agn_004 purity=0.995 fallback=0
agn_010 purity=0.645 fallback=1
agn_023 purity=0.808 fallback=0
agn_024 purity=0.755 fallback=0
agn_025 purity=0.699 fallback=1
agn_056 purity=0.534 fallback=1
agn_057 purity=0.577 fallback=1
agn_058 purity=0.518 fallback=1
agn_069 purity=0.725 fallback=0
agn_070 purity=0.512 fallback=1
agn_071 purity=0.766 fallback=0
agn_072 purity=0.584 fallback=1
```

## Result

```text
variant,score,delta,fighter,time,fp,wins,n_pred,n_changed
baseline,0.361655,0.000000,0.463034,0.481776,0.056347,0,1139,0
role_map_gated,0.356873,-0.004782,0.440310,0.481542,0.056347,0,1139,84
role_map_all,0.343091,-0.018565,0.372183,0.481636,0.056347,0,1139,346
oracle_cluster_map,0.342864,-0.018791,0.370689,0.481835,0.056347,0,1139,331
```

Even the validation oracle cluster mapping regresses, so the current ResNet50
track-cluster embeddings do not align with the fighter labels for matched punch
events. This is worse than simple no-op.

## Decision

Kill this implementation as a direct fighter-label correction path. Future
identity work should use fixed-row matched-event supervised models or richer
tracklet/pose continuity, not unsupervised two-cluster crop remapping.
