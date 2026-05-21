# RGB Candidate Contact Smoke - 2026-05-21

Goal: move RGB/contact earlier than the fixed-row postprocess stage. The
previous fixed-row frozen ViT-B clip branch regressed because it could only
drop or shift already selected rows. This smoke reranks a wider pose candidate
pool before final NMS/count selection.

No Kaggle upload.

## Tool

Added:

```text
tools/evaluate_rgb_candidate_contact.py
```

The tool builds a pre-selection `PunchCandidate` pool with `score_pose_tracks`
and light `fighter_hand` NMS, extracts temporal RGB clip embeddings for each
candidate, trains the existing temporal contact head in leave-one-fight OOF
folds, then reranks candidates before `build_rows`/`select_candidates`.

It reuses the optimized sequential video decode from
`tools/evaluate_rgb_contact_clip.py`, and all GPU work is run with
`CUDA_VISIBLE_DEVICES=1`.

## Smoke Config

Command shape:

```text
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/evaluate_rgb_candidate_contact.py \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --feature-cache data/processed/rgb_features/candidate_contact_vitb16_union_clip4_stride2_pool250.npz \
  --max-candidates-per-video 250 \
  --clip-len 4 \
  --frame-stride 2 \
  --crop-modes union \
  --decode-mode sequential \
  --epochs 12 \
  --head-batch-size 256 \
  --hidden 128 \
  --thresholds 0.05,0.10,0.20,0.40 \
  --nms-frames 8,10 \
  --cross-nms-frames 2 \
  --count-modes root_count \
  --count-multipliers 0.88 \
  --max-shifts 0,2 \
  --shift-scales 0.0,0.25
```

Feature cache:

```text
data/processed/rgb_features/candidate_contact_vitb16_union_clip4_stride2_pool250.npz
shape=(3250, 4, 768)
```

Candidate pool:

```text
ready=13 validation videos
pool=250 candidates/video
n=3250
pos=1771
pos_rate=0.5449
target_offset_mae=2.429 frames
```

## Result

Same-pool raw pose baseline:

```text
baseline_candidate_pool score=0.330787
fighter=0.543885
time=0.560897
fp=0.156681
n=1475
```

Best RGB rerank result:

```text
score=0.359641
delta_vs_same_pool=+0.028854
fighter=0.522353
time=0.542819
fp=0.110506
wins=9
n_rows=1352
score_mode=blend
blend_alpha=1.0
threshold=0.4
nms=10
cross_nms=2
count_mode=root_count
count_multiplier=0.88
max_shift=0
```

The RGB head separates positives and negatives enough to improve a raw-pose
candidate pool. Several video-level diagnostics show higher `pos_mean_p` than
`neg_mean_p`, and the final FP penalty falls sharply versus the same-pool
baseline.

However, absolute OOF is still far below the current best local source:

```text
current best fixed-row source = 0.401483
candidate RGB smoke best      = 0.359641
gap                           = -0.041842
```

## Decision

Keep candidate-level RGB/contact as a live signal, but not as a direct raw-pose
row source. The useful next branch is to inject this RGB score into the
sequence-TCN candidate scores before final NMS/count selection, not to spend
more cycles on raw-pose pool thresholds.

No test CSV was generated and this is not a submit candidate.
