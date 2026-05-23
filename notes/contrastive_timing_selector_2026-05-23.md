# Contrastive timing selector experiment - 2026-05-23

## Goal

Test whether an auxiliary contrastive loss can improve the pose sequence timing
selector by making punch-impact embeddings sharper than nearby non-punch or
wrong-stream frames.

Target baseline for the overall timing selector work:

| source | macro | time | fighter | fp |
| --- | ---: | ---: | ---: | ---: |
| current fixed OOF baseline | 0.415261 | 0.530111 | 0.537953 | 0.061715 |

Sequence branch control used for this experiment:

| config | macro | time | fighter | fp | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| no contrastive, count 0.95 | 0.393302 | 0.573637 | 0.570610 | 0.109203 | 1394 |

## Implementation

Branch: `codex/contrastive-timing-selector`

Files:

- `tools/evaluate_pose_sequence_spotter.py`
- `tools/make_pose_sequence_submission.py`

The sequence TCN now exposes its hidden frame embedding before the final
`1x1` output head. The default behavior is unchanged because all contrastive
flags default to no-op:

```text
--contrastive-weight 0.0
```

When enabled, training uses:

```text
loss = weighted_bce + contrastive_weight * local_temporal_infonce
```

Contrastive sampling:

- Anchors: frames where dense stream label `y >= 0.70`.
- Positives: same stream, local offsets `t-1` and `t+1` when `y >= 0.20`.
- Negatives: batch frames where stream label `y <= 0.05` and sample weight is
  non-zero.
- Hard negatives: 50% of the negative pool is selected from highest sample
  weights, which includes wrong-stream frames near real punches.
- Random negatives: remaining negative slots are sampled randomly.

Default contrastive parameters:

```text
temperature=0.10
positive_window=1
negatives=512
max_anchors=128
hard_negative_frac=0.5
```

## OOF command template

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/evaluate_pose_sequence_spotter.py \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --witness-tracks-dirs data/processed/pose_tracks/val_yolo11s_conf035,data/processed/pose_tracks/val_yolo26l_conf035 \
  --epochs 8 --chunks-per-epoch 1600 --seeds 41,42,43 --threads 16 \
  --thresholds 0.6 --nms-frames 10 --cross-nms-frames 2 \
  --cross-nms-diff-fighter-ratio-thresholds 0.0 \
  --same-group-nms-ratio-thresholds None \
  --snap-windows 4 --snap-features sequence \
  --count-modes root_count --count-multipliers 0.88,0.95 \
  --pose-priors 0.2 --top-k 5 --quiet \
  --contrastive-weight <weight>
```

## Results

| contrastive_weight | best_count_multiplier | macro | delta_vs_sequence_control | delta_vs_current_baseline | time | fighter | fp | n |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.005 | 0.88 | 0.390435 | -0.002867 | -0.024826 | 0.555059 | 0.552445 | 0.095327 | 1304 |
| 0.020 | 0.95 | 0.391790 | -0.001512 | -0.023471 | 0.570069 | 0.567861 | 0.107460 | 1377 |
| 0.050 | 0.95 | 0.387912 | -0.005390 | -0.027349 | 0.563188 | 0.566223 | 0.106039 | 1368 |

Smoke check:

- `--contrastive-weight 0.05 --epochs 1 --chunks-per-epoch 8` ran through all
  7 OOF folds without runtime errors.

## Single-seed epoch trace update

The first implementation used a Python loop over anchors and copied anchor
indices to CPU. That was replaced with a vectorized CUDA positive-window gather:

- no `.cpu().tolist()` in the contrastive loss,
- positives for arbitrary `+/-W` windows are collected with tensor indexing,
- the training protocol can now print per-epoch fold validation with
  `--trace-epochs`.

Trace command shape:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/evaluate_pose_sequence_spotter.py \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --witness-tracks-dirs data/processed/pose_tracks/val_yolo11s_conf035,data/processed/pose_tracks/val_yolo26l_conf035 \
  --epochs 8 --chunks-per-epoch 1600 --seeds 41 --threads 16 \
  --thresholds 0.6 --nms-frames 10 --cross-nms-frames 2 \
  --cross-nms-diff-fighter-ratio-thresholds 0.0 \
  --same-group-nms-ratio-thresholds None \
  --snap-windows 4 --snap-features sequence \
  --count-modes root_count --count-multipliers 0.95 \
  --pose-priors 0.2 --top-k 3 --quiet --trace-epochs \
  --contrastive-weight 0.02 --contrastive-positive-window <W>
```

Final epoch, single seed:

| config | positive_window | macro | time | fighter | fp | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no contrastive | - | 0.371902 | 0.571381 | 0.570072 | 0.129368 | 1445 |
| contrastive | 2 | 0.365401 | 0.565217 | 0.557179 | 0.128350 | 1438 |
| contrastive | 3 | 0.370003 | 0.566276 | 0.562827 | 0.125047 | 1420 |
| contrastive | 4 | 0.368694 | 0.569307 | 0.563438 | 0.129171 | 1434 |

Fold-wise best-epoch diagnostic, weighted by validation-video count:

| config | positive_window | best-epoch weighted macro |
| --- | ---: | ---: |
| no contrastive | - | 0.396265 |
| contrastive | 2 | 0.388888 |
| contrastive | 3 | 0.391955 |
| contrastive | 4 | 0.393341 |

The per-epoch traces show train loss falling almost monotonically while fold
validation often peaks around epochs 2-6. That points to early stopping as a
real training-protocol issue, but this contrastive loss is still worse than the
same one-seed no-contrastive control under both final-epoch and best-epoch
diagnostics.

## Select-best-epoch rerun

`tools/evaluate_pose_sequence_spotter.py` now has:

```text
--select-best-epoch
```

This is a diagnostic mode: for each OOF fold and seed it evaluates validation
after every epoch, stores the best checkpoint by fold validation macro, then
uses that checkpoint for OOF predictions. It is useful for detecting fixed-epoch
overfitting, but it is still optimistic until wrapped in a nested protocol.

Single-seed results:

| config | positive_window | macro | time | fighter | fp | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no contrastive | - | 0.391137 | 0.577999 | 0.573223 | 0.114634 | 1426 |
| contrastive | 4 | 0.388615 | 0.579456 | 0.580310 | 0.119464 | 1441 |

The no-contrastive sequence scorer improved from final-epoch `0.371902` to
best-epoch `0.391137`, so fixed final epoch was indeed hurting this one-seed
run. Contrastive still loses to the no-contrastive control.

Audio-contact bridge rerun with the same best-epoch sequence control:

| config | macro | time | fighter | fp | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| best no-audio row | 0.389410 | 0.545483 | 0.555543 | 0.090373 | 1296 |
| best audio alpha 0.25 row | 0.382468 | 0.544001 | 0.554543 | 0.096608 | 1305 |

Audio-contact did not survive this controlled rerun. The best row used
`audio_alpha=0.0`, so the old audio bridge lift was not robust under the
single-seed best-epoch protocol.

RGB-contact bridge could not be rerun from the old cache without regeneration:
the current pool has `22742` candidates, while
`seq_bridge_contact_vitb16_union_clip4_stride2_pool1800_ab.npz` contains
`22691` features. The cache is stale relative to the current candidate-pool
code.

## Conclusion

This first contrastive formulation is negative. It does not beat the
non-contrastive sequence control, and it is far below the current fixed OOF
baseline `0.415261`.

The likely issue is that dense-label local positives encourage temporal
smoothness around an already broad Gaussian label, while the selector needs
better ordering among very similar local pool candidates. A better second
version would use candidate/event IDs from the suppressor audit directly:

- anchor: GT-positive suppressed pool candidate,
- positive: the nearest labeled impact candidate for the same GT event,
- negative: the actual suppressor candidate and nearby wrong-fighter/wrong-hand
  high-score candidates.
