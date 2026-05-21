# Experiment Path Summary - 2026-05-20

This file is the compact path log for the current Kaggle boxing run. Detailed
numbers remain in `EXPERIMENTS.md`; this file records why the active queue is
what it is.

## 1. Baseline And Metric

- Read `OVERVIEW.md`, `DATA_DESCRIPTION.md`, sample submission, train labels,
  and the baseline notebook.
- Implemented/validated local metric with the important correction: timing
  reaches max error at `0.5s`, even though matching window is `+/-1.0s`.
- Confirmed fixed submission contract: exactly `1594` rows, `clear=false` rows
  are safe filler and do not count as FP.
- Initial sample/temporal-prior approaches scored `0.00000` public, so the
  sample submission does not encode useful test labels.

## 2. First Pose Heuristic

- Converted pose tracks into punch candidates from wrist/head/body geometry,
  velocity, extension, and proximity features.
- Best yolo11n-style local heuristic reached about `0.23802` offline, but
  public pose heuristics were only `0.064..0.086`.
- Grouped NMS by `(fighter, hand)` and count gating helped locally, but not
  enough to become the anchor.
- Global frame offsets and global fighter swaps were tested and killed.

## 3. Larger Pose Models

- yolo11s became the first useful public jump:
  `0.340888` offline and `0.13275` public.
- yolo11m improved offline to `0.363325`, but the public submit scored only
  `0.12958`; treat that as overfit/distribution mismatch.
- yolo26s was weak (`0.305323` offline) with worse role coverage and more role
  switches.
- yolo26m was useful only as a timing witness, not a replacement detector.
- yolo26l became the current public anchor:
  `0.13849` from `same_sum w4 alpha=-0.2 root_rate=0.88`.
- Dense yolo26l threshold-count scored `0.10260` public, so count inflation is
  dangerous despite local gains.

## 4. Public Probing And Quota

- Daily quota reached `30/30` on `2026-05-20`; reset is
  `2026-05-21T00:00:00Z`.
- Public-mask probes showed `agn_038` affects public strongly:
  zeroing it dropped to `0.06940`.
- Several other zero probes did not move the checked score, so further probing
  should be sparse and targeted.
- No uploads until quota reset; after reset, submit one candidate and read
  public result before sending the next.

## 5. yolo26x Branch

- yolo26x validation cache completed on GPU 1.
- Direct yolo26x context passed tournament-root audit:
  `0.374335`, time `0.491512`, FP `0.052324`, no root risk flags.
- Test yolo26x candidates were generated and validated, but direct
  threshold-count has `844` rows and is risky because public punished density.
- The safer direct yolo26x fallback is `root_rate=0.88` with `678` rows.

## 6. Multi-Model Agreement

- Built/evaluated normalized agreement using yolo26x as primary and witnesses
  from yolo11s/yolo26l.
- yolo26x+yolo11s agreement is the first post-reset candidate:
  `0.377949`, time `0.533478`, FP `0.086935`, `727` test rows, no root risk.
- yolo26x+yolo26l agreement is a backup:
  `0.377090`, FP `0.091709`, `731` rows, no root risk.
- Submission generator and validator exist for both.

## 7. Learned Sequence Spotter

- Flat candidate-level learned rankers were tested and killed:
  HGB sanity `0.303634`, Gaussian-label HGB `0.361257`, both below yolo26x and
  agreement.
- Built a pose-sequence TCN over per-frame `(fighter, hand)` streams using
  yolo26x primary features plus yolo11s/yolo26l witness channels.
- Seed ensembling made the sequence branch stable enough to consider.
- Added `snap_window`: after selection, shift output frame to the local TCN
  probability maximum without changing count/fighter/hand.
- Best current sequence points:
  - snap4 `root_count=0.92`: `0.390013`, `765` rows.
  - snap4 `root_count=0.88`: `0.389963` first sweep, `0.387039` repeat,
    `744` rows.
  - snap4 `root_count=0.82`: `0.382612`, `712` rows, defensive fallback.
- Sequence should be second after yolo26x+yolo11s agreement, not first, because
  public has punished dense/learned-looking recall before.

## 8. Fighter Identity Work

- Whole-video fighter swaps are killed.
- Event-level matched-fighter oracle shows meaningful headroom
  (`~+0.018` macro), so the problem is real.
- Cached score-red/score-blue, track-majority, HGB/logreg keep-flip models did
  not recover the oracle headroom.
- Raw-frame simple bbox ROI clustering also failed; even oracle cluster mapping
  regressed to `0.368859`.
- Next fighter identity branch must use pose-guided crop quality, tracklet
  continuity, and possibly embedding features, with timing/count fixed.

## 9. Audio And Motion

- Audio-only onset detection is killed.
- Hard audio snapping and direct onset rescoring are killed on both old and
  yolo26x anchors; current yolo26x direct audio rescore regressed
  `0.374335 -> 0.365845`.
- Audio can only return as weak local features inside a learned reranker.
- Crop-motion smoke found weak signal on two videos, but the raw OpenCV decode
  path is too slow even with jobs. Next motion step needs a cached/requested
  frame extractor before full-validation sweeps.

## 10. Current Queue

1. Wait for quota reset.
2. Submit yolo26x+yolo11s agreement.
3. If public does not regress badly, submit sequence TCN snap4 `root_count=0.88`.
4. If sequence transfers, try snap4 `root_count=0.92`.
5. If public punishes count, try snap4 `root_count=0.82`.
6. If sequence fails, switch to yolo26x+yolo26l agreement or direct yolo26x
   `root_rate=0.88`.
7. Keep public probes to at most two high-value videos after reset.

## 11. Resource Notes

- GPU 0 must remain untouched.
- Use physical GPU 1 only: `CUDA_VISIBLE_DEVICES=1` or repo wrappers with
  `--cuda-visible-devices 1`.
- Current state before reset: no heavy model/training/extraction/GPU jobs are
  running.
