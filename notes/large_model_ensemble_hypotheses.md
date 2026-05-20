# Large Model Ensemble Hypotheses

Context: current public anchor is `yolo11s-pose` fighter+hand grouped NMS
(`0.13275` public, `0.340888` validation). `yolo11m-pose` is the warning case:
it improved validation to `0.363325` but scored only `0.12958` public. `yolo26s`
validation is reported around `0.305323` and has worse role coverage / more
track switches than YOLO11, so larger models must prove transfer safety, not
just higher local macro. No GPU run and no Kaggle submit from this note.

## Hypotheses

1. Large models are best used as secondary timing witnesses first.
   Start from the public-proven `yolo11s` selected events, then use `yolo11m`,
   completed `yolo26m`, and future `yolo26l/x` candidates only to snap frames
   within `+/-4` or `+/-8` when the same `(fighter, hand)` agrees. This tests
   punch timing without changing count or FP pressure.

2. Agreement boost is safer than a large-model replacement.
   Pool candidates from `yolo11s` plus one large model, normalize scores per
   model/video, and boost same `(fighter, hand)` candidates that agree within
   `2/4/6` frames. Keep `yolo11s` as primary unless a large model has track
   quality at least comparable to `yolo11s` and wins across tournament roots.

3. Disagreement should be asymmetric.
   A `yolo11s`-only candidate can survive on its existing score. A large-model
   only candidate should need high normalized score, stable role track history,
   and count budget. This avoids repeating the `yolo11m` public failure where
   stronger validation recall did not transfer.

4. TTA should target stability, not raw recall.
   If future extraction is justified, use one low-cost TTA/control at a time:
   horizontal flip with fighter-role remap, slightly larger image size, or one
   lower detector confidence. Promote only if the same punch peak appears under
   both views and role IDs remain stable. Do not combine TTA, confidence,
   tracker, and model-size changes in one run.

5. Track fusion can fix fighter identity better than frame-level voting.
   Fuse tracks by temporal IoU / bbox proximity / keypoint similarity, then
   vote fighter identity over a short segment before scoring punch candidates.
   Large models may provide cleaner wrists, while `yolo11s` may provide safer
   role continuity. The useful signal is fewer red/blue switches and fewer
   wrong-fighter near-misses, not just more detections.

6. Score calibration must be model- and video-aware.
   Raw score thresholds are not comparable (`yolo11s` best near `1.15`,
   `yolo11m` near `0.9`, `yolo26s` lower density). Use per-video p95,
   percentile rank, or robust z-score with a raw-score floor before any union.
   Check selected punches per minute by root; reject calibration that overfills
   weak videos or sample-capacity-heavy test templates.

7. Attributes stay anchored unless agreement is strong.
   Timing and fighter identity dominate expected gain. Keep `fighter`, `hand`,
   and target from the primary `yolo11s` event unless a large model agrees on
   timing and has better local role continuity. Do not submit attribute-only
   changes.

## Local Checks

- Complete-cache gate for any `val_yolo26m/l/x_conf*`: 13 JSONL files, nonzero,
  line counts equal to `data/raw/train/videos.csv` `frame_count`.
- Track-quality table: both-role coverage, red/blue presence, red/blue
  `track_id` switches per 1k frames, compared to known `yolo11s`, `yolo11m`,
  and `yolo26s` baselines.
- Score-scale table: raw candidate count, no-threshold `fighter_hand` NMS
  count, p90/p95/p99 score, counts above calibrated thresholds.
- Count-fixed timing snap: start from exact `yolo11s` validation selected set;
  compare keep-frame, large-model frame, and weighted midpoint. Report macro,
  time score, FP penalty, and per-video wins.
- Agreement grid on cached tracks only: `yolo11s` primary plus one secondary,
  windows `2/4/6`, asymmetric source weights, `fighter_hand` NMS `10/4`, and
  `root_count` or fixed-`yolo11s` counts.
- Root audit: report deltas on all validation, tournament-only validation,
  `tournament_1`, and `tournament_2`. A gain concentrated in `training|boks`
  is not public-safe.
- Fighter-identity audit: for near-GT candidates, count wrong-fighter cases
  before/after fusion or snapping. Require identity gain or no regression.

## Submission Gate

No Kaggle submission or test extraction unless one documented candidate passes:

- Exact validation model/cache is complete and local CSV validation will pass.
- Macro is at least `0.350888` (`0.340888 + 0.010`) or a manual-review case has
  clear timing gain with fixed counts and no FP/count inflation.
- FP penalty does not regress by more than `0.01` versus the matching `yolo11s`
  baseline.
- At least 10 of 13 validation-video wins versus the `yolo11s` anchor.
- Non-negative delta on both tournament roots, not only on training videos.
- Clear-true count is plausible relative to the successful `yolo11s` test
  candidate (`730`) and does not look like a threshold-only count increase.
- Candidate is materially new: large-model agreement, timing snap, TTA
  stability, track fusion, or calibration. Do not spend a submit on another
  threshold/NMS/count tweak around a failed large-model path.
