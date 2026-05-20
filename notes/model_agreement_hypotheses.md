# Cross-Model Agreement Hypotheses

Context: current public best is `0.13275` from the `yolo11s-pose`
fighter+hand grouped-NMS candidate. The best local validation result is
`0.363325` from `yolo11m-pose` same-score context, but the matching public
submission scored only `0.12958`. Treat `yolo11s` as the public-transfer anchor
and `yolo11m` as a high-recall, higher-risk secondary signal until an agreement
audit explains the local/public gap.

Use cached tracks only. Do not run GPU jobs and do not make new Kaggle
submissions from this note.

## Local Evidence

- Public baseline:
  `submissions/yolo11s_fighterhand_thr115_same10_cross4_rootcount1_OFFLINE_CANDIDATE.csv`
  scored `0.13275`.
- Failed public challenger:
  `submissions/yolo11m_context_samesum_w8_am02_thr09_same10_cross4_OFFLINE_CANDIDATE.csv`
  scored `0.12958`, despite `0.363325` local validation.
- Best checked `yolo11s` validation config:
  `0.340888`, fighter+hand grouped NMS, `threshold=1.15`,
  same-group NMS `10`, cross-group NMS `4`, `root_count=1.0`, `n=1202`.
- Best checked `yolo11m` validation configs:
  `0.363216` raw grouped-NMS at `threshold=0.9`, same-group NMS `8`,
  cross-group NMS `4`, `root_count=1.0`, `n=1385`; and `0.363325`
  same-score context with window `8`, `alpha=-0.2`, same-group NMS `10`,
  cross-group NMS `4`, `n=1198`.
- A post-public `yolo11m` count audit only moved `0.363325 -> 0.363567`, so
  another threshold/context/count tweak around `yolo11m` is not materially new.
- Validation caches exist for both models:
  `data/processed/pose_tracks/val_yolo11s_conf035/` and
  `data/processed/pose_tracks/val_yolo11m_conf035/`. Test caches also exist for
  both models, but this note is about validation-only agreement research.
- Validation split composition is not identical to public test behavior:
  3 `training|бокс` videos, 3 `tournament_1|Турнир Бокс` videos, and
  7 `tournament_2|Турнир Бокс 2` videos. Public test has only tournament roots.
- Sample capacity is highly uneven on test, especially `agn_062` with 746 rows
  and 378 sample-true rows. Count policies must not blindly inherit sample
  counts.
- Local helper `tools/evaluate_pose_model_agreement.py` already sketches a
  CPU-only agreement grid: per-model score normalization, cross-model agreement
  windows, source weights, grouped NMS, and count modes. It is currently an
  untracked local file, so treat it as a draft tool to review before relying on
  its outputs.

## Main Hypotheses

1. Intersection should improve precision but may lose too much recall.

   Select or boost only events where both models have a same `(fighter, hand)`
   candidate within `+/-2`, `+/-4`, or `+/-6` frames. Use `yolo11s` frame and
   attributes as the default anchor because it transferred to public. Try a
   secondary variant that snaps the frame to the higher normalized local peak
   when the two models agree within `+/-4`.

   Expected win: lower FP penalty and fewer public-risk `yolo11m` hallucinations.
   Expected failure mode: missed punches where only one model fires, especially
   fast exchanges where one detector loses a wrist.

2. Union should be allowed only under strong count clipping.

   Pool candidates from both models after normalization, then run
   `fighter_hand` grouped NMS with same-group windows `8..12` and cross-group
   windows `3..4`. Unbounded threshold-only union is likely to inflate false
   positives. Test union with `root_count` or `root_round_count` multipliers
   `0.8`, `0.9`, `1.0`, plus a fixed-count control that uses the exact per-video
   `yolo11s` selected counts.

   Expected win: `yolo11m` adds recall where `yolo11s` under-detects.
   Expected failure mode: macro gain comes from validation count overfit and
   disappears when counts are held fixed.

3. Agreement boost is safer than full replacement.

   Keep both sources in the pool, but score candidates as:
   `normalized_score * source_weight * (1 + alpha * log1p(other_model_score))`.
   Start with windows `2`, `4`, `6`; `alpha` in `0.2`, `0.5`, `1.0`; and
   asymmetric source weights:
   `yolo11s=1.0..1.2`, `yolo11m=0.5..0.9`.

   Expected win: consensus candidates rise above single-model noise while
   `yolo11m`-only events need unusually high normalized score to survive.
   Expected failure mode: if normalization is poor, `yolo11m` still dominates
   because its raw score scale and best threshold differ from `yolo11s`.

4. Consensus snapping can test timing gains without count changes.

   Start from the exact `yolo11s` selected set and do not add or remove rows.
   For each `yolo11s` event, find the best `yolo11m` same `(fighter, hand)`
   candidate within `+/-4` or `+/-8`. Compare three frame policies:
   keep `yolo11s`; snap to `yolo11m`; and use a normalized-score weighted
   midpoint rounded to frame. Attributes should stay from the `yolo11s` anchor
   unless `yolo11m` agrees on frame but strongly disagrees on target.

   Expected win: direct timing-score improvement with unchanged FP penalty.
   Expected failure mode: global offset sweeps already failed, so snapping must
   prove it fixes local peak placement rather than shifting noise.

5. Disagreement should be stratified, not globally rejected.

   A `yolo11m`-only candidate may be useful on hard validation fights but risky
   on public. Report model-only and shared candidates by `data_root`,
   `dataset_type`, `fight_folder`, count, time score, FP penalty, and video
   wins. Reject any setting whose lift is concentrated in `training|бокс`,
   because public test has only tournament roots.

   Expected win: identify whether `yolo11m` overfit is root-specific.
   Expected failure mode: 13 validation videos are small, so root splits can be
   noisy; require tournament-wide consistency, not just one fight.

6. Score normalization must be per model and probably per video.

   Raw thresholds are not comparable (`yolo11s` best around `1.15`, `yolo11m`
   around `0.9`). Normalize candidate scores before union/intersection. Start
   with per-video/model `score / p95(score)` from a lightly pre-NMSed pool, and
   compare robust z-score or percentile rank as controls. For cross-root audits,
   also report whether the same normalized threshold selects similar counts per
   minute on `tournament_1` and `tournament_2`.

   Expected win: source weights become interpretable and no model wins simply
   because of score scale.
   Expected failure mode: per-video normalization can over-promote bad videos
   where all candidates are weak; include raw-score floor controls.

7. Source weighting should default to `yolo11s`-first.

   Because `yolo11s` transferred and `yolo11m` did not, the first ensemble grid
   should use `yolo11s` as primary and make `yolo11m` earn inclusion through
   agreement. Reasonable first settings:
   `yolo11s_weight=1.0,1.2`; `yolo11m_weight=0.5,0.7,0.9`; agreement windows
   `4,6`; and `root_count` multipliers `0.9,1.0`.

   A reverse primary grid is useful only as a diagnostic: if `yolo11m` primary
   still wins offline but loses fixed-count/tournament gates, it confirms the
   public-risk pattern rather than justifying another submit.

## Gates

Use gates for validation research only; this note does not authorize new
submissions.

- Baselines: compare every variant against exact `yolo11s` local `0.340888` and
  against the failed `yolo11m` context `0.363325`.
- Fixed-count check: macro/time gain must remain when per-video counts are held
  to the `yolo11s` selected counts. Otherwise the result is mostly count tuning.
- Precision check: FP penalty must not regress by more than `0.01` versus
  `yolo11s`; for intersection variants it should improve.
- Video wins: require at least 10/13 wins versus `yolo11s`, and no collapse on
  the tournament-only subset.
- Root gate: require non-negative macro delta separately on
  `tournament_1|Турнир Бокс` and `tournament_2|Турнир Бокс 2`. Do not accept a
  result driven primarily by `training|бокс`.
- Count plausibility: report `n_pred` and per-video counts. Reject variants that
  rely on sample-capacity oddities, especially behavior that would overfill
  `agn_062` on test.
- Materiality: a candidate must be a true agreement/ensemble change. Another
  `yolo11m` threshold, context-alpha, or root-count tweak is already ruled out
  by the failed public check and post-public count audit.
- Minimum discussion bar: prefer variants with `>= +0.010` macro over `yolo11s`
  and a timing-score lift, but a count-fixed snapping result can be discussed
  with a smaller macro gain if FP penalty is unchanged.

## Light CPU Experiment Plan

1. Agreement diagnostics only: build per-video tables of
   `shared`, `yolo11s_only`, and `yolo11m_only` candidates after light pool NMS,
   grouped by root/fight. Include counts, normalized score percentiles, and
   distance to nearest GT on validation.
2. Count-fixed snapping: start from the exact `yolo11s` selected rows and test
   only frame replacement/midpoint policies using nearby `yolo11m` candidates.
   This isolates timing from count and FP changes.
3. Narrow intersection grid: same `(fighter, hand)` agreement windows `2,4,6`,
   `yolo11s` anchor, `root_count=1.0`, same/cross NMS `10/4`, and one
   threshold sweep on normalized agreement score.
4. Narrow union grid: per-video p95 normalization, `yolo11s_weight=1.2`,
   `yolo11m_weight=0.5,0.7`, `alpha=0.2,0.5`, windows `4,6`, same/cross NMS
   `10/4`, count modes `fixed_yolo11s_count`, `root_count=0.9`, `root_count=1.0`.
5. Root audit: repeat score summaries on all validation videos, tournament-only
   validation videos, and by individual `data_root`. Promote no setting that
   fails the root gate.

