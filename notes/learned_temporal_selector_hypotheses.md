# Learned Temporal Selector / Localizer Hypotheses

Context: current best public is `0.13275` from `yolo11s-pose`
fighter+hand grouped NMS; best local validation is `0.340888`. The cached
`yolo11s` pool already covers GT very well (`+/-15 ~= 0.978`, `+/-8 ~= 0.953`,
`+/-4 ~= 0.918`), so the next learned step should select and localize events,
not replace pose detection. Use cached tracks and CPU-light experiments first;
do not run GPU work for this note.

## Repo Evidence

- Metric weight is timing-heavy: time `50%`, fighter `20%`, punch type `10%`,
  effectiveness `8%`, hand/target `6%` each, plus FP penalty. Rows with
  `clear=false` do not match and are not FP.
- `rascar_boxing/pose_heuristic.py` already emits per-candidate pose features:
  score, closing, arm_forward, reach, proximity, keypoint confidence,
  role/color confidence, normalized target/head/body distance, forward distance,
  fighter, hand, and target.
- `tools/evaluate_pose_temporal_context.py` found the best non-model context
  signal so far: same `(fighter, hand)` dominance around `+/-4` frames.
- `tools/evaluate_pose_ranker.py` has an sklearn
  `HistGradientBoostingClassifier`, but the old feature set is mostly
  single-frame and underperformed the heuristic.
- Cached tracks exist for `val_yolo11s_conf035`, `test_yolo11s_conf035`, and
  `val_yolo11m_conf035`; model packages available locally include `sklearn`,
  `lightgbm`, `catboost`, and `xgboost`.

## Feature Plan

Build a wide pool from `score_pose_tracks(..., min_score=0.0)` plus very light
pre-NMS (`nms=2..3`) only if duplicate volume is too high.

Pose candidate features:

- Existing scalar features from `PunchCandidate.features`.
- Log pose score, per-video score rank/percentile, local rank inside `+/-4`,
  `+/-8`, `+/-15`.
- Same/all/other `(fighter, hand)` score sums, counts, dominance, and local peak
  distance using prefix sums.
- Temporal shape around the candidate: wrist/elbow velocity and acceleration at
  lags `1,2,3,5`, max closing before/after, arm extension delta, distance-to-
  target minimum, and pre/post asymmetry.
- Track quality: keypoint visibility rates, role color margin, bbox scale,
  fighter distance, missing-fighter flags, and sudden role/track jumps.

CPU-light video features:

- Optional crop-motion tie-breakers only around high-recall pose candidates:
  small wrist/opponent-head/body ROI absdiff or optical-flow magnitude at
  `t-3..t+3`.
- Aggregate as max/mean/p95 motion and motion asymmetry; avoid full-frame video
  models unless a tiny validation subset shows clear signal.
- Audio can stay a weak optional feature later, but no hard frame shifting.

## Labels

- Positive anchor: for each clear GT punch, choose the best/closest candidate in
  `+/-15` frames, preferring matching fighter and hand when available.
- Soft positives: other candidates within `+/-8` of the same GT can receive
  lower weight, especially if they share fighter/hand.
- Hard negatives: high-score candidates `16..45` frames from any clear GT,
  wrong-fighter/wrong-hand candidates near real punches, and calm-frame local
  peaks.
- Far negatives: sample lightly from the rest to keep class balance sane.
- Do not train on `clear=false` punches as positives; optionally use them only
  as ignored windows so noisy labels do not create false negatives.

## OOF Protocol

- Group by fight using the existing split key:
  `dataset_type|data_root|fight_index|fight_folder`.
- Prototype on cached validation tracks with leave-one-fight/group-out OOF;
  report macro, time score, FP penalty, `n_pred`, and video wins against the
  exact `yolo11s` grouped-NMS baseline.
- Final protocol needs all train pose tracks for the chosen detector. Use
  `GroupKFold` or leave-one-group-out; fit feature normalizers, priors, count
  models, thresholds, and offsets inside each fold only.
- Use OOF predictions to choose threshold/NMS/count parameters. Train the final
  selector on all labeled train groups only after OOF gates pass.

## Models

- First baseline: sklearn `HistGradientBoostingClassifier` plus
  `HistGradientBoostingRegressor` for offset.
- Preferred CPU model if it beats sklearn: `lightgbm.LGBMClassifier` /
  `LGBMRegressor` with `num_leaves=15..63`, class/sample weights, early
  stopping on held-out groups, and fixed seeds.
- CatBoost is a fallback for robustness with categorical features
  (`fighter`, `hand`, `target`, `data_root`, `round_number`), but keep depth
  small and iterations bounded.
- Avoid neural sequence/video training until the tabular OOF path beats
  `0.340888` convincingly.

## Offset Regression

- Train offset target `gt_frame - candidate.frame` for positive anchors,
  clipped to `[-15, 15]`.
- Predict offset only for high-probability selected candidates; shrink or clip
  applied shifts to `[-8, 8]` unless OOF proves larger shifts help.
- Compare three controls: no offset, regressed offset, and local best-frame
  snapping inside `+/-4` by selector score.
- Reject any offset model that improves recall/count but lowers time score or
  increases FP penalty.

## Postprocess

- Replace heuristic score with calibrated selector probability times a small
  pose-score/context prior.
- Apply grouped NMS as the current best default:
  `nms_group_mode=fighter_hand`, same-group `8..12`, cross-group `2..4`.
- Tune count policy separately: threshold-only, root/root-round count, and
  probability-mass count. Keep sample capacity clipping and fill unused rows as
  `clear=false`.
- Keep existing attribute priors until timing/selection improves; attributes
  are lower leverage and previous attribute models moved score only slightly.
- Submit gate: beat `0.340888` by at least `+0.010` macro, no FP-penalty
  regression above `0.01`, and at least 10/13 validation-video wins.

## Speed Notes

- CPU first: parse each JSONL once into numpy arrays or a small `.npz` feature
  cache; precompute per-frame score/count arrays and prefix sums for all
  temporal windows.
- Vectorize feature extraction per video, keep features `float32`, and parallel
  only across videos/folds with bounded workers.
- Limit expensive crop-motion to top local pose candidates, not every frame.
- LightGBM/CatBoost should run on CPU with explicit thread limits; store OOF
  predictions so threshold/NMS sweeps do not retrain.
- GPU 1 is only for future missing pose extraction, not for this CPU selector:
  use `tools/run_pose_batch.py --cuda-visible-devices 1` and omit `--device`.
