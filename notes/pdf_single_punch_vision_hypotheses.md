# PDF Single Punch Vision Hypotheses

Context: the PDF is Khasanshin, Utkin, and Derbin, "Application of machine
vision technologies for recognition of single direct punches in boxing"
(Science and Sport, 2022, DOI `10.36028/2308-8826-2022-10-2-43-48`). The paper
uses MediaPipe/BlazePose 33 pose landmarks and a `k-NN` classifier on still
images of final punch phases. Current challenge baseline context remains:
public best `0.13275` from `yolo11s-pose` fighter+hand grouped NMS, best local
validation `0.340888`.

## Useful Evidence

- Their most transferable choice is not the model, but the representation:
  normalize human pose landmarks and classify endpoint geometry instead of raw
  pixels.
- They include a non-punch "stance" class because a guard with a partly
  extended lead hand can look like a jab. That maps directly to this challenge's
  false-positive risk around guards, feints, and post-punch recovery.
- They split straight punches by front/back hand and stance side. This supports
  keeping `fighter_hand` grouping and testing stance-aware hand geometry rather
  than only global temporal NMS.
- Reported front-hand classes are weaker than rear-hand/cross classes:
  front-hand precision/F1 around `0.86-0.87`, cross/rear-hand classes around
  `0.94-0.96`. The likely transfer is "lead hand vs guard" confusion, not their
  absolute score.
- Dataset scale was small and controlled: 14 athletes, 150 samples per punch
  type per athlete, 8400 final-phase images, plus stance negatives. This argues
  for simple validation-only feature probes before any heavier modeling.

## Features To Check

- Arm extension: wrist-to-shoulder distance, wrist-to-nose/head distance,
  elbow angle, and hand-forward distance normalized by shoulder width or torso
  size. These should separate true final-phase punches from guard positions.
- Stance and hand role: infer front/back side from shoulder/hip orientation,
  fighter track direction, and which wrist is already forward before the
  candidate frame. Test separate thresholds for lead-hand and rear-hand
  candidates.
- Endpoint stability: score frames where extension peaks and then relaxes within
  a short window. This is closer to the PDF's final-phase setup than a raw
  per-frame pose score.
- Guard-negative heuristics: penalize candidates where both wrists stay close
  to the head/torso, elbow remains strongly bent, or hand extension has no local
  velocity/closing peak.
- Lightweight pose classifiers: try `k-NN`, logistic regression, or
  HistGradientBoosting over normalized pose features as a reranker of existing
  candidates, not as a standalone detector.
- Per-class diagnostics: report validation errors by `hand`, `fighter`,
  candidate score band, and nearest same-fighter same-hand neighbor. The PDF's
  main useful warning is class-specific confusion.

## What Does Not Transfer

- The paper classifies isolated still images of final phases; the Kaggle task is
  event localization in continuous fight video with count, timing, and false
  positive penalties.
- BlazePose has 33 landmarks, while current cached YOLO pose tracks use the COCO
  17-keypoint layout. Hand/finger detail and BlazePose-specific foot/heel
  landmarks are not available unless a new extractor is added.
- Their data is controlled, high-resolution, and single-action; challenge videos
  have occlusion, camera motion, exchanges, guards, misses, blocks, and mixed
  punch types.
- Their average score near `0.93` is not a target for this repo. It was measured
  on a different task, metric, and sampling regime.
- A pure endpoint classifier can improve candidate ranking, but it cannot solve
  exact impact timing by itself.

## Validation-Only Experiments

1. Guard confusion audit: on `val_yolo11s_conf035`, collect top false positives
   from the current `fighter_hand` grid and manually tag whether they are guard,
   feint, recovery, occlusion, or wrong fighter. Proceed only if guard/lead-hand
   cases are common enough to justify a feature gate.

2. Arm-extension ablation: add a validation-only table from cached pose tracks
   with normalized wrist reach, elbow angle proxy, wrist-head distance,
   wrist-shoulder distance, and local delta over `+/-4` and `+/-8` frames. Score
   each feature as a candidate rerank term with `alpha=0` as the control.

3. Lead-hand penalty grid: rerun selection with separate score multipliers for
   likely lead-hand candidates whose extension is low or static. Keep prediction
   counts fixed where possible so any gain is timing/FP quality, not count drift.

4. Simple pose reranker: extend the existing `tools/evaluate_pose_ranker.py`
   feature set with endpoint geometry features and run leave-one-video-out on
   complete validation tracks:

```bash
python3 tools/evaluate_pose_ranker.py --tracks-dir data/processed/pose_tracks/val_yolo11s_conf035 --label-window 15 --nms-frames 8,10,12 --top-k 30
```

5. Baseline grid control: compare every new feature against the known grouped
   NMS surface, not against a weak global baseline:

```bash
python3 tools/evaluate_pose_selection_variants.py --tracks-dir data/processed/pose_tracks/val_yolo11s_conf035 --group-modes fighter_hand --thresholds 0.8,0.9,1.0,1.15,1.3 --same-nms-frames 8,10,12 --cross-nms-frames 2,4 --count-modes threshold,root_count,root_round_count --count-multipliers 0.9,1.0 --top-k 30
```

6. Cross-model sanity: repeat only the best validation-only feature family on
   `val_yolo11m_conf035` after confirming complete frame counts. Reject features
   that only help the already public-risk `yolo11m` path.

## Submit Gates

- No Kaggle submission from PDF-inspired changes unless the best validation
  candidate beats `0.340888` by at least `+0.010` macro.
- FP penalty must not regress by more than `0.01` versus the matching
  no-feature `yolo11s` baseline.
- Require at least 10 of 13 validation-video wins versus the current
  `yolo11s-pose` candidate. A mean-only gain is not enough.
- The gain must persist when prediction counts are held close to the baseline;
  reject count inflation disguised as pose understanding.
- Lead-hand/front-hand false positives should decrease in the manual audit or
  in per-hand diagnostics. If the method does not address the PDF's strongest
  transfer signal, do not submit it.
- Test CSV generation is allowed only after validation gates pass, and the CSV
  must pass `python3 tools/validate_data.py --submission ...`.
- Submit at most one materially new PDF-inspired candidate. Do not spend public
  submissions on small threshold or multiplier sweeps.
