# Experiments

## Kaggle Submission Policy

The competition limit is 30 submissions/day. Do not submit small threshold
sweeps. Submit only when offline validation shows a clear improvement and the
candidate is meaningfully different from already submitted variants.

## Public Submissions

Best public score so far:

```text
0.08639  submissions/pose_heuristic_thr115_nms12.csv
```

Other checked variants:

```text
0.07902  pose_heuristic_thr125_nms12.csv
0.07486  pose_heuristic_thr135_nms12.csv
0.06582  pose_heuristic_thr15_nms12.csv
0.06567  pose_heuristic_thr15_nms12_offp3.csv
0.06403  pose_heuristic_thr1_nms12.csv
0.05760  pose_heuristic_thr15_nms12_offm3.csv
0.05167  pose_heuristic_thr15_nms12_swapfighter.csv
0.04423  pose_heuristic_thr2_nms12.csv
0.00000  temporal prior baseline
0.00000  sample baseline
```

Conclusions from public checks:

- Global fighter swap is bad.
- Frame offset `+3` is neutral/slightly worse; `-3` is worse.
- Around the initial pose heuristic, recall helped up to `thr=1.15`; overly
  conservative thresholds under-predicted events.

## Metric Correction

`OVERVIEW.md` clarifies that if a matched prediction is at least 0.5 seconds
away from GT, attribute errors are maximal. The local metric now enforces this.
Self-score on train labels remains 1.0.

## Offline Validation

Current validation cache:

```text
data/processed/pose_tracks/val_yolo11n_conf035/
```

This is ignored by git. It contains the full current fight-level validation
split.

With corrected metric on all 13 validation videos:

```text
0.23802  thr=0.8   nms=6   n=1119
0.23230  thr=0.9   nms=6   n=1029
0.23072  thr=0.7   nms=6   n=1237
0.22803  thr=0.8   nms=8   n=1016
0.22744  thr=0.7   nms=8   n=1107
```

Submitted-public analogs on the same offline split:

```text
0.19836  thr=1.15  nms=12
0.19460  thr=1.25  nms=12
0.20130  thr=1.0   nms=12
0.18557  thr=1.5   nms=12
```

Candidate CSVs generated but not submitted:

```text
submissions/pose_heuristic_thr07_nms8_OFFLINE_CANDIDATE.csv
submissions/pose_heuristic_thr07_nms12_OFFLINE_CANDIDATE.csv
```

Need regenerate `thr=0.8,nms=6` before any possible submit.

## Hypotheses Checked

- Metadata count priors are useful for reasoning, but pure temporal priors are
  too weak.
- Sample submission frames/counts are not labels.
- Global fighter swap, all-red/all-blue, and confidence-threshold role fixes do
  not beat current role assignment offline.
- Supervised candidate ranker using current pose features underperforms the
  simple heuristic on mean LOOV; it is not ready for test generation.
- Supervised attribute models give only a small offline gain
  (`0.21299 -> 0.21590` in one timing setup), not enough to justify a submit.
- Global frame-diff motion-only candidates are noisy. Crop-motion reranking is
  more plausible, but the first grid did not beat the simple full-split
  threshold/NMS candidate.
- Audio streams exist in all train/test videos and test A/V stream starts are
  aligned at container level. Test videos match the manifest `frame_count / 30`
  timeline; only old `boks` train videos show small duration/FPS drift, so audio
  features should map decoded samples with `frame = round(t * 30)`.
- Audio onset detection is a weak timing signal, not a standalone detector.
  On full validation, audio-only oracle-count peaks reach only `0.02927`
  macro. Shifting pose candidates to nearby audio peaks hurts:
  `0.23802 -> 0.22862` at `+/-3` frames and `0.21543` at `+/-6` frames.
- Audio as a weak pose reranker also did not beat the full-validation baseline.
  A no-pre-NMS wide-pool grid reproduced `0.23802` only when `alpha=0`; the
  best positive audio boost seen was `0.23789` (`window=3`, `alpha=0.05`,
  `threshold=1.15`, `nms=6`, 9/13 video wins but worse FP/macro). No audio
  submission is justified yet.
- Count calibration remains a public-risk area. Sample true counts are odd on
  some test videos (`agn_062` especially); root/root-round rate counts are more
  plausible but did not beat threshold-only on the current validation grid.

## Next Useful Work

1. Build temporal-context features around pose candidates rather than scoring a
   single frame only.
2. Train a robust event selector with group/fight validation, but keep the
   simple heuristic as a fallback where the ranker fails.
3. Explore per-video count control: dense NMS improves offline, but public
   penalty may differ by test fight.
4. If revisiting audio, use it only as a learned feature
   (`max_onset +/-3/6/12`, nearest onset distance) inside a pose-dominant ranker;
   do not hard-shift frames to audio peaks.
5. Improve attributes only after timing/selection improves; current gains are
   too small.
