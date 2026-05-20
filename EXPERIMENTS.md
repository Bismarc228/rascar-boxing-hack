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
submissions/pose_heuristic_thr08_nms6_OFFLINE_CANDIDATE.csv
submissions/pose_heuristic_grouped_fighter_thr08_same8_cross2_OFFLINE_CANDIDATE.csv
submissions/pose_heuristic_grouped_fighter_thr08_same8_cross2_rootcount09_OFFLINE_CANDIDATE.csv
submissions/pose_context_dominance_w4_a02_thr105_nms7_cross2_OFFLINE_CANDIDATE.csv
```

The grouped-NMS test candidate validates and has 737 `clear=true` rows after
sample capacity clipping, versus 699 for the global `thr=0.8,nms=6` candidate.
The root-count variant validates and has 723 `clear=true` rows.
The temporal-context candidate validates and has 739 `clear=true` rows.

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
- Grouped NMS is a small but real offline improvement. Best checked variant:
  `nms_group_mode=fighter`, `threshold=0.8`, same-group NMS `8`,
  cross-group NMS `2`: `0.24246` vs baseline `0.23802`, 9/13 video wins,
  `time=0.38748`, `fp_penalty=0.09466`, `n=1151`. This is useful for candidate
  generation, but the improvement is below the current Kaggle submit gate.
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
  plausible but did not materially beat threshold-only on the current validation
  grid. Best checked grouped-count variant was root_count `0.9`: `0.242684`
  versus grouped threshold `0.242460`.
- Frame offsets do not explain the remaining timing error for grouped NMS.
  Around the best grouped setup, offset `0` is best; `+1` drops to about
  `0.2385`, `-1` to about `0.2369`, and larger shifts degrade quickly.
- Fighter/attribute priors are not currently a useful standalone improvement.
  On the grouped baseline, simple color-confidence fighter override only moved
  `0.242460 -> 0.242501`, and root/root-round attribute priors generally
  lowered punch-type score.
- Temporal context has the best new offline signal so far, but still below an
  automatic-submit threshold. Rescoring by same `(fighter, hand)` dominance in
  a `+/-4` frame window (`alpha=0.2`) with `threshold=1.05`, same-fighter NMS
  `7`, cross-fighter NMS `2` gives `0.245871`, `time=0.396658`, 11/13 video
  wins, and `n=1170` on validation.

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
