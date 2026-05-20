# Experiments

## Kaggle Submission Policy

The competition limit is 30 submissions/day. Do not submit small threshold
sweeps. Submit only when offline validation shows a clear improvement and the
candidate is meaningfully different from already submitted variants.

## Public Submissions

Best public score so far:

```text
0.13849  submissions/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088_OFFLINE_CANDIDATE.csv
```

Other checked variants:

```text
0.13664  submissions/yolo26l_context_samesum_w6_am02_thr085_same10_cross4_rootrate088_OFFLINE_CANDIDATE.csv
0.13664  submissions/yolo26l_samesum_w6_am02_thr085_same10_cross4_rootrate084_OFFLINE_CANDIDATE.csv
0.13664  submissions/yolo26l_samesum_w6_am02_thr085_same10_cross4_rootrate086_OFFLINE_CANDIDATE.csv
0.13664  submissions/yolo26l_samesum_w8_am02_thr085_same10_cross4_rootrate088_OFFLINE_CANDIDATE.csv
0.13580  submissions/yolo26l_context_samecount_w8_am01_thr085_same10_cross4_rootrate088_OFFLINE_CANDIDATE.csv
0.13580  submissions/yolo26l_context_samecount_w8_am01_thr085_same10_cross4_rootrate09_OFFLINE_CANDIDATE.csv
0.12958  submissions/yolo11m_context_samesum_w8_am02_thr09_same10_cross4_OFFLINE_CANDIDATE.csv
0.12020  submissions/yolo26l_samesum_w6_am025_thr085_same10_cross4_rootrate088_OFFLINE_CANDIDATE.csv
0.10260  submissions/yolo26l_context_samecount_w8_am01_thr085_same10_cross4_threshold_OFFLINE_CANDIDATE.csv
0.08639  pose_heuristic_thr115_nms12.csv
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

- Larger pose model transferred to public: `yolo11s-pose` with fighter/hand
  grouped NMS improved public score from `0.08639` to `0.13275`.
- `yolo11m-pose` plus same-score temporal context did not transfer despite a
  large offline gain: public `0.12958`, below the `yolo11s` best.
- `yolo26l-pose` transferred and is now the public anchor. The best checked
  public variant is `same_sum`, window `4`, alpha `-0.2`, threshold `0.85`,
  same-group NMS `10`, cross-NMS `4`, `root_rate=0.88`: public `0.13849`.
- Dense threshold-count on the same `yolo26l` pool failed hard (`0.10260`), so
  public currently rewards precision/count control more than raw recall.
- The daily budget was exhausted at `30/30` submissions on 2026-05-20. The
  helper reports reset at `2026-05-21T00:00:00Z`; do not submit again before
  that reset.
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
submissions/yolo11s_fighterhand_thr115_same10_cross4_rootcount1_OFFLINE_CANDIDATE.csv
```

The grouped-NMS test candidate validates and has 737 `clear=true` rows after
sample capacity clipping, versus 699 for the global `thr=0.8,nms=6` candidate.
The root-count variant validates and has 723 `clear=true` rows.
The temporal-context candidate validates and has 739 `clear=true` rows.

`yolo11s-pose` candidates produced the first large offline and public jump. The
full validation cache
`data/processed/pose_tracks/val_yolo11s_conf035/` is complete. Best checked
offline config on that cache is:

```text
0.340888  yolo11s, fighter_hand grouped NMS, thr=1.15,
          same-group nms=10, cross-group nms=4, root_count=1.0, n=1202
```

This is a large improvement over both the `yolo11n` simple baseline (`0.23802`)
and the best `yolo11n` temporal-context candidate (`0.245871`). The matching
test candidate validates, has 730 `clear=true` rows, and scored `0.13275`
public.

`yolo11m-pose` validation tracks are complete in
`data/processed/pose_tracks/val_yolo11m_conf035/`. They improve again over
`yolo11s` on the same 13-video split:

```text
0.363216  yolo11m, fighter_hand grouped NMS, thr=0.9,
          same-group nms=8, cross-group nms=4, root_count=1.0, n=1385
0.359951  yolo11m, fighter_hand grouped NMS, thr=0.9,
          same-group nms=8, cross-group nms=4, root_count=0.8, n=1195
0.363325  yolo11m, same_sum context window=8 alpha=-0.2,
          fighter_hand grouped NMS, thr=0.9, same-group nms=10,
          cross-group nms=4, n=1198
```

The last point is the current cleaner candidate: it beats `yolo11s` by
`+0.022437` macro, has timing score `0.493503`, FP penalty `0.065930`,
and wins 11/13 validation videos. A focused frame-offset sweep around the
`yolo11m` grid kept offset `0` at the top, so there is no evidence for a global
sync shift. The matching public submission scored `0.12958`, below the
`yolo11s` public best, so the `yolo11m` validation lift is currently an overfit
or test-distribution mismatch signal rather than a new baseline.

YOLO26 pose models are installed through the current Ultralytics package and
are being checked as larger detector witnesses. `yolo26s-pose` is not
competitive on the validation split:

```text
0.305323  yolo26s, fighter_hand grouped NMS, thr=0.7,
          same-group nms=12, cross-group nms=4, n=1366
```

Its track quality is worse than both `yolo11s` and `yolo11m`:

```text
model    both_roles  red_present  blue_present  red_sw/1k  blue_sw/1k
yolo11s  0.8430      0.9326       0.9057        11.20      11.30
yolo11m  0.8693      0.9405       0.9235         8.65       7.90
yolo26s  0.7948      0.9062       0.8796        20.18      16.53
yolo26m  0.8400      0.9281       0.9078        12.86       8.26
```

`yolo26m-pose` recovers most of the track quality and has a small local signal,
but not enough transfer evidence for a submit:

```text
0.350484  raw yolo26m, fighter_hand grouped NMS, thr=0.85,
          same-group nms=10, cross-group nms=2,
          root_round_count=0.9, time=0.508861, fp=0.091754, n=1293
0.352804  yolo26m same_sum context, window=8, alpha=-0.2,
          fighter_hand grouped NMS, thr=0.85,
          same-group nms=10, cross-group nms=2,
          time=0.478999, fp=0.063795, n=1165
```

The context candidate beats the `yolo11s` anchor overall by `+0.011916`, but
the root audit is not public-safe: `Турнир Бокс` is `-0.010997`,
`Турнир Бокс 2` is `-0.000852`, and the gain comes from the old `бокс` root
(`+0.064619`). The nearby conservative variants show the same pattern. Do not
submit `yolo26m` threshold/context candidates unless a materially different
agreement, snapping, or fusion test fixes the tournament-root regression.

`yolo26m` is slightly useful as a timing witness while keeping `yolo11s` counts
fixed. Snapping `yolo11s` selected frames to nearby `yolo26m` raw candidates
with the same `(fighter, hand)` gives:

```text
0.343372  yolo11s fixed-count events, yolo26m raw timing snap,
          window=8, sec_thr=0.85, score-distance weighted frame,
          time=0.471365, fp=0.068115, wins=10/13, n=1202
```

The gain is small but well distributed by root (`+0.003762`,
`+0.002910`, `+0.000211`). This is below the submit gate, but it supports using
large models as secondary timing witnesses rather than wholesale replacements.

Fighter identity still has meaningful headroom, but not through a full-video
label swap. On the `yolo11s` anchor, swapping all red/blue labels worsens
`0.340888 -> 0.271555`, and per-video oracle keep/swap chooses keep for all 13
validation videos. However, an event-level oracle that fixes only the `fighter`
label on matched predictions raises `yolo11s` to `0.359788` (`+0.018899`) and
raises the `yolo26m` context candidate to `0.369845` (`+0.017041`). So the next
fighter-identity work should be local tracklet/per-video color calibration and
switch smoothing, not a global or whole-video swap.

`yolo26l-pose` validation tracks are complete in
`data/processed/pose_tracks/val_yolo26l_conf035/`. It is weak as a plain global
NMS replacement, but strong with grouped NMS and temporal context:

```text
0.360318  raw yolo26l, fighter_hand grouped NMS, thr=0.95,
          same-group nms=8, cross-group nms=4, threshold count,
          time=0.510790, fp=0.084732, n=1289
0.369802  yolo26l same_count context, window=8, alpha=-0.1,
          fighter_hand grouped NMS, thr=0.85,
          same-group nms=10, cross-group nms=4,
          time=0.495463, fp=0.061657, wins=10/13, n=1185
0.366769  yolo26l same_sum context, window=6, alpha=-0.2,
          fighter_hand grouped NMS, thr=0.85,
          same-group nms=10, cross-group nms=4,
          time=0.479037, fp=0.050365, wins=10/13, n=1126
0.365177  yolo26l same_sum context, window=4, alpha=-0.2,
          fighter_hand grouped NMS, thr=0.85,
          same-group nms=10, cross-group nms=4,
          root_rate=0.88, time=0.481575, fp=0.053874, n=1136
```

The top context candidate beats the `yolo11s` anchor by `+0.028913` macro and
improves mean FP penalty by `-0.006490`, but it still has a public-risk flag:
`Турнир Бокс` is `-0.006526`, while `Турнир Бокс 2` is `+0.021366` and old
`бокс` is `+0.081964`. The test manifest has three `Турнир Бокс` videos and
six `Турнир Бокс 2` videos, so this is plausible but not automatic-submit
safe. `test_yolo26l_conf035` extraction was started for a candidate CSV and
local validation/count audit. Public checks changed the practical anchor: the
`same_sum` `window=4` `alpha=-0.2` variant has weaker offline macro than the
`same_count` offline winner, but scored best publicly (`0.13849`). Nearby
`root_rate=0.84/0.86/0.88` and `window=6/8` variants all clustered at
`0.13664`, while `alpha=-0.15` dropped to `0.13095` and `alpha=-0.25` dropped
to `0.12020`.

The public-mask probes that zeroed individual test videos imply at least
`agn_038` is in the public split: zeroing it dropped score to `0.06940`, while
zeroing `agn_047`, `agn_049`, `agn_062`, `agn_063`, `agn_064`, or `agn_039`
left the score at `0.13664` for that baseline. Treat this as useful evidence,
but do not spend more submissions on mask probing before the UTC reset.

`yolo26x-pose` validation tracks are complete in
`data/processed/pose_tracks/val_yolo26x_conf035/`. It is the first larger
model that passes the tournament-root audit cleanly:

```text
0.371622  raw yolo26x, fighter_hand grouped NMS, thr=0.65,
          same-group nms=8, cross-group nms=6,
          root_round_rate=0.78, time=0.513862, fp=0.074331,
          wins=10/13, n=1229, risk_flags=none
0.374335  yolo26x same_count context, window=10, alpha=-0.2,
          fighter_hand grouped NMS, thr=0.65,
          same-group nms=8, cross-group nms=4,
          threshold count, time=0.491512, fp=0.052324,
          wins=10/13, n=1153, risk_flags=none
0.372166  same yolo26x context candidate with root_rate=0.88,
          time=0.486328, fp=0.050154, wins=10/13, n=1135,
          risk_flags=none
```

Root deltas for the top yolo26x context candidate versus the public-proven
`yolo11s` anchor are positive on all roots: `Турнир Бокс +0.043377`,
`Турнир Бокс 2 +0.022672`, `бокс +0.048659`. `test_yolo26x_conf035`
extraction was started on GPU 1 after submissions were stopped so a candidate
CSV can be prepared after reset without waiting on detector inference.

The `test_yolo26x_conf035` cache completed on GPU 1. Three post-reset candidate
CSVs were generated and validated locally, but not submitted because the daily
quota is exhausted:

```text
submissions/yolo26x_samecount_w10_am02_thr065_same8_cross4_threshold_OFFLINE_CANDIDATE.csv
  selected=agn_037:84,agn_038:63,agn_039:60,agn_047:112,agn_048:121,
           agn_049:96,agn_062:110,agn_063:103,agn_064:95  total=844
submissions/yolo26x_samecount_w10_am02_thr065_same8_cross4_rootrate088_OFFLINE_CANDIDATE.csv
  selected=agn_037:52,agn_038:63,agn_039:57,agn_047:112,agn_048:44,
           agn_049:62,agn_062:110,agn_063:103,agn_064:75  total=678
submissions/yolo26x_raw_thr065_same8_cross6_rootroundrate078_OFFLINE_CANDIDATE.csv
  selected=agn_037:52,agn_038:91,agn_039:57,agn_047:100,agn_048:44,
           agn_049:62,agn_062:126,agn_063:120,agn_064:75  total=727
```

A first normalized agreement grid found stronger validation numbers, but it
needed a dedicated submission generator before it could be tested publicly:

```text
0.377949  yolo26x primary + yolo11s secondary agreement,
          window=4, alpha=0.2, primary_weight=1.0,
          secondary_weight=0.8, threshold=1.4, nms=10,
          cross=2, root_count=0.88, n=1284
0.377090  yolo26x primary + yolo26l secondary agreement,
          window=4, alpha=1.0, primary_weight=1.0,
          secondary_weight=0.8, threshold=1.4, nms=10,
          cross=2, root_count=0.84, n=1283
```

`tools/make_pose_agreement_submission.py` now generates matching test
submissions. Two post-reset CSVs were generated and validated locally:

```text
submissions/yolo26x_yolo11s_agree_w4_a02_pw10_sw08_thr14_nms10_cross2_rootcount088_OFFLINE_CANDIDATE.csv
  selected=agn_037:52,agn_038:108,agn_039:57,agn_047:90,agn_048:44,
           agn_049:62,agn_062:125,agn_063:114,agn_064:75  total=727
submissions/yolo26x_yolo26l_agree_w4_a10_pw10_sw08_thr14_nms10_cross2_rootcount084_OFFLINE_CANDIDATE.csv
  selected=agn_037:52,agn_038:104,agn_039:57,agn_047:99,agn_048:44,
           agn_049:62,agn_062:119,agn_063:119,agn_064:75  total=731
```

`tools/audit_pose_agreement_candidate.py` adds the root/video audit for these
fusion configs. Both agreement candidates pass `risk_flags=none`:

```text
yolo26x + yolo11s agreement:
  macro=0.377949, time=0.533478, fp=0.086935, wins=10/13, n=1284
  root deltas: Турнир Бокс +0.032473, Турнир Бокс 2 +0.038638, бокс +0.037968

yolo26x + yolo26l agreement:
  macro=0.377090, time=0.537665, fp=0.091709, wins=12/13, n=1283
  root deltas: Турнир Бокс +0.025302, Турнир Бокс 2 +0.031736, бокс +0.057522
```

Because public just punished over-dense recall variants, the lower-FP
`yolo26x + yolo11s` fusion is the cleaner first post-reset candidate despite
having fewer validation video wins.

`tools/splice_submission_videos.py` can now build public/private hybrid CSVs by
replacing only selected test-video rows. Since `agn_038` is the confirmed
public-impact video, three validated `agn_038` hybrids are ready for post-reset
probing:

```text
submissions/hybrid_yolo26l_best_agn038_yolo26x_yolo11s_agree_OFFLINE_CANDIDATE.csv
  base=yolo26l public best, override agn_038 from yolo26x+yolo11s agreement,
  selected agn_038=108, total=680
submissions/hybrid_yolo26l_best_agn038_yolo26x_yolo26l_agree_OFFLINE_CANDIDATE.csv
  base=yolo26l public best, override agn_038 from yolo26x+yolo26l agreement,
  selected agn_038=104, total=676
submissions/hybrid_yolo26l_best_agn038_yolo26x_context_rootrate088_OFFLINE_CANDIDATE.csv
  base=yolo26l public best, override agn_038 from yolo26x context root_rate=0.88,
  selected agn_038=63, total=635
```

A fast learned-selector sanity check on the yolo26x cache did not justify
another candidate-ranker path. Command shape:

```text
tools/evaluate_learned_temporal_selector.py --tracks-dir val_yolo26x_conf035
  --model hgb --max-candidates-per-video 1200 --windows 2,4,8
```

Best result was only:

```text
0.303634  time=0.536335, fp=0.157001, n=1476,
          pose_prior=0.2, threshold=0.12, nms=8, cross=4,
          root_rate=0.84
```

An ad hoc Gaussian-label HGB candidate regressor on the same yolo26x cache was
better but still below the heuristic/fusion anchor:

```text
0.361257  time=0.534848, fp=0.105266, n=1323,
          pose_prior=0.4, threshold=0.30, nms=8, cross=2,
          root_rate=0.78
```

This is below direct yolo26x (`0.374335`) and yolo26x agreement (`0.377949`).
Treat simple candidate-level rankers on the current features as killed unless
the feature set changes substantially; the next learned approach should be a
calibrated sequence/anchor spotter with explicit count/FP control.

External research notes point in the same direction: treat impact spotting as
the primary problem, keep pose/track identity as support, calibrate fighter
colors per video, and use refractory windows rather than a single global NMS.
T-DEED/E2E-Spot-style Gaussian-label spotting is the next larger modeling path
if cached pose heuristics stop improving.

A first per-video count-controller tool was added in
`tools/evaluate_count_controller.py`. It treats count/precision as a learned
postprocess over fixed candidates, then reuses the existing `fighter_hand` NMS.
Two offline checks did not beat the existing count policies:

```text
direct yolo26x context:
  best policy       0.374359  root_round_count=1.0
  count controller  0.374335  ridge residual_count
  oracle count      0.374335

yolo26x + yolo11s agreement:
  best policy       0.377949  root_count=0.88
  count controller  0.375863  ridge residual_count
  oracle count      0.377747
```

For the direct yolo26x context anchor, the base threshold already limits the
selected rows enough that count prediction has little room. For the agreement
anchor, the learned count controller inflated FP on several videos. Keep
root/count policies as the active baseline; revisit learned count control only
with a materially richer confidence model or a looser candidate pool.

A pose-sequence TCN evaluator was added in
`tools/evaluate_pose_sequence_spotter.py`. It predicts per-frame probability
for each `(fighter, hand)` stream from yolo26x candidate channels plus
yolo11s/yolo26l witness channels, then emits only primary yolo26x candidates
through the existing grouped NMS/count gates. Single-seed runs were noisy
(`0.381271` once, `0.371275` on repeat), so the useful path is seed ensembling.

The best checked 3-seed ensemble (`seeds=41,42,43`, `epochs=8`,
`chunks_per_epoch=1600`) reached:

```text
0.382073  time=0.578456, fp=0.123397, wins=11/13, n=1433,
          pose_prior=0.4, threshold=0.5, nms=10, cross=2,
          root_count=0.92

0.379314  time=0.568053, fp=0.117141, wins=11/13, n=1388,
          pose_prior=0.4, threshold=0.5, nms=10, cross=2,
          root_count=0.88
```

Root audit for the `0.382073` point:

```text
Турнир Бокс    score=0.411245  time=0.543674  fp=0.072521  n=333
Турнир Бокс 2  score=0.424374  time=0.606603  fp=0.103566  n=850
бокс           score=0.254197  time=0.547560  fp=0.220546  n=250
```

The weak `бокс` root is not in test; all test videos are tournament videos, so
this is a real post-reset candidate family. A matching generator was added in
`tools/make_pose_sequence_submission.py`. Two validated test CSVs were written:

```text
submissions/seq_tcn_yolo26x_witness_3seed_thr05_nms10_cross2_rootcount088_OFFLINE_CANDIDATE.csv
  selected=agn_037:52,agn_038:108,agn_039:57,agn_047:125,agn_048:44,
           agn_049:62,agn_062:125,agn_063:125,agn_064:75,total=773

submissions/seq_tcn_yolo26x_witness_3seed_thr05_nms10_cross2_rootcount092_OFFLINE_CANDIDATE.csv
  selected=agn_037:52,agn_038:113,agn_039:57,agn_047:131,agn_048:44,
           agn_049:62,agn_062:131,agn_063:131,agn_064:75,total=796
```

Because public has punished dense variants before, the `root_count=0.88`
sequence CSV is the safer first sequence submit; `root_count=0.92` is the
higher-offline, higher-count variant.

A follow-up postprocess sweep added `snap_window` to
`tools/evaluate_pose_sequence_spotter.py`: after candidate selection, output
frames can be shifted to the local TCN probability maximum without changing
fighter/hand/count. This materially improved the sequence family:

```text
0.390013  time=0.560876, fp=0.101781, wins=11/13, n=1357,
          pose_prior=0.4, threshold=0.6, nms=10, cross=2,
          snap_window=4, root_count=0.92

0.389963  time=0.545952, fp=0.088316, wins=11/13, n=1292,
          pose_prior=0.2, threshold=0.6, nms=10, cross=2,
          snap_window=4, root_count=0.88
```

Root audit for the `0.390013` point:

```text
Турнир Бокс    score=0.390725  time=0.521094  fp=0.072744  n=333
Турнир Бокс 2  score=0.434546  time=0.590635  fp=0.083208  n=806
бокс           score=0.285390  time=0.531224  fp=0.174156  n=218
```

`tools/make_pose_sequence_submission.py` now supports the same `--snap-window`
postprocess. Two validated snap4 test CSVs were written:

```text
submissions/seq_tcn_yolo26x_witness_3seed_thr06_nms10_cross2_snap4_rootcount088_OFFLINE_CANDIDATE.csv
  selected=agn_037:52,agn_038:81,agn_039:57,agn_047:125,agn_048:44,
           agn_049:62,agn_062:125,agn_063:123,agn_064:75,total=744

submissions/seq_tcn_yolo26x_witness_3seed_thr06_nms10_cross2_snap4_rootcount092_OFFLINE_CANDIDATE.csv
  selected=agn_037:52,agn_038:85,agn_039:57,agn_047:131,agn_048:44,
           agn_049:62,agn_062:131,agn_063:128,agn_064:75,total=765
```

The `root_count=0.88` snap4 CSV is the safer sequence candidate: it beats the
older `root_count=0.92` sequence validation score while reducing test rows.

A learned fighter-correction diagnostic was added in
`tools/evaluate_fighter_identity_model.py`. It keeps selected events fixed and
trains an OOF model to decide whether to keep or flip only the `fighter` label.
On the direct yolo26x context anchor, the oracle matched-fighter upper bound is
large:

```text
baseline               0.374335  fighter=0.473601  n=1153
oracle_matched_fighter 0.392941  fighter=0.566183  n=1153
```

But models using current cached pose/color/track features did not recover that
headroom:

```text
HGB keep/flip model:    best 0.374335 with 0 flips; any flips regressed
balanced logistic model best 0.374173 with 5 flips; larger flip sets regressed
```

Together with the older `track_role_majority` and `track_color_mean` regressions
on yolo26x, this kills lightweight identity correction from current
`score_red`/`score_blue` and track-majority signals. The next fighter-identity
experiment needs real per-video ROI/tracklet appearance extraction from raw
frames, keeping timing/count fixed.

A first raw-frame ROI/tracklet appearance evaluator was added in
`tools/evaluate_per_video_fighter_identity_calibration.py`. It samples source
frames, extracts upper/lower bbox HSV/Lab/chromaticity descriptors per track,
clusters tracks into two video-local identities, and evaluates only fighter
label remapping with event timing/count fixed.

On the yolo26x context anchor, appearance clusters were usually internally
pure but did not align with useful GT fighter corrections:

```text
baseline            0.374335  fighter=0.473601  n=1153
role_map_gated      0.370425  fighter=0.454804  changed=68
role_map_all        0.368859  fighter=0.447336  changed=98
oracle_cluster_map  0.368859  fighter=0.447336  changed=98
```

The oracle cluster mapping failing is the important result: simple bbox-level
appearance clustering separates something stable, but not the identity signal
needed for matched punch fighter labels. Do not build a test submission from
this identity branch. Future identity work needs pose-guided ROI quality, real
tracklet continuity/splitting, or a stronger visual embedding; otherwise keep
current fighter labels.

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
- `tools/evaluate_crop_motion_context.py` now tests crop-motion as a matched
  rescoring ablation on the current yolo26x context anchor. A quick
  two-tournament-video smoke (`agn_023,agn_069`, max 3000 candidates/video,
  resize 160) showed possible signal but not a submit-ready policy:

```text
baseline       0.401618  time=0.494436  fp=0.031065  n=219
best smoke     0.422798  alpha=0.1 beta=0.1  n=267
small-gain     0.405940  alpha=-0.03 beta=0.05  n=224
```

  The best smoke result gets its gain with row inflation and worse FP on one
  root. Continue only as a fixed-count/full-validation feature ablation, not a
  direct submit branch.
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
- A matched current-anchor audio rerank was added in
  `tools/evaluate_audio_pose_context.py` and tested on the yolo26x
  `same_count w10 alpha=-0.2, threshold=0.65, same NMS=8, cross=4` anchor.
  The `alpha=0` control exactly reproduces the pose anchor. Any nonzero local
  onset multiplier regressed:

```text
threshold baseline     0.374335  time=0.491512  fp=0.052324  n=1153
best nonzero audio     0.365845  window=6  alpha=0.03  n=1364
root_rate=0.88 base    0.372166  time=0.486328  fp=0.050154  n=1135
best nonzero audio     0.371687  window=6  alpha=0.03  n=1285
```

  Positive audio boosts improve raw timing but inflate row count and FP; negative
  boosts cut recall too hard. Keep audio only as a possible learned feature in a
  richer reranker, not as a direct multiplicative rescore.
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
- Larger pose model is the first submit-grade offline signal. On the same
  validation split, `yolo11s-pose` raises the best local score to `0.340888`.
  The best `yolo11s` grid point uses fighter+hand grouped NMS rather than the
  earlier context boost; context did not beat the grouped selector on
  `yolo11s`.
- `yolo11m-pose` is another submit-grade offline signal. The best raw grid
  point reaches `0.363216`, and a lower-FP temporal-context variant reaches
  `0.363325` with plausible prediction count. Public did not improve, so do not
  submit more `yolo11m` threshold/context tweaks without a new validation
  argument.
- A post-public `yolo11m` count audit did not justify another submit. Adding
  root-count capping to the submitted same-sum context setup only moved offline
  `0.363325 -> 0.363567`; the strongest low-FP same-count context variant was
  `0.362558`. These are threshold/context/count tweaks around the failed public
  candidate, not a materially new approach.
- Cross-model `yolo11s`/`yolo11m` agreement did not produce a submit candidate.
  A cached union/agreement grid topped out at `0.357725`, below the failed
  `yolo11m` context score and not materially better than the public-proven
  `yolo11s` path.
- A quick PDF-inspired false-positive audit supports the "guard/stance
  negative" hypothesis but not a simple hard filter. On the `yolo11s` validation
  candidate, matched predictions have mean `reach=1.224` and
  `arm_forward=0.622`, while unmatched predictions have mean `reach=0.986` and
  `arm_forward=0.533`. The separation is real but overlapping, so any endpoint
  feature should be a learned/rerank feature, not a blunt threshold.
- First learned temporal-selector smoke tests on `yolo11m` OOF are not
  competitive. LightGBM with a wide `+/-15` positive window topped out at
  `0.276141`; a stricter `+/-4` label window improved to `0.315975`, but both
  remain below the heuristic/context `0.363325`. This path needs cleaner anchor
  labels, offset regression, or a different postprocess before any submit.
- `yolo26l` is the first YOLO26 model with a stronger local candidate than
  `yolo11m`: top context score `0.369802`, lower FP than `yolo11s`, and 10/13
  wins. The blocker is a small `Турнир Бокс` root regression, not full-split
  score.

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
6. If building a learned spotter, use Gaussian labels around impact frames and
   keep the current pose selector as a precision-oriented fallback.
