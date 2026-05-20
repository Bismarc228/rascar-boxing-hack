# Public LB Strategy

Context as of 2026-05-20: public leaderboard is roughly `66%` of test and
private is roughly `34%`, so public movement is informative but still noisy.
Current best known public anchor is `yolo26l same_sum w4 alpha=-0.2` with
`root_rate=0.88`, public `0.13849`. Do not use GPU 0 for future extraction; any
GPU run should target GPU 1 explicitly with `--cuda-visible-devices 1` and a
quick `nvidia-smi` check. As of 2026-05-20 20:34 UTC the daily quota is
exhausted (`30/30`) and reset is `2026-05-21T00:00:00Z`; do not submit before
that reset.

## Next Experiments

1. Count calibration around `yolo26l`.
   Treat `root_rate=0.88` as the local public anchor, not the center of a wide
   blind sweep. Try a narrow CPU-only rescore grid around completed caches:
   `root_rate=0.84,0.86,0.88,0.90,0.92`, plus per-root rates if validation/root
   diagnostics show systematic over/under-counting. Track public-risk proxies:
   `n_pred`, counts per minute, counts by test root/template, and FP penalty on
   validation. Prefer a smaller count change that preserves timing score over a
   public-only count bump.

2. Threshold and context variants on `yolo26l`.
   Keep model/cache fixed and vary only candidate scoring. Start from the best
   `yolo26l` threshold/NMS/count setting, then test tight threshold shifts,
   same-group NMS `8,10,12`, cross-group NMS `2,4,6`, and temporal context
   features already used in the repo (`same_sum`, `dominance`, small windows).
   Reject variants whose gain disappears under fixed per-video counts; the next
   useful result should improve local timing or FP behavior, not just select
   more rows.

3. `yolo11s` / `yolo26l` fusion with `yolo26l` primary.
   Since `yolo26l` now beats the older `yolo11s` public anchor, use it as the
   primary source but keep `yolo11s` as a conservative witness. First test
   count-fixed timing snap: for each selected `yolo26l` event, snap frame only
   when `yolo11s` has the same `(fighter, hand)` within `+/-4` or `+/-6`
   frames and the normalized score is credible. Then test agreement boosts that
   keep `yolo26l`-only events alive but slightly demote them when `yolo11s`
   strongly disagrees nearby.

4. Asymmetric union/intersection ensemble.
   Build a narrow CPU fusion grid over cached `yolo11s` and `yolo26l` candidate
   pools using per-video score normalization. Candidate policy should be
   asymmetric: `yolo26l` events survive on their own score; `yolo11s`-only
   events need stronger normalized score or spare count budget; shared events
   get a boost inside `2/4/6` frames. Compare fixed-`yolo26l` counts,
   `root_rate=0.88`, and a slightly lower count mode such as `0.86` to avoid
   union-driven FP inflation.

5. `yolo26x` as the next post-reset candidate.
   The validation cache is complete and the first useful config is
   `same_count w10 alpha=-0.2`, threshold `0.65`, same-NMS `8`, cross-NMS `4`.
   It scores `0.374335` offline with no tournament-root risk flags. The test
   cache is complete and validated CSVs are ready for post-reset review:
   threshold-count (`844` rows), `root_rate=0.88` (`678` rows), and raw
   `root_round_rate=0.78` (`727` rows). Wait for the UTC quota reset before any
   upload.

6. yolo26x agreement/fusion generator.
   Offline agreement is stronger than direct yolo26x: `0.377949` for yolo26x
   primary plus yolo11s secondary, and `0.377090` for yolo26x plus yolo26l.
   The current script only evaluates validation, so implement a matching test
   submission generator before spending reset-day uploads on fusion.

7. Fighter identity calibration.
   Audit wrong-fighter near misses around the `yolo26l root_rate=0.88` selected
   set with timing/count fixed. Then test per-video identity smoothing or
   `yolo11s`/`yolo26l` role agreement: keep the event frame and hand unchanged,
   but allow fighter label changes only when track continuity and color/role
   evidence agree. Report fighter score, red/blue confusion, and per-root
   deltas; do not mix identity changes with new counts.

8. Timing snap with audio as a weak local feature.
   Use audio only near existing pose candidates. For selected `yolo26l` events,
   add features such as nearest onset peak within `+/-3/+/-6/+/-12`, local
   prominence, and peak rank, then test small rerank/snap policies under fixed
   counts. No audio-only rows, no hard global shift, and no submit unless timing
   score improves locally without FP regression.

9. Public/private robustness audit before spending submissions.
   Because public is large but not complete, any candidate should beat the
   `yolo26l root_rate=0.88` anchor on validation in at least one count-fixed or
   timing-specific check, be non-negative on tournament roots, and have plausible
   counts for high-capacity test videos. Avoid repeated public probing of tiny
   threshold changes; bundle only materially different hypotheses such as
   count-calibrated `yolo26l`, `yolo26l+yolo11s` agreement, identity-fixed, or
   timing-snap variants.

## Minimal Logging

For every future candidate, record: source cache, threshold/context settings,
count policy, `n_pred`, validation macro/time/FP, per-root deltas, public score
if later submitted by request, and whether GPU 0 stayed unused during any
extraction.
