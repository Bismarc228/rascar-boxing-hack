# Revived Component Hypotheses - 2026-05-23

Goal: re-open hypotheses that may have been rejected too early because the
first test used the wrong proxy: public-only movement, broad row churn, macro
instead of a single metric component, weak first implementation, or an old
anchor.

## Selection Rules

- Check one primary score component at a time: timing, FP/clear/count,
  effectiveness, fighter, punch_type, or target.
- Require prior evidence: oracle headroom, positive component delta, per-video
  lift, AUC/AP signal, or a clear failure analysis that says the first
  implementation was not the right target.
- Treat row churn as an upload/private-risk flag, not as a reason to skip
  internal validation.
- Do not use GPU 0. Do not upload without current explicit approval. Validate
  any candidate CSV before upload.
- Do not use quarantined public-LB hacks as anchors.

## Ranked Shortlist

1. Timing cluster-first local replacement.
   Prior: strict OOF + YOLO26L oracle local replacements improve
   `0.415261 -> 0.457824`, time `0.530111 -> 0.565805`, with `151` accepted
   replacements. Global pool selectors and broad top-200 application are killed,
   but that does not kill a two-stage detector: first identify bad local
   selected-row clusters, then rank candidates inside them.

2. FP/clear precision-ranked non-landed gate.
   Prior: oracle FP drop has large headroom (`0.407232 -> 0.480698`), guarded
   `blocked,miss` p_keep gives a small positive OOF result (`0.407232 ->
   0.408272`), and combined pose+ViT clear ensemble previously improved
   `0.410125 -> 0.414485`. Raw global thresholds are killed, but precision
   ranking under a non-landed guard remains open.

3. Effectiveness visual transition guard.
   Prior: VideoMAE d2res effectiveness improves the component
   `0.278732 -> 0.314986` and macro `0.408457 -> 0.411357`. Public movement was
   weak, but public LB is now known to be only three videos. Re-test should focus
   on effectiveness transition guards and per-video risk.

4. Fighter dual-attacker plus calibration.
   Prior: fighter oracle headroom is around `+0.014..0.018` macro, dual-attacker
   strict OOF gave `0.415261 -> 0.415832`, and appearance calibration separates
   most test videos. Global color swaps and crop clustering are killed, but
   event-level calibrated fighter flips remain open.

5. Punch type CLIP/token head.
   Prior: CLIP ptype raises `0.203789 -> 0.215465`, candidate-token ptype margin
   raises `0.203789 -> 0.213339`, and root-out ptype-only component delta was
   `+0.046665`. DINO public ptype failed, but that does not kill CLIP/token or
   pose-logreg ptype-only work.

6. Target pose-geometry correction.
   Prior: root-out target-only correction improves `0.304879 -> 0.307061` and
   target `0.388285 -> 0.424656`, with `48` wins. The rejected test artifact had
   broad public-sensitive churn; target-only component evidence remains valid.

## Assigned Verification Shape

- Timing: apply at most a small high-confidence replacement set; pass requires
  macro `+0.005`, time `+0.008`, FP non-worse.
- FP/clear: fixed rows only; pass requires FP penalty improvement around
  `-0.006`, macro `+0.002`, dropped FP / dropped scorable TP above `1.0`.
- Effectiveness: fixed rows only; pass requires effectiveness `+0.020`, macro
  `+0.0015`, and limited per-video damage.
- Fighter: fixed rows only; pass requires fighter `+0.010` or macro `+0.002`,
  low flip count, time/FP unchanged.
- Punch type: fixed rows only; pass requires punch_type at least `0.235` or
  weighted macro `+0.003`, with jab/uppercut recall gains.
- Target: fixed rows only; pass requires target `+0.020` or macro `+0.0012`,
  hand unchanged, and non-negative tournament-root deltas.
