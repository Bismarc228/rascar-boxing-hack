# Revived Component Verification - 2026-05-23

Goal: verify the six revived component-only hypotheses from
`notes/revived_component_hypotheses_2026-05-23.md`. Six workers ran independent
checks with no Kaggle upload. Generated CSVs are under `data/processed/`.

## Summary

| hypothesis | primary component | best result | decision |
| --- | --- | --- | --- |
| timing cluster-first replacement | timing | best safe move `+0.000553` macro, time `+0.000124`, FP `-0.000447`; 40 replacements regress | fail |
| clear/FP precision-ranked gate | FP/clear | `miss` top-8: macro `+0.002797`, FP penalty `-0.004533`, drops `4 FP / 3 scorable TP / 1 time-only TP` | near miss |
| visual effectiveness transition guard | effectiveness | DINO ROI margin03 `landed_to_nonlanded`: effectiveness `+0.021803`, weighted macro `+0.001744`, `8/13` wins | pass |
| calibrated fighter flips | fighter | 20 flips: fighter `+0.004047`, macro `+0.000859`, time/FP unchanged | fail |
| punch_type CLIP/token/pose-logreg | punch_type | root-out pose-logreg ptype: ptype `+0.046665`, macro `+0.004666`, jab `+0.126635`, uppercut `+0.167722` | pass on root-out |
| target pose-geometry guard | target | 11 target changes: target `+0.003790`, macro `+0.000227`, hand unchanged | fail |

## Decisions

- Best next candidate-building path is effectiveness first: the DINO ROI
  `landed_to_nonlanded` guard meets the assigned component and weighted-macro
  thresholds while keeping all non-effectiveness columns fixed.
- Punch type is alive in root-out, not on the current strict anchor. The strict
  token/CLIP variants remain too weak or root-unstable, but root-out pose-logreg
  restores meaningful jab/uppercut recall.
- Clear/FP has a useful small diagnostic (`miss` top-8), but it misses the
  strict FP target and is concentrated in `agn_003/010`; treat as a precision
  feature, not a standalone candidate yet.
- Timing cluster-first, calibrated fighter flips, and target-only guard all
  show weak positive component movement, but none is strong enough for a
  candidate by itself.

## Worker Notes

```text
notes/timing_cluster_first_replacement_worker_2026-05-23.md
notes/clear_precision_ranked_gate_worker_2026-05-23.md
notes/effectiveness_transition_guard_worker_2026-05-23.md
notes/calibrated_fighter_flip_worker_2026-05-23.md
notes/punch_type_component_worker_2026-05-23.md
notes/target_component_worker_2026-05-23.md
```

## Tooling Note

`tools/score_predictions.py` and `tools/summarize_prediction_components.py`
were updated after this review so future scoring output lists component means
and component deltas before macro. This avoids hiding large component shifts
behind small metric weights.
