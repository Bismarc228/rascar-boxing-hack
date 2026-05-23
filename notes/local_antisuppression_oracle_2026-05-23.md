# Local Anti-Suppression Oracle Audit - 2026-05-23

This note records an oracle-only audit for the proposed local anti-suppression
reranker. It is not a submission generator and it is not upload-approved.

## Question

The strict yolo26l pool audit showed that most missed scorable validation punches
are already present in the pool but are suppressed before hard selection:

- strict pool coverage: `1712/1742 = 98.3%`
- same-fighter pool coverage: `1697/1742 = 97.4%`
- selected scorable coverage: `1044/1742 = 59.9%`
- same-fighter covered but not selected scorable: `654`
- locally suppressed subset: `546/654 = 83.5%`

The concrete question was whether a local post-selection swap has real headroom
before building a deployable selector.

## Tool

Added:

```bash
tools/evaluate_local_antisuppression_oracle.py
```

Inputs:

```text
predictions:
data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv

pool cases:
data/processed/timing_audits/strict_clearens_yolo26l_pool_cases_20260523.csv

suppressor pairs:
data/processed/timing_audits/strict_clearens_yolo26l_pool_suppressor_pairs_20260523.csv
```

Mechanics:

1. Read suppressed GT-positive pool candidates from the strict pool audit.
2. Sweep `pair_gap`, `replace_window`, and suppressor type.
3. For each proposal, replace a nearby selected row with the suppressed
   pool-positive candidate.
4. Preserve old `punch_type` and `effectiveness`; replace only
   `frame/fighter/hand/target`.
5. Accept a replacement only if the per-video validation metric improves.

That last step uses validation labels, so this is deliberately an oracle audit.

## Commands

Metric-guard oracle:

```bash
python3 tools/evaluate_local_antisuppression_oracle.py \
  --output-summary data/processed/timing_audits/local_antisuppression_oracle_summary_20260523.csv \
  --output-selected data/processed/timing_audits/local_antisuppression_oracle_selected_20260523.csv \
  --write-best-rows data/processed/timing_audits/local_antisuppression_oracle_best_rows_20260523.csv
```

NMS-conflict guarded oracle:

```bash
python3 tools/evaluate_local_antisuppression_oracle.py \
  --require-no-nms-conflict \
  --output-summary data/processed/timing_audits/local_antisuppression_oracle_noconflict_summary_20260523.csv \
  --output-selected data/processed/timing_audits/local_antisuppression_oracle_noconflict_selected_20260523.csv \
  --write-best-rows data/processed/timing_audits/local_antisuppression_oracle_noconflict_best_rows_20260523.csv
```

Baseline for both runs:

```text
macro=0.415261
time=0.530111
fighter=0.537953
fp_penalty=0.061715
clear rows=1188
```

## Results

Best metric-guard configuration:

```text
gap10_w90_all_metricguard
accepted=218
macro_delta=+0.059310
time_delta=+0.046299
fighter_delta=+0.062046
punch_type_delta=+0.046776
effectiveness_delta=+0.042695
hand_delta=+0.020819
target_delta=+0.018922
fp_penalty_delta=-0.013274
```

Best NMS-conflict guarded configuration:

```text
gap10_w90_all_noconflict
accepted=213
macro_delta=+0.053752
time_delta=+0.039206
fighter_delta=+0.055285
punch_type_delta=+0.047498
effectiveness_delta=+0.047597
hand_delta=+0.012740
target_delta=+0.013841
fp_penalty_delta=-0.012939
```

Tighter local variants still have large oracle headroom:

```text
gap4_w10_all_metricguard
accepted=168
macro_delta=+0.027093
time_delta=+0.021284
fighter_delta=+0.031065
fp_penalty_delta=-0.003744

gap4_w10_all_noconflict
accepted=159
macro_delta=+0.026254
time_delta=+0.020397
fighter_delta=+0.029693
fp_penalty_delta=-0.003744

gap2_w10_all_noconflict
accepted=105
macro_delta=+0.016041
time_delta=+0.013976
fighter_delta=+0.019687
fp_penalty_delta=-0.001989
```

## Interpretation

The local anti-suppression shape is real. Even under an NMS-conflict guard, a
large part of the pool oracle headroom is recovered by swapping a nearby selected
row to a suppressed pool-positive candidate.

The gain is mostly in the intended components:

- `time_delta` improves materially.
- `fighter_delta` improves because the suppressed pool-positive often carries
  the correct fighter identity.
- `fp_penalty_delta` becomes lower, so this is not only count inflation.

The positive `punch_type` and `effectiveness` deltas are side effects of better
matching after timing/fighter fixes, not direct attribute modeling, because the
tool preserves those labels from the old selected row.

## Next Step

Build a non-oracle local selector only around these suppressed-neighborhood
cases. Do not globally rerank the full pool. The deployable version should train
or calibrate a swap/no-swap decision using local features:

- gap from suppressor to pool-positive candidate
- score ratio
- same-group versus cross-group suppressor type
- local score/rank neighborhood
- selected-row weakness features
- NMS-conflict features
- video/root held-out validation

No Kaggle upload should be made from this oracle artifact.
