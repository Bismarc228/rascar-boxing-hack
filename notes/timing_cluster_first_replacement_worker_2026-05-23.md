# Timing Cluster-First Replacement Worker - 2026-05-23

Goal: verify revived hypothesis #1, timing cluster-first local replacement. No
GPU and no Kaggle upload were used.

## Tool

Added:

```text
tools/evaluate_timing_cluster_first_replacements.py
```

The audit:

- reads strict OOF baseline rows;
- uses OOF cluster-first row probabilities from the RF cluster audit;
- builds local pose-pool replacement candidates only for the top cluster rows;
- trains a second-stage OOF candidate score by fight group;
- applies thresholded replacements with `max_replacements_total <= 40`;
- reports macro, timing, fighter, and FP penalty by threshold.

The artifact run used `target-mode=single_delta`, which trains the second stage
on single-replacement local metric delta. This was stronger than the sparse
exact-oracle candidate-key target in exploratory checks.

## Command

```bash
.venv/bin/python tools/evaluate_timing_cluster_first_replacements.py \
  --baseline-predictions data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26l_conf035 \
  --oracle-audit data/processed/timing_audits/strict_clearens_yolo26l_oracle_replace_keep_audit_20260523.csv \
  --cluster-first-probs data/processed/timing_audits/strict_clearens_yolo26l_cluster_first_rf_20260523.csv \
  --target-mode single_delta \
  --model rfreg \
  --combine-mode product \
  --max-first-stage-rows 160 \
  --max-candidates-per-row 80 \
  --max-replacements-total 40 \
  --max-replacements-per-video 6 \
  --thresholds 0.005,0.002,0.0015,0.001,0.00075,0.0005,0.00025,0.0001,0,-0.0005,-0.001 \
  --output-sweep data/processed/timing_audits/timing_cluster_first_replacement_sweep_20260523.csv \
  --output-selected data/processed/timing_audits/timing_cluster_first_replacement_selected_20260523.csv \
  --output-rows data/processed/timing_audits/timing_cluster_first_replacement_rows_20260523.csv
```

## Result

Setup summary:

```text
first_stage_rows=160
first_stage_oracle_rows=33/151
proposals=19308
positive_delta_proposals=2983
candidate_auc=0.586481
candidate_ap=0.248372
combined_ap=0.244361
```

Component sweep:

```text
label       macro     macro_delta time      time_delta fighter   fighter_delta fp_penalty fp_penalty_delta repl
baseline    0.415261  +0.000000   0.530111  +0.000000  0.537953  +0.000000     0.061715   +0.000000        0
thr 0.001   0.415814  +0.000553   0.530235  +0.000124  0.538419  +0.000466     0.061268   -0.000447        1
thr 0       0.415756  +0.000495   0.530114  +0.000004  0.538419  +0.000466     0.061268   -0.000447        4
thr -0.0005 0.413377  -0.001884   0.528695  -0.001416  0.534376  -0.003577     0.061710   -0.000005        40
```

Best threshold was `0.001`:

```text
macro_delta=+0.000553
time_delta=+0.000124
fighter_delta=+0.000466
fp_penalty_delta=-0.000447
replacements=1
oracle_hits=0
```

Pass criteria were not met:

```text
macro required +0.005: failed
time required +0.008:  failed
FP non-worse:          passed
```

## Failure

The cluster-first row detector still has oracle headroom, but the second-stage
OOF candidate score does not convert that row signal into enough safe local
replacements. At high confidence it finds only one useful non-oracle move. At
the permissive threshold needed to reach the `<=40` replacement cap, timing and
fighter regress and macro drops. The official active thresholds selected zero
exact oracle replacements, so the current cluster-first local replacement policy
does not pass.

## Strict Top-80 Follow-Up

After the initial run, a stricter proposal regime checked whether the failure
was mainly proposal-set noise:

```text
max_first_stage_rows=80
max_candidates_per_row=40
max_replacements_total=20
max_replacements_per_video=3
```

Four variants were evaluated:

```text
rfreg product single_delta
rfreg sqrt_product single_delta
etreg product single_delta
hgb product exact_oracle
```

The stricter first stage reduced oracle coverage:

```text
first_stage_oracle_rows=19/151
proposals=5353
```

Best strict80 outcomes:

```text
rfreg product:
best active threshold=0.0005
macro_delta=-0.000074
time_delta=-0.000099
fighter_delta=+0.000000
fp_penalty_delta=+0.000000
replacements=1
oracle_hits=0

rfreg sqrt_product:
best active threshold=0.001
macro_delta=-0.000074
time_delta=-0.000099
fighter_delta=+0.000000
fp_penalty_delta=+0.000000
replacements=1
oracle_hits=0

etreg product:
best threshold=0.00025
macro_delta=+0.000083
time_delta=+0.000059
fighter_delta=+0.000000
fp_penalty_delta=+0.000000
replacements=1
oracle_hits=0

hgb exact_oracle product:
all checked thresholds were no-op
macro_delta=+0.000000
time_delta=+0.000000
fighter_delta=+0.000000
fp_penalty_delta=+0.000000
replacements=0
```

The only positive strict80 row is an `agn_072` one-frame move
`3461 -> 3462` with a target metadata change `head -> body`; it is not an
oracle replacement.

Interpretation: simply narrowing the cluster-first proposal set does not fix
the second-stage ranking. It either becomes no-op or finds at most one tiny
timing/fighter-neutral move. The missing piece is still a better candidate
ranking target/features, not a stricter first-stage cutoff alone.

## Row-Delta Feature Follow-Up

Added symmetric local context features to the second-stage proposal scorer:

- source row local-cluster features;
- candidate-as-row local-cluster features;
- candidate-row minus source-row feature deltas.

The bounded smoke reused the strict top-80 proposal regime:

```text
max_first_stage_rows=80
max_candidates_per_row=40
max_replacements_total=20
max_replacements_per_video=3
target_mode=single_delta
combine_mode=product
```

Artifacts:

```text
data/processed/timing_audits/timing_cluster_first_replacement_strict80_rowdelta_rfreg_product_sweep_20260523.csv
data/processed/timing_audits/timing_cluster_first_replacement_strict80_rowdelta_rfreg_product_selected_20260523.csv
data/processed/timing_audits/timing_cluster_first_replacement_strict80_rowdelta_rfreg_product_rows_20260523.csv
data/processed/timing_audits/timing_cluster_first_replacement_strict80_rowdelta_etreg_product_sweep_20260523.csv
data/processed/timing_audits/timing_cluster_first_replacement_strict80_rowdelta_etreg_product_selected_20260523.csv
data/processed/timing_audits/timing_cluster_first_replacement_strict80_rowdelta_etreg_product_rows_20260523.csv
```

OOF ranking signal changed only modestly:

```text
rfreg row-delta:
candidate_auc=0.575290
candidate_ap=0.226114
combined_auc=0.572170
combined_ap=0.223523

etreg row-delta:
candidate_auc=0.558093
candidate_ap=0.211364
combined_auc=0.552268
combined_ap=0.208204
```

Component sweep summary:

```text
rfreg row-delta best threshold=0.005:
time_delta=+0.000000
fighter_delta=+0.000000
fp_penalty_delta=+0.000000
macro_delta=+0.000000
replacements=0

rfreg first active threshold=0.0005:
time_delta=-0.000363
fighter_delta=-0.000606
fp_penalty_delta=+0.000495
macro_delta=-0.000863
replacements=1
oracle_hits=0

etreg row-delta best threshold=0.005:
time_delta=+0.000000
fighter_delta=+0.000000
fp_penalty_delta=+0.000000
macro_delta=+0.000000
replacements=0

etreg first active threshold=-0.0005:
time_delta=-0.000826
fighter_delta=-0.000888
fp_penalty_delta=+0.000000
macro_delta=-0.000805
replacements=20
oracle_hits=0
```

Conclusion: row-delta features do not fix the second-stage timing replacement
ranker. The only active moves are non-oracle replacements and they hurt
`time_delta`; the small `fp_penalty_delta` gain in one rfreg setting is the
wrong component for this hypothesis. Keep the feature expansion as a
reproducible negative control, but do not build a submission candidate from it.

## YOLO26X Pool Control

Because the YOLO26X cluster-first detector had stronger row-level diagnostics
than YOLO26L in some cuts, the actual replacement evaluator was repeated on the
YOLO26X pool. The first run kept the same second-stage `rfreg` setup and used
the YOLO26X RF cluster-first probabilities:

```bash
.venv/bin/python tools/evaluate_timing_cluster_first_replacements.py \
  --baseline-predictions data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --oracle-audit data/processed/timing_audits/strict_clearens_yolo26x_oracle_replace_keep_audit_20260523.csv \
  --cluster-first-probs data/processed/timing_audits/strict_clearens_yolo26x_cluster_first_rf_20260523.csv \
  --target-mode single_delta \
  --model rfreg \
  --combine-mode product \
  --max-first-stage-rows 160 \
  --max-candidates-per-row 80 \
  --max-replacements-total 40 \
  --max-replacements-per-video 6 \
  --thresholds 0.005,0.002,0.0015,0.001,0.00075,0.0005,0.00025,0,-0.0005,-0.001 \
  --output-sweep data/processed/timing_audits/timing_cluster_first_replacement_yolo26x_rfreg_product_sweep_20260523.csv \
  --output-selected data/processed/timing_audits/timing_cluster_first_replacement_yolo26x_rfreg_product_selected_20260523.csv \
  --output-rows data/processed/timing_audits/timing_cluster_first_replacement_yolo26x_rfreg_product_rows_20260523.csv
```

Setup summary:

```text
first_stage_rows=160
first_stage_oracle_rows=31/144
proposals=19430
positive_delta_proposals=3052
candidate_auc=0.633436
candidate_ap=0.225679
combined_auc=0.619510
combined_ap=0.221840
```

Component sweep summary:

```text
label        macro_delta time_delta fighter_delta fp_penalty_delta repl oracle_hits
baseline     +0.000000   +0.000000  +0.000000     +0.000000        0    0
thr 0.00075  -0.000130   -0.000033  +0.000000     +0.000000        1    0
thr 0.0005   -0.000157   -0.000033  +0.000000     +0.000000        2    0
thr 0.00025  +0.000257   -0.000476  -0.000950     -0.000713        3    0
thr 0        -0.001641   -0.001155  -0.002202     +0.000423        10   0
thr -0.0005  -0.001992   -0.001316  -0.003274     +0.000423        40   0
```

The apparent best row is not a timing improvement. It has
`time_delta=-0.000476` and `fighter_delta=-0.000950`; its
`macro_delta=+0.000257` comes from `fp_penalty_delta=-0.000713`. It also has
`oracle_hits=0`, so the policy is not recovering the intended oracle
replacement set.

The ET cluster-first probabilities were also checked with the same `rfreg`
second stage:

```text
first_stage_rows=160
first_stage_oracle_rows=33/144
proposals=19492
positive_delta_proposals=3213
candidate_auc=0.623077
candidate_ap=0.222503
combined_auc=0.621087
combined_ap=0.223256
```

Component sweep summary:

```text
label        macro_delta time_delta fighter_delta fp_penalty_delta repl oracle_hits
baseline     +0.000000   +0.000000  +0.000000     +0.000000        0    0
thr 0.0005   -0.000130   -0.000033  +0.000000     +0.000000        1    0
thr 0.00025  -0.000412   -0.000560  +0.000000     +0.000000        4    0
thr 0        -0.001632   -0.001027  -0.000054     +0.000885        12   0
thr -0.0005  -0.002369   -0.001907  -0.000996     +0.000885        40   0
```

Conclusion: the YOLO26X pool does not rescue cluster-first timing replacement.
Its detector diagnostics are somewhat better, but active replacements still
miss the oracle rows and move the wrong components. Do not build a submission
candidate from this family until the second-stage target/features are changed
substantially.
