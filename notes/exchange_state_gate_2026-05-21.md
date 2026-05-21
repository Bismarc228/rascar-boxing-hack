# Exchange-State Gate - 2026-05-21

Scope: fixed selected rows only. No Kaggle upload. The hypothesis is that some
selected pose/sequence events are low-exchange or no-punch rows, and can be
dropped with a fight-level OOF keep/drop model.

## Tools Added

- `tools/evaluate_exchange_state_gate.py`
  - Builds row-level pose/exchange features from the raw candidate pool around
    each selected prediction row.
  - Labels rows as `tp_scorable`, `tp_time_only`, or `fp` using local GT.
  - Trains an OOF keep/drop model by fight group and evaluates `clear=false`
    drops across probability thresholds.
  - Can write OOF validation rows with `--write-oof-rows`.
- `tools/make_exchange_state_gate_submission.py`
  - Trains the same gate on validation rows and applies it to a test submission,
    turning low-keep rows into `clear=false`.

## Validation Results

### yolo26l public-base rows

Command:

```bash
python3 tools/evaluate_exchange_state_gate.py \
  --predictions data/processed/validation_rows/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26l_conf035 \
  --model hgb
```

Result:

- Baseline: `0.365177`.
- Best gate: `0.370306` at threshold `0.40`, dropping 14 rows.
- Main gain is FP reduction with only small timing/fighter movement.

### sequence snap4 OOF rows

- Baseline: `0.379682`.
- Best gate: `0.383814` at threshold `0.40`, dropping 67 rows.
- This helps the high-FP sequence source, but it also reduces time/fighter
  components; use only as a gated source, not a standalone replacement.

### gated hybrid rows

- Baseline: `0.389125`.
- No positive threshold. The plain gated hybrid is already too conservative for
  this drop gate.

### gated hybrid + rival rows

Best current identity/timing branch before exchange gate:
`data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_w0_samehand_r13_min1_fht.csv`.

Command:

```bash
python3 tools/evaluate_exchange_state_gate.py \
  --predictions data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_w0_samehand_r13_min1_fht.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --model hgb \
  --thresholds 0.20,0.22,0.24,0.25,0.26 \
  --write-threshold 0.24 \
  --write-oof-rows data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_hgb_p024_oof.csv
```

Result:

- Baseline: `0.389672`.
- Best gate: `0.390962` at threshold `0.24` or `0.25`, dropping 5 rows.
- OOF output rows:
  `data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_hgb_p024_oof.csv`.

## Source Ensemble Evidence

Source ensemble across base/sequence/hybrid-rival/exchange/motion:

```bash
python3 tools/evaluate_row_source_ensemble.py \
  --source base=data/processed/validation_rows/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088.csv \
  --source seq=data/processed/validation_rows/seq_tcn_snap4_rootcount088_oof.csv \
  --source hybrid_rival=data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_w0_samehand_r13_min1_fht.csv \
  --source exchange=data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_hgb_p024_oof.csv \
  --source motion=data/processed/validation_rows/yolo26l_cropmotion_samesum_w4_am02_thr085_same10_cross4_rootrate088_ma-004_mb008.csv
```

Result:

- `exchange`: `0.390962`.
- Pairwise base+exchange oracle: `0.395149`.
- Full source oracle: `0.403225`.
- Exchange uniquely wins `agn_024` and `agn_025`; hybrid-rival still wins
  `agn_070`; motion still wins `agn_056/057`.

## Test Artifact

Generated from the public-anchor rival offline candidate:

```bash
python3 tools/make_exchange_state_gate_submission.py \
  --train-predictions data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_w0_samehand_r13_min1_fht.csv \
  --train-tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --input submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_rival_w0_samehand_r13_min1_fht_OFFLINE_CANDIDATE.csv \
  --test-tracks-dir data/processed/pose_tracks/test_yolo26x_conf035 \
  --output submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_rival_exchange_hgb_p024_OFFLINE_CANDIDATE.csv \
  --model hgb \
  --threshold 0.24
```

Validation:

```text
python3 tools/validate_data.py --data-root data/raw \
  --submission submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_rival_exchange_hgb_p024_OFFLINE_CANDIDATE.csv
Validation passed.
```

The train-all test gate dropped only one clear row:

- `agn_062`, id `614`, frame `2716`, red left head.

Also generated the OOF-analog private-risk branch by applying rival+exchange to
the gated test candidate (`agn_038,agn_062,agn_063` sequence replacement):

```bash
python3 tools/make_fighter_rival_flip_submission.py \
  --input submissions/hybrid_yolo26l_best_seq_tcn_snap4_gate_oof395_test_OFFLINE_CANDIDATE.csv \
  --output submissions/hybrid_yolo26l_best_seq_tcn_snap4_gate_oof395_rival_w0_samehand_r13_min1_fht_OFFLINE_CANDIDATE.csv \
  --tracks-dir data/processed/pose_tracks/test_yolo26x_conf035 \
  --window 0 \
  --match-mode same_hand \
  --ratio 1.3 \
  --min-rival-score 1.0 \
  --update-mode fighter_hand_target
```

This changes 11 fighter/target fields and keeps 731 clear rows. Applying the
exchange gate to that gated+rival file at threshold `0.24` drops zero rows:

```text
submissions/hybrid_yolo26l_best_seq_tcn_snap4_gate_oof395_rival_exchange_hgb_p024_OFFLINE_CANDIDATE.csv
Validation passed.
```

So the exchange test effect is source-dependent: it is useful OOF, but the
train-all threshold is nearly inactive on the private-risk gated test source.

## Decision

- Keep as an ensemble candidate and source-gating signal.
- Do not upload directly: OOF gain is real but modest, and the test artifacts
  are extremely conservative at the validated threshold.
- Next useful step is a fight-level source policy over base/seq/motion/rival/
  exchange, not another blind threshold sweep.
