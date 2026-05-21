# Row-Source Stacker - 2026-05-21

Goal: test whether the saved independent row sources can be combined at row
level instead of selecting a whole source per video. No Kaggle upload.

## Tool

Added:

```text
tools/evaluate_row_source_stacker.py
```

The tool:

- reads multiple saved validation row sources;
- builds a union of clear rows per video;
- trains a fight-group OOF `HistGradientBoostingRegressor` to score candidate
  rows from local source agreement, frame proximity, attributes, source counts,
  and source identity;
- selects top rows under the chosen count source with fighter/hand NMS;
- can write the best OOF row artifact for scoring diagnostics.

## Inputs

```text
base      data/processed/validation_rows/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088.csv
seq       data/processed/validation_rows/seq_tcn_snap4_rootcount088_oof.csv
rival     data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_w0_samehand_r13_min1_fht.csv
exchange  data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_hgb_p024_oof.csv
attr_all  data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_attr_all_oof.csv
attr_eff  data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_attr_effectiveness_oof.csv
motion    data/processed/validation_rows/yolo26l_cropmotion_samesum_w4_am02_thr085_same10_cross4_rootrate088_ma-004_mb008.csv
```

## Result

Best row-level stacker:

```text
score=0.368373
time=0.490033
fp_penalty=0.063133
n_pred=1185
count_source=attr_all
nms=8
cross_nms=4
```

Artifact:

```text
data/processed/validation_rows/row_source_stacker_attr_anchor_oof.csv
```

This is far below the fixed-row attribute source:

```text
attr_all=0.396329
attr_eff=0.395018
exchange=0.390962
```

The row-level target is too noisy: nearly all union rows are already near real
events, and the model cannot reliably separate subtle timing/attribute/fighter
errors from useful alternate rows with the current source-agreement features.

## Source Policy With Attributes

Adding `attr_all` and `attr_eff` to the fight-level table increases oracle
headroom but does not create a learned policy:

```text
source_oracle=0.408357
mean_global=0.396329  # simply choose attr_all
mean_root=0.391194
ridge=0.372111
hgb=0.366116
```

Per-video oracle choices:

```text
agn_003 attr_all
agn_004 seq
agn_010 attr_all
agn_023 attr_all
agn_024 attr_all
agn_025 attr_all
agn_056 motion
agn_057 attr_all
agn_058 attr_eff
agn_069 attr_eff
agn_070 attr_all
agn_071 attr_eff
agn_072 seq
```

## Decision

- Do not use row-level source stacking as a submit path.
- Keep `tools/evaluate_row_source_stacker.py` as a diagnostic for future, richer
  features, but current source-agreement features are killed.
- The best simple source policy remains `attr_all` directly; no learned policy
  beats it.
