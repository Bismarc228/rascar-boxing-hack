# Release RC2 E2E - Old-Attribute Source Switch

Branch: `release/old-attribute-source-switch`

This branch materializes the `RC2: Old-Attribute Source Switch` candidate from
`notes/release_candidates_2026-05-23.md`.

## Outputs

Validation OOF rows:

```text
data/processed/validation_rows/seq_motion_old_attr_sourcele15_093_oof.csv
```

Test analog:

```text
submissions/seq_motion_old_attr_sourcele15_093_OFFLINE_CANDIDATE.csv
```

Expected SHA-256:

```text
validation: 1123c112665e351ea0249814a19b6d4a01a29d65273a7482fa0d0981515bc731
test:       a34c7eb6bbe11f1731a586ec8869a413a2ab1dd04c4648efeedb61847139102e
```

## Rebuild

```bash
.venv/bin/python tools/release_build_old_attribute_source_switch.py
```

The runner performs the full release path:

1. Rebuilds the validation fight-level source table for `seq_motion` and
   `old_attr`.
2. Runs the stump audit and verifies the documented in-sample best rule
   `old_attr:source_le15>=0.9394` and group-OOF score `0.399638`.
3. Applies the release rule `old_attr.source_le15 >= 0.93`.
4. Materializes validation rows and verifies `macro_score=0.410780`,
   `n_pred=1263`, `n_tp=1144`, `n_fp=119`.
5. Rebuilds the test source table and applies the same threshold.
6. Materializes `submissions/seq_motion_old_attr_sourcele15_093_OFFLINE_CANDIDATE.csv`.
7. Runs `tools/validate_data.py` and verifies both output SHA-256 hashes.

## Selected Videos

Validation videos selected by the rule:

```text
agn_003,agn_024,agn_025,agn_056,agn_057,agn_058,agn_069,agn_070,agn_071
```

Test videos selected by the direct analog:

```text
agn_038,agn_062,agn_063
```

## Required Local Caches

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_gate_oof.csv
data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_attr_all_oof.csv
submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_gate_OFFLINE_CANDIDATE.csv
submissions/hybrid_yolo26l_best_seq_tcn_snap4_gate_oof395_rival_exchange_attr_all_OFFLINE_CANDIDATE.csv
```
