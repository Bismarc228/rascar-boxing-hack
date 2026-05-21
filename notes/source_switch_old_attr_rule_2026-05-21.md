# Old-Attribute Source Switch Rule - 2026-05-21

Goal: preserve the current best local source while testing whether a simple
fight-level frame-consistency rule can recover the `old_attr` source oracle
headroom.

No Kaggle upload.

## Sources

Base:

```text
seq_motion=data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_gate_oof.csv
score=0.401483
```

Override:

```text
old_attr=data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_attr_all_oof.csv
score=0.396329
```

## Rule

Use `old_attr` for a video when the override has high local frame agreement
with the base source:

```text
old_attr.source_le15 >= 0.93
```

Selected validation videos:

```text
agn_003, agn_024, agn_025, agn_056, agn_057,
agn_058, agn_069, agn_070, agn_071
```

OOF artifact:

```text
data/processed/validation_rows/seq_motion_old_attr_sourcele15_093_oof.csv
```

Score:

```text
seq_motion baseline: 0.401483
source switch:       0.410780
delta:              +0.009297
```

Per-video reality check: the lift is dominated by `agn_003`.

```text
without agn_003, seq_motion: 0.419714
without agn_003, switch:     0.418878
delta:                      -0.000836
```

The rule also selects losing validation switches (`agn_024`, `agn_025`,
`agn_057`), so it is not reliable enough for an upload by itself.

## Group-OOF Stump Check

Added:

```text
tools/evaluate_source_decision_stumps.py
```

This evaluator chooses the best single-feature source-switch rule on all but
one fight group, then applies it to the held-out group. With the full source
table and `seq_motion` as base:

```text
base=0.401483
best in-sample stump=old_attr:source_le15>=0.9394
best in-sample score=0.411775
group_oof=0.399638
group_oof_delta=-0.001845
```

The fold that holds out `agn_003` chooses a rule that does not switch `agn_003`,
so the apparent large in-sample win disappears under honest group validation.

## Test Analog

Diagnostic table:

```text
data/processed/diagnostics/source_switch_old_attr_test_20260521.csv
```

Direct private-risk analog:

```text
base=submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_gate_OFFLINE_CANDIDATE.csv
override=submissions/hybrid_yolo26l_best_seq_tcn_snap4_gate_oof395_rival_exchange_attr_all_OFFLINE_CANDIDATE.csv
selected=agn_038,agn_062,agn_063
output=submissions/seq_motion_old_attr_sourcele15_093_OFFLINE_CANDIDATE.csv
```

Validation:

```text
python3 tools/validate_data.py --submission submissions/seq_motion_old_attr_sourcele15_093_OFFLINE_CANDIDATE.csv
Validation passed.
```

Clear-row counts:

```text
root:       703
seq_motion: 743
old_attr:   731
switch:     734
```

The direct analog touches `agn_038`, which is public-sensitive and already has
better public evidence in the root anchor. The public-anchor override version
of the same threshold selects only `agn_038`.

## Decision

Save as a useful source-policy/ensemble hypothesis, not as an automatic submit.
It has a strong validation number, but the evidence is too concentrated in one
validation video and the test analog changes public-sensitive `agn_038`.
