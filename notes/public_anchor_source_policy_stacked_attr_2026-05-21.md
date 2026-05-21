# Public-Anchor Source Policy With Stacked RGB Attributes - 2026-05-21

Goal: check whether the stacked public-anchor RGB attribute sources can be
turned into a testable per-video source policy, rather than only full
attribute-only artifacts.

No Kaggle upload. No new submission artifact was generated.

## Tables

Validation table:

```text
data/processed/diagnostics/source_table_public_anchor_stacked_attr_rgb_val_20260521.csv
```

Test table:

```text
data/processed/diagnostics/source_table_public_anchor_stacked_attr_rgb_test_20260521.csv
```

Validation sources:

```text
root            0.365177
public_rgb_eff  0.371229
attr_eff_rgb    0.397369
attr_all_rgb    0.398315
```

## Oracle

```text
source_oracle=0.404058
choices=agn_003:attr_all_rgb,
        agn_004:attr_eff_rgb,
        agn_010:attr_all_rgb,
        agn_023:attr_all_rgb,
        agn_024:attr_all_rgb,
        agn_025:attr_all_rgb,
        agn_056:public_rgb_eff,
        agn_057:attr_all_rgb,
        agn_058:attr_eff_rgb,
        agn_069:attr_eff_rgb,
        agn_070:attr_all_rgb,
        agn_071:attr_eff_rgb,
        agn_072:attr_eff_rgb
```

Pairwise oracle:

```text
attr_all_rgb    0.401531
attr_eff_rgb    0.400865
public_rgb_eff  0.371523
```

There is real hindsight headroom, but it is not converted by the available
policies.

## Policies

Fight-level source policy:

```text
mean_global  0.398315  # choose attr_all_rgb everywhere
mean_root    0.397507
ridge        0.398149
hgb          0.379230
```

Simple stumps:

```text
best in-sample: attr_all_rgb everywhere, 0.398315
group_oof: 0.377069
```

Test rankings:

```text
ridge choices: attr_all_rgb for every test video
hgb choices: public_rgb_eff for every test video
stump full-data rule: attr_all_rgb for every test video
```

## Decision

No new source-policy artifact. The best learned/testable policy collapses to an
already generated artifact:

```text
submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_attr_all_rgb_eff_m03_OFFLINE_CANDIDATE.csv
```

Keep the oracle numbers as diagnostics, but do not generate or submit a
per-video public-anchor source-policy CSV from this check.
