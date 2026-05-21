# Source Oracle With Stacked RGB Attributes - 2026-05-21

Goal: retest source-oracle and fight-level source-policy headroom after adding
the stacked public-anchor RGB attribute OOF rows.

No Kaggle upload. No test artifact from this branch.

## Sources

Base:

```text
rgb_eff_m02=data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_rgb_effectiveness_margin02_oof.csv
score=0.407232
```

New sources:

```text
stacked_attr_all_rgb=data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_attr_all_rgb_effectiveness_margin03_oof.csv
score=0.398315

stacked_attr_eff_rgb=data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_attr_effectiveness_rgb_effectiveness_margin0_oof.csv
score=0.397369
```

Diagnostic table:

```text
data/processed/diagnostics/source_table_stacked_attr_rgb_20260521.csv
```

## Oracle

Focused source oracle:

```text
source_oracle=0.422839
delta_vs_base=+0.015608
choices=agn_003:stacked_attr_all_rgb,
        agn_023:stacked_attr_all_rgb,
        agn_056:public_rgb_eff,
        agn_058:stacked_attr_eff_rgb,
        agn_069:stacked_attr_eff_rgb,
        agn_070:stacked_attr_all_rgb,
        agn_071:stacked_attr_eff_rgb
```

Pairwise oracle versus `rgb_eff_m02`:

```text
stacked_attr_all_rgb  0.419845  delta=+0.012613
stacked_attr_eff_rgb  0.419842  delta=+0.012610
old_attr              0.418053  delta=+0.010821
public_rgb_eff        0.414680  delta=+0.007448
```

The stacked RGB attribute sources improve the old-attribute oracle headroom, but
the winning videos are still fight/video-specific.

## Source Policies

Simple stumps:

```text
best in-sample rule: stacked_attr_all_rgb:base_med_gap<=1
in_sample_score=0.413770
group_oof_score=0.392902
base_score=0.407232
```

The stump family overfits and is not usable.

Regressor/source policy:

```text
mean_global=0.407232
mean_root=0.391777
ridge=0.407417
hgb=0.398873
```

The ridge policy is a tiny OOF improvement. Materialized rows:

```text
data/processed/validation_rows/source_policy_stacked_attr_rgb_ridge_preserve_order_oof.csv
macro_score=0.407417
```

The materialization preserves each selected source's original per-video row
order. Sorting rows within a video changes some tie matching and scores only
`0.407281`, so preserve-order is the canonical diagnostic artifact.

Ridge choices:

```text
agn_003:rgb_eff_m02
agn_004:rgb_eff_m02
agn_010:rgb_eff_m02
agn_023:stacked_attr_all_rgb
agn_024:stacked_attr_all_rgb
agn_025:stacked_attr_all_rgb
agn_056:stacked_attr_all_rgb
agn_057:stacked_attr_all_rgb
agn_058:stacked_attr_all_rgb
agn_069:stacked_attr_all_rgb
agn_070:stacked_attr_all_rgb
agn_071:stacked_attr_all_rgb
agn_072:rgb_eff_m02
```

## Decision

Keep this only as source-policy bookkeeping:

- It is the best materialized local OOF score so far (`0.407417`), but the lift
  over the best single source is only `+0.000185`.
- The usable stump family regresses badly OOF.
- There is no clean matching test analog because the base `rgb_eff_m02` depends
  on a local-only tracklet source, and row-order tie behavior matters.

Do not generate a Kaggle submission from this policy. The best single/testable
source remains the RGB effectiveness branch already recorded separately.
