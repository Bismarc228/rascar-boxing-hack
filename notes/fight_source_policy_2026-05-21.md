# Fight-Level Source Policy - 2026-05-21

Scope: table-only source selection over already-generated validation/test row
sources. No Kaggle upload. The goal was to check whether fight/video-level
diagnostics can choose among `base`, `seq`, `hybrid_rival`, `exchange`, and
`motion` better than the best single validation source.

## Tool Added

- `tools/evaluate_fight_source_policy.py`
  - Reads `fight_level_sources_*` tables.
  - Evaluates leave-one-fight source policies:
    - global mean source choice,
    - data-root mean source choice,
    - Ridge regressor,
    - HistGradientBoosting regressor.
  - Prints test source rankings, but only for diagnostics.

## Inputs

Validation table:

```text
data/processed/diagnostics/fight_level_sources_validation_with_exchange.csv
```

Test table with matching source names:

```text
data/processed/diagnostics/fight_level_sources_test_policy_sources.csv
```

Sources:

```text
base, seq, hybrid_rival, exchange, motion
```

## Validation Evidence

Single-source means:

| Source | Mean validation score |
| --- | ---: |
| `base` | `0.365177` |
| `seq` | `0.379682` |
| `hybrid_rival` | `0.389672` |
| `exchange` | `0.390962` |
| `motion` | `0.368244` |

Per-video source oracle:

```text
0.403225
```

Oracle choices:

```text
agn_003:base
agn_004:seq
agn_010:base
agn_023:seq
agn_024:exchange
agn_025:exchange
agn_056:motion
agn_057:motion
agn_058:seq
agn_069:seq
agn_070:hybrid_rival
agn_071:seq
agn_072:seq
```

OOF policy results:

| Policy | OOF score | Decision |
| --- | ---: | --- |
| `mean_global` | `0.376386` | Killed. |
| `mean_root` | `0.389690` | Nearly ties `hybrid_rival`, but below single-source `exchange`. |
| `ridge` | `0.368829` | Killed. |
| `hgb` | `0.357144` | Killed. |

## Decision

- Current fight/source table features are not enough for a learned source
  policy.
- Do not generate or submit a policy-spliced CSV from these models.
- Keep the source oracle as headroom evidence only.
- A future policy needs stronger features, probably from actual clip/RGB
  exchange evidence or per-fight tracklet quality, not only count/overlap gaps.
