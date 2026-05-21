# Fixed-Row Attribute Model - 2026-05-21

Scope: keep selected rows, timing, fighter, and counts fixed; learn only event
attributes from pose/context features. No Kaggle upload.

## Tools Added

- `tools/evaluate_fixed_row_attribute_model.py`
  - OOF by fight group.
  - Trains classifiers for `punch_type`, `effectiveness`, `hand`, and `target`
    using matched/scorable fixed rows.
  - Can write OOF rows.
- `tools/make_fixed_row_attribute_submission.py`
  - Trains on validation rows and applies the attribute model to an existing
    test submission CSV.

## Validation Source

Input:

```text
data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_hgb_p024_oof.csv
```

Baseline:

```text
score=0.390962
punch_type=0.177104
effectiveness=0.189015
hand=0.497780
target=0.445035
```

OOF results:

| Variant | Score | Delta | Notes |
| --- | ---: | ---: | --- |
| `punch_type` | `0.391825` | `+0.000862` | Small positive. |
| `effectiveness` | `0.395018` | `+0.004056` | Main gain. |
| `ptype_eff` | `0.395881` | `+0.004918` | Good. |
| `hand_target` | `0.391411` | `+0.000449` | Small positive despite hand drop. |
| `all_attrs` | `0.396329` | `+0.005367` | Best. |

OOF rows:

```text
data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_attr_all_oof.csv
data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_attr_effectiveness_oof.csv
```

Source ensemble check with base/seq/exchange/attr/motion:

- `attr`: `0.396329`.
- Pairwise base+attr oracle: `0.399425`.
- Full source oracle: `0.407632`.
- `attr` uniquely wins `agn_003`, `agn_010`, `agn_023`, `agn_024`,
  `agn_025`, `agn_057`, `agn_070`, and `agn_071`.

## Test Artifacts

Current public-anchor shape:

```bash
python3 tools/make_fixed_row_attribute_submission.py \
  --train-predictions data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_hgb_p024_oof.csv \
  --train-tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --input submission.csv \
  --test-tracks-dir data/processed/pose_tracks/test_yolo26x_conf035 \
  --output submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_attr_all_OFFLINE_CANDIDATE.csv \
  --variant all_attrs \
  --label-window 12
```

Result:

```text
changed=498
Validation passed.
```

Conservative effectiveness-only public-anchor artifact:

```text
submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_attr_effectiveness_OFFLINE_CANDIDATE.csv
```

Result:

```text
changed=191
Validation passed.
```

Private-risk gated shape:

```text
submissions/hybrid_yolo26l_best_seq_tcn_snap4_gate_oof395_rival_exchange_attr_all_OFFLINE_CANDIDATE.csv
```

Result:

```text
changed=525
Validation passed.
```

Conservative effectiveness-only private-risk artifact:

```text
submissions/hybrid_yolo26l_best_seq_tcn_snap4_gate_oof395_rival_exchange_attr_effectiveness_OFFLINE_CANDIDATE.csv
```

Result:

```text
changed=214
Validation passed.
```

## Decision

- Keep as a real ensemble/submission candidate branch.
- Keep both variants:
  - `effectiveness` is the safer version (`+0.004056` OOF, fewer changes).
  - `all_attrs` is the stronger local version (`+0.005367` OOF).
- Do not upload automatically: this is attribute-only and should be weighed
  against the current public best and submit budget.
- The branch is independent from timing/fighter work and should remain saved
  for future ensemble/splice experiments.

## Current Best Re-Audit

After the later audio gate, exchange-side identity, and tracklet-appearance
micro-source, the current best fixed rows are:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_appearance_oof.csv
baseline=0.405200
```

Re-running the fixed-row attribute evaluator on these rows:

```text
.venv/bin/python tools/evaluate_fixed_row_attribute_model.py \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_appearance_oof.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --label-window 12
```

Result:

```text
variant        score     delta      n_changed
punch_type     0.404153  -0.001047  351
effectiveness  0.404679  -0.000521  123
ptype_eff      0.403632  -0.001568  439
hand_target    0.405045  -0.000155  106
all_attrs      0.403477  -0.001723  503
```

Decision update: the attribute model remains useful as an older independent
branch and public-anchor candidate, but it should not be reapplied on top of the
current `audio_tracklet` best source without new features or labels.
