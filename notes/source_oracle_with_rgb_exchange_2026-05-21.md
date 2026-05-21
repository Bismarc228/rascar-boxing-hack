# Source Oracle With RGB Bridge And Exchange Side - 2026-05-21

Goal: check whether the newly saved weak sources provide useful per-video
headroom when combined with the current best local source.

No Kaggle upload.

## Sources

Base:

```text
seq_motion=data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_gate_oof.csv
score=0.401483
```

Additional sources:

```text
exchange_side=data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_exchange_side_logreg_oof.csv
seq_bridge_cap400=data/processed/validation_rows/seq_rgb_contact_bridge_cap400_medium_oof.csv
seq_attr=data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_all_oof.csv
seq_exchange=data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_hgb_p024_oof.csv
old_attr=data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_attr_all_oof.csv
yolo26l=data/processed/validation_rows/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088.csv
```

## Row Source Oracle

Command:

```text
python3 tools/evaluate_row_source_ensemble.py \
  --source seq_motion=... \
  --source exchange_side=... \
  --source seq_bridge_cap400=... \
  --source seq_attr=... \
  --source seq_exchange=... \
  --source old_attr=... \
  --source yolo26l=...
```

Result:

```text
source_oracle=0.420937
delta_vs_base=+0.019454
```

Pairwise oracles versus `seq_motion`:

```text
old_attr          0.414829  +0.013346
yolo26l           0.414100  +0.012617
seq_bridge_cap400 0.407967  +0.006484
seq_exchange      0.402352  +0.000869
seq_attr          0.402310  +0.000827
exchange_side     0.402204  +0.000720
```

New-source wins in the full oracle:

```text
exchange_side: agn_004, agn_025
seq_bridge_cap400: agn_069, agn_072
```

The RGB bridge source is poor as a global row source but is locally strong on a
small number of validation videos.

## Fight-Level Policy Retest

Diagnostic table:

```text
data/processed/diagnostics/source_table_with_rgb_exchange_20260521.csv
```

Policy results:

```text
source_oracle=0.420862
mean_global=0.395386
mean_root=0.386354
ridge=0.393753
hgb=0.389049
```

All current source-selection policies remain below simply using the best single
source (`exchange_side=0.401720`, `seq_motion=0.401483`). The oracle headroom is
real, but the present fight-level features do not recover it.

## Decision

Keep `exchange_side` and `seq_bridge_cap400` as diagnostic/ensemble sources,
not as submit candidates. Do not spend uploads or test CSV generation on the
current automatic source-policy stack.

## Audio Gate Follow-Up

The later fixed-row audio gate adds a stronger single source:

```text
audio_gate=data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_gate_oof.csv
score=0.404479
```

With `audio_gate` included, the source oracle rises to:

```text
source_oracle=0.422619
delta_vs_seq_motion=+0.021136
```

Audio-gate uniquely wins `agn_010`, `agn_025`, `agn_057`, and `agn_072` in the
full oracle, but the current fight-level policy stack still underperforms the
best single source (`mean_global=0.400837`, `ridge=0.394631`).
