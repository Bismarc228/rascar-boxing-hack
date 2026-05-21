# Current Source Policy With Tracklet And RGB - 2026-05-21

Goal: retest source-oracle and automatic fight-level source selection after the
latest saved row sources:

- `audio_tracklet`: then-best local OOF source at `0.405200`.
- `rgb_full`: full pool-1800 RGB bridge A/B rows at `0.391237`.
- older sequence, attribute, yolo26l, exchange, and RGB cap-400 sources.

No Kaggle upload and no test CSV generation.

## Row Source Oracle

Command:

```text
.venv/bin/python tools/evaluate_row_source_ensemble.py \
  --source audio_tracklet=...audio_exchange_tracklet_appearance_oof.csv \
  --source audio_exchange=...audio_gate_exchange_side_oof.csv \
  --source audio_gate=...audio_gate_oof.csv \
  --source seq_motion=...motion_gate_oof.csv \
  --source old_attr=...gated_rival_exchange_attr_all_oof.csv \
  --source yolo26l=...yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088.csv \
  --source seq_attr=...exchange_attr_all_oof.csv \
  --source seq_exchange=...repeat_exchange_hgb_p024_oof.csv \
  --source exchange_side=...motion_exchange_side_logreg_oof.csv \
  --source rgb_full=...seq_rgb_contact_bridge_pool1800_ab_oof.csv \
  --source rgb_cap400=...seq_rgb_contact_bridge_cap400_medium_oof.csv
```

Single-source scores:

```text
audio_tracklet  0.405200
audio_exchange  0.404915
audio_gate      0.404479
exchange_side   0.401720
seq_motion      0.401483
seq_attr        0.400288
old_attr        0.396329
seq_exchange    0.393593
rgb_full        0.391237
rgb_cap400      0.388112
yolo26l         0.365177
```

Full oracle:

```text
source_oracle=0.429550
delta_vs_audio_tracklet=+0.024350
choices=agn_003:old_attr,agn_004:exchange_side,agn_023:rgb_full,agn_024:seq_motion,agn_056:yolo26l,agn_058:seq_exchange,agn_069:rgb_cap400,agn_070:rgb_full,agn_071:rgb_full,agn_072:rgb_full
```

Best pairwise oracles versus `audio_tracklet`:

```text
old_attr    0.416973  +0.011773
rgb_full    0.415440  +0.010240
yolo26l     0.412586  +0.007386
rgb_cap400  0.410019  +0.004819
seq_attr    0.409649  +0.004449
seq_motion  0.409500  +0.004299
```

The headroom is large, and `rgb_full` is a strong per-video oracle source even
though it is a weak global row source.

## Fight-Level Table

Generated:

```text
data/processed/diagnostics/source_table_current_audio_tracklet_rgbfull_20260521.csv
```

Table summary confirmed `audio_tracklet=0.405200` as the best mean source at
that stage.
Several weak global sources still win individual videos against the base, which
explains the oracle headroom.

## Stumps

Command:

```text
.venv/bin/python tools/evaluate_source_decision_stumps.py \
  --validation-table data/processed/diagnostics/source_table_current_audio_tracklet_rgbfull_20260521.csv \
  --base-source audio_tracklet \
  --sources audio_exchange,audio_gate,seq_motion,old_attr,yolo26l,seq_attr,seq_exchange,exchange_side,rgb_full,rgb_cap400
```

Best in-sample rules:

```text
old_attr:n_pred<=33          score=0.411640  switches=1
rgb_full:base_le15>=0.9592   score=0.410882  switches=6
rgb_full:n_pred>=122         score=0.409761  switches=3
```

Group OOF result:

```text
OOF_SELECTED score=0.396653
base=0.405200
delta=-0.008547
switches=7 wins=2 losses=5
```

The attractive source-switch rules are not robust under leave-one-fight
selection.

## Learned Policies

Command:

```text
.venv/bin/python tools/evaluate_fight_source_policy.py \
  --validation-table data/processed/diagnostics/source_table_current_audio_tracklet_rgbfull_20260521.csv \
  --sources audio_tracklet,audio_exchange,audio_gate,seq_motion,old_attr,yolo26l,seq_attr,seq_exchange,exchange_side,rgb_full,rgb_cap400
```

Results:

```text
source_oracle  0.429550
mean_global    0.401423
mean_root      0.376651
ridge          0.402975
hgb            0.403028
base           0.405200
```

Decision: current automatic source selection with the latest tracklet/audio/RGB
sources is killed as a submit path. The oracle headroom is real, but the present
fight-level features do not recover it out of sample. Keep `rgb_full` and
`old_attr` as diagnostic sources only; any future source selector needs new
out-of-sample features, not another threshold sweep on this table.

## RGB Effectiveness Follow-Up

After the RGB fixed-row effectiveness witness became the best single source:

```text
rgb_eff=data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_rgb_effectiveness_oof.csv
score=0.406345
```

Re-running the same source audit with `rgb_eff` as the base gives:

```text
source_oracle=0.429715
delta_vs_rgb_eff=+0.023370
```

Best pairwise oracles:

```text
old_attr    0.417202  +0.010856
rgb_full    0.416243  +0.009898
yolo26l     0.413454  +0.007108
rgb_cap400  0.410862  +0.004516
seq_motion  0.410255  +0.003909
```

Generated updated diagnostic table:

```text
data/processed/diagnostics/source_table_current_rgb_eff_20260521.csv
```

Selectors still fail:

```text
stumps_oof  0.396887
mean_global 0.406345
ridge       0.403028
hgb         0.403410
base        0.406345
```

Decision update: the new RGB effectiveness source raises the base and leaves
oracle headroom, but current fight-level source selection remains below the best
single source. Do not create a source-policy test CSV or upload.

## RGB Effectiveness Margin Follow-Up

The confidence-gated effectiveness source is now the best single source:

```text
rgb_eff_m02=data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_rgb_effectiveness_margin02_oof.csv
score=0.407232
```

Re-running the source audit with `rgb_eff_m02` as base:

```text
source_oracle=0.430014
delta_vs_rgb_eff_m02=+0.022782
```

Best pairwise oracles:

```text
old_attr    0.418053  +0.010821
rgb_full    0.416978  +0.009746
yolo26l     0.414046  +0.006815
rgb_cap400  0.411821  +0.004590
seq_attr    0.410684  +0.003453
```

Generated updated diagnostic table:

```text
data/processed/diagnostics/source_table_current_rgb_eff_margin02_20260521.csv
```

Selectors still fail:

```text
stumps_oof  0.397342
mean_global 0.407232
ridge       0.403667
hgb         0.404206
base        0.407232
```

Decision update: source-policy remains diagnostic only. The best learned policy
is still below simply using `rgb_eff_m02`.
