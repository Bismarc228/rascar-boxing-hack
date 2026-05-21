# RGB Stacked Public-Anchor Attributes - 2026-05-21

Goal: test whether cached RGB clip features add effectiveness signal on top of
the existing fixed-row public-anchor attribute models.

No Kaggle upload.

## Validation Sources

RGB model:

```text
cache=data/processed/rgb_features/seq_bridge_contact_vitb16_union_clip4_stride2_pool1800_ab.npz
tracks=data/processed/pose_tracks/val_yolo26x_conf035
matched=1180/1185
```

### Attribute-All Base

Input:

```text
data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_attr_all_oof.csv
baseline=0.396329
```

OOF results:

```text
effectiveness_margin_0    0.397569  delta=+0.001240  changed=479
effectiveness_margin_0.1  0.397904  delta=+0.001575  changed=366
effectiveness_margin_0.2  0.397992  delta=+0.001663  changed=272
effectiveness_margin_0.3  0.398315  delta=+0.001986  changed=198
```

Saved OOF rows:

```text
data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_attr_all_rgb_effectiveness_margin03_oof.csv
```

Scorer verification:

```text
macro_score=0.398315
```

### Effectiveness-Only Base

Input:

```text
data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_attr_effectiveness_oof.csv
baseline=0.395018
```

OOF results:

```text
effectiveness_margin_0    0.397369  delta=+0.002351  changed=482
effectiveness_margin_0.1  0.396548  delta=+0.001530  changed=369
effectiveness_margin_0.2  0.397316  delta=+0.002297  changed=263
effectiveness_margin_0.3  0.396928  delta=+0.001910  changed=177
```

Saved OOF rows:

```text
data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_attr_effectiveness_rgb_effectiveness_margin0_oof.csv
```

Scorer verification:

```text
macro_score=0.397369
```

## Test Artifacts

Both artifacts pass `tools/validate_data.py` and keep root clear counts exactly
unchanged:

```text
agn_037:52, agn_038:81, agn_039:57, agn_047:112, agn_048:44,
agn_049:62, agn_062:113, agn_063:107, agn_064:75
```

Strongest local attribute stack:

```text
submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_attr_all_rgb_eff_m03_OFFLINE_CANDIDATE.csv
```

Generation summary:

```text
train_rows=1185 trainable=1022 test_clear_rows=703 matched_test=679
column=effectiveness margin=0.3 changed=130
```

Diff versus root:

```text
changed_ids=528
columns=effectiveness:242, hand:24, punch_type:326, target:157
effectiveness transitions: landed->blocked:89, landed->miss:153
```

Cleaner effectiveness-only public-anchor stack:

```text
submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_attr_effectiveness_rgb_eff_m0_OFFLINE_CANDIDATE.csv
```

Generation summary:

```text
train_rows=1185 trainable=1022 test_clear_rows=703 matched_test=679
column=effectiveness margin=0 changed=295
```

Diff versus root:

```text
changed_ids=345
columns=effectiveness:345
effectiveness transitions: landed->blocked:146, landed->miss:199
```

Public-sensitive `agn_038`-only splice for a lower-blast-radius public probe:

```text
submissions/hybrid_root_attr_effectiveness_rgb_eff_m0_agn038_only_OFFLINE_CANDIDATE.csv
```

Gate/validation summary:

```text
replace_keys=agn_038
base_count=81
override_count=81
Validation passed.
```

Diff versus root:

```text
changed_ids=47
columns=effectiveness:47
videos=agn_038:47
effectiveness transitions: landed->blocked:38, landed->miss:9
```

Stronger but less conservative `agn_038`-only `attr_all` splice:

```text
submissions/hybrid_root_attr_all_rgb_eff_m03_agn038_only_OFFLINE_CANDIDATE.csv
```

Gate/validation summary:

```text
replace_keys=agn_038
base_count=81
override_count=81
Validation passed.
```

Diff versus root:

```text
changed_ids=73
columns=effectiveness:38, hand:2, punch_type:45, target:32
videos=agn_038:73
effectiveness transitions: landed->blocked:30, landed->miss:8
```

## Decision

This is a real stacked public-anchor attribute branch:

- `attr_all + RGB effectiveness m0.3` is the strongest local public-anchor
  attribute artifact at `0.398315`, but it inherits the broader
  punch-type/hand/target churn of the all-attribute model.
- `attr_effectiveness + RGB effectiveness m0` is the cleanest public-safe
  variant: it stays fixed-row and changes only `effectiveness` versus root while
  improving the effectiveness-only OOF source from `0.395018` to `0.397369`.

No automatic upload. Under the current submit guardrail, the effectiveness-only
stack replaces the earlier raw RGB effectiveness probe as the best fixed-row
public-safe candidate if an upload is explicitly approved. The `agn_038`-only
splice is the narrower public-probe version because previous submissions showed
`agn_038` is public-sensitive. The `attr_all` `agn_038`-only splice is bounded
to the same video but changes punch type/hand/target too, so it is the stronger
but less conservative public probe.
