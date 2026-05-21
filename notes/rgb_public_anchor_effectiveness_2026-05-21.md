# RGB Public-Anchor Effectiveness - 2026-05-21

Goal: test whether cached RGB clip features can improve attributes on the
public-safe yolo26l/root anchor without changing timing, counts, fighter, hand,
target, or punch type.

No Kaggle upload.

## Validation Source

Input:

```text
data/processed/validation_rows/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088.csv
baseline=0.365177
```

The evaluator matches these yolo26l rows back to the yolo26x full RGB candidate
pool:

```text
rgb_feature_rows=22691
pred_rows=1136
matched=1113
match_rate=0.9798
```

OOF result:

```text
effectiveness_margin_0    0.371229  delta=+0.006053  wins=11  changed=555
effectiveness_margin_0.1  0.370254  delta=+0.005078  wins=11  changed=427
effectiveness_margin_0.2  0.369373  delta=+0.004196  wins=11  changed=316
effectiveness_margin_0.3  0.368714  delta=+0.003537  wins=12  changed=227
```

The ungated `punch_type + effectiveness` variant is slightly higher
(`0.371615`), but it changes 918 rows and is less public-safe. The saved
candidate focuses on `effectiveness` only.

Saved OOF rows:

```text
data/processed/validation_rows/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088_rgb_effectiveness_margin0_oof.csv
data/processed/validation_rows/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088_rgb_effectiveness_margin02_oof.csv
```

Scorer verification:

```text
margin0  macro_score=0.371229
margin02 macro_score=0.369373
```

## Test Artifacts

Applied to the current public-best root `submission.csv`:

```text
submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_rgb_eff_m0_OFFLINE_CANDIDATE.csv
submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_rgb_eff_m02_OFFLINE_CANDIDATE.csv
```

Both pass `tools/validate_data.py`.

Diff versus root:

```text
margin0:
  changed=371
  changes=effectiveness only
  transitions: landed->blocked 171, landed->miss 200

margin02:
  changed=223
  changes=effectiveness only
  transitions: landed->blocked 119, landed->miss 104
```

Clear counts are identical to root for every test video:

```text
agn_037:52, agn_038:81, agn_039:57, agn_047:112, agn_048:44,
agn_049:62, agn_062:113, agn_063:107, agn_064:75
```

## Decision

Keep as the best current public-safe RGB attribute probe. It is not uploaded
automatically, but unlike sequence/private-risk splices it preserves the
public-proven timing/count/fighter structure and changes only `effectiveness`.
If an explicit public-safe upload is approved, the ungated margin `0` artifact
has the stronger OOF evidence; margin `0.2` is the lower-churn fallback.
