# Fixed-Row Target Attribute Smoke, 2026-05-22

Goal: test whether the current root-out fixed-row source still has useful
hand/target attribute headroom after the noisy-label audits. No GPU was used,
no video decode was run, and no Kaggle upload was made.

## Code Change

Fixed a typo in the pose attribute features:

```text
row["effectiveness"] == "missed" -> row["effectiveness"] == "miss"
```

The data uses `miss`, so the old feature never activated. This affects
`tools/evaluate_fixed_row_attribute_model.py` and the submission generator that
imports its feature builder.

Added separate `hand` and `target` variants to:

```text
tools/evaluate_fixed_row_attribute_model.py
tools/make_fixed_row_attribute_submission.py
```

This avoids forcing `hand_target` together when only one side transfers.

## OOF Source

```text
data/processed/root_out/yolo11s_pose_geom_id_attr_model_pose_logreg_noctx_c100_joint_margin_20260521/sources/ptype_eff_margin_p0p1_e0p05/all_val_rows.csv
```

Baseline:

```text
macro=0.304879
punch_type=0.222214
effectiveness=0.251486
hand=0.435530
target=0.388285
rows=5364
labeled_rows=4190
```

## OOF Results

```text
variant        macro     delta      punch_type  effectiveness  hand      target    wins  changed
punch_type     0.303656  -0.001222  0.209990    0.251486       0.435530  0.388285  26    2403
effectiveness  0.303534  -0.001345  0.222214    0.234674       0.435530  0.388285  23    1963
ptype_eff      0.302311  -0.002567  0.209990    0.234674       0.435530  0.388285  20    3440
hand           0.304872  -0.000007  0.222214    0.251486       0.435410  0.388285  23     181
target         0.307061  +0.002182  0.222214    0.251486       0.435530  0.424656  48    1506
hand_target    0.307054  +0.002175  0.222214    0.251486       0.435410  0.424656  48    1579
all_attrs      0.304486  -0.000392  0.209990    0.234674       0.435410  0.424656  28    4100
```

Interpretation: the only useful signal in this pose-feature attribute smoke is
`target`. `hand` is neutral/slightly negative, and `punch_type`/`effectiveness`
regress on this already attribute-tuned source.

## Test-Side No-Upload Artifacts

Generated target-only artifact:

```text
submissions/prime_rootout_pose_gate035_geom_id_logreg_noctx_c100_ptype_eff_p01_e005_target_hgb_missfeat_splice_047_062_063_protected037038039_OFFLINE_CANDIDATE.csv
```

It validates with `tools/validate_data.py`.

Diff versus the protected splice baseline:

```text
base_clear=734
new_clear=734
target_changed=139
by_video: agn_037=3, agn_038=7, agn_039=5, agn_047=30, agn_048=3,
  agn_049=10, agn_062=35, agn_063=42, agn_064=4
by_effectiveness: blocked=8, landed=120, miss=11
target_changes: body->head=137, head->body=2
```

Protected controls:

```text
submissions/prime_rootout_pose_gate035_geom_id_logreg_noctx_c100_ptype_eff_p01_e005_target_hgb_missfeat_noagn037038039_splice_047_062_063_protected037038039_OFFLINE_CANDIDATE.csv
  selected=124
  by_video: agn_047=30, agn_048=3, agn_049=10, agn_062=35, agn_063=42, agn_064=4
  by_effectiveness: blocked=7, landed=108, miss=9

submissions/prime_rootout_pose_gate035_geom_id_logreg_noctx_c100_ptype_eff_p01_e005_target_hgb_missfeat_nonlanded_splice_047_062_063_protected037038039_OFFLINE_CANDIDATE.csv
  selected=19
  by_video: agn_038=2, agn_039=1, agn_047=5, agn_049=2, agn_062=5, agn_063=4
  by_effectiveness: blocked=8, miss=11
```

Both controls validate locally.

## Interpretation

Target-only replacement is a real local signal, but the raw test artifact is
broad and public-sensitive: it changes many `landed` rows and 15 rows in
`agn_037/038/039`. The no-public variant is the more plausible private-risk
control, while the non-landed variant is safer but likely much weaker.

No artifact here is approved for Kaggle upload.
