# Rebuild clearens test artifact, 2026-05-22

Goal: recover the missing test-side path for the stricter combined clear gate
and materialize a reproducible offline candidate. No Kaggle upload was made.

## Toolchain Recovery

The required combined-gate tools were recovered from
`experiments/vit-attribute-detectors-20260522`:

```text
tools/evaluate_vit_fixed_row_attribute_head.py
tools/evaluate_vit_neural_attribute_head.py
tools/evaluate_vit_fixed_row_gate.py
tools/evaluate_combined_fixed_row_gate.py
tools/make_combined_fixed_row_gate_submission.py
```

The relevant OOF source for reproducing the submitted `clearens090_008`
artifact is:

```text
data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_direct_h192_w07_ls004_m03_fighter_target_m095_rows_20260522.csv
```

Using the later `d2res` source produced a near miss, but not the exact current
artifact:

```text
thresholds 0.90/0.08 from d2res source:
base_clear=706
new_clear=705
changed_ids=3
```

Using the direct source reproduces the current public anchor clear gate exactly:

```text
thresholds 0.90/0.08 from direct source:
base_clear=706
new_clear=706
changed_ids=0
```

Reproduction output:

```text
submissions/debug_rebuild_clearens090_008_directtrain_from_tools_20260522.csv
```

Validation passed.

## Stricter Candidate

Generated the stricter `posehgb094_vitlogreg012` test-side artifact from the
same recovered path:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python tools/make_combined_fixed_row_gate_submission.py \
  --train-predictions data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_direct_h192_w07_ls004_m03_fighter_target_m095_rows_20260522.csv \
  --train-tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --train-feature-caches \
    data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_20260522.npz \
    data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_current_oof_cuda_preproc_20260522.npz \
  --input submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_t07_target_m090_20260522.csv \
  --test-tracks-dir data/processed/pose_tracks/test_yolo26x_conf035 \
  --test-feature-caches \
    data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_prime_publicbest_test_cuda_preproc_20260522.npz \
    data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_prime_publicbest_test_cuda_preproc_20260522.npz \
  --ensemble-mode drop_if_both_low \
  --left-threshold 0.94 \
  --right-threshold 0.12 \
  --output submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_20260522_OFFLINE_CANDIDATE.csv
```

Result:

```text
train_rows=1226
train_pos=1061
test_preclear_rows=739
dropped=47
clear_rows=692
Validation passed.
```

Output:

```text
submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_20260522_OFFLINE_CANDIDATE.csv
```

## Blast Radius

Versus pre-clear anchor:

```text
changed_ids=47
changed_by_column=clear only
clear true->false=47
by_video:
  agn_037=5
  agn_038=12
  agn_039=5
  agn_047=1
  agn_049=4
  agn_062=6
  agn_063=9
  agn_064=5
```

Versus current `clearens090_008` public anchor:

```text
changed_ids=14
changed_by_column=clear only
clear true->false=14
by_video:
  agn_037=3
  agn_038=1
  agn_039=1
  agn_047=1
  agn_062=4
  agn_063=4
```

Additional drops versus current anchor:

```text
id,video_key,fighter,punch_type,effectiveness,hand,target
3,agn_037,blue,uppercut,blocked,right,head
11,agn_037,red,hook,miss,right,head
21,agn_037,blue,hook,miss,right,head
63,agn_038,blue,hook,blocked,right,head
282,agn_039,blue,uppercut,blocked,left,head
347,agn_047,blue,cross,blocked,left,head
580,agn_062,blue,cross,blocked,left,head
587,agn_062,red,jab,blocked,left,head
709,agn_062,red,jab,blocked,left,head
1059,agn_062,blue,jab,landed,left,head
1320,agn_063,red,cross,blocked,left,head
1352,agn_063,red,hook,landed,left,head
1355,agn_063,red,hook,blocked,right,head
1357,agn_063,red,jab,blocked,left,head
```

## Interpretation

The missing stricter clearens test artifact is now recovered. This is a real
offline candidate, not a synthetic row patch: the same toolchain exactly
reproduces the submitted `clearens090_008` clear mask before applying stricter
thresholds.

This is still private-risk. The local OOF stricter gate improved macro by
`+0.000776` over the softer clear gate, but the test candidate drops 14
additional rows versus the current public anchor, including 2 landed rows and
one public-sensitive `agn_038` row. Do not upload without explicit current-turn
approval and a public/private-risk decision.
