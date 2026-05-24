# Private Component Stack - 2026-05-23

Goal: combine the independently positive strict OOF component guards into one
private-risk test artifact while keeping public-sensitive videos protected. No
Kaggle upload was made.

## Strict OOF Stack

Components:

```text
effectiveness: DINO ROI margin03 landed_to_nonlanded
punch_type: root-out pose-logreg transfer, base punch_type in cross/hook only
target: guarded target miss-row margin0.25 stable2
```

Commands:

```bash
python3 tools/filter_row_updates.py \
  --base data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --candidate data/processed/vit_features/punch_type_component_strict_pose_logreg_aligned_rows_20260523.csv \
  --copy-columns punch_type \
  --base-filter punch_type=cross,hook \
  --output data/processed/vit_features/punch_type_component_strict_pose_logreg_aligned_basecrosshook_rows_20260523.csv \
  --fail-if-empty

python3 tools/combine_submission_columns.py \
  --base data/processed/vit_features/effectiveness_transition_guard_pass_rows_20260523/dino_roi_margin03_landed_to_nonlanded_rows.csv \
  --source punch_type=data/processed/vit_features/punch_type_component_strict_pose_logreg_aligned_basecrosshook_rows_20260523.csv \
  --source target=data/processed/vit_features/target_component_guarded_target_strict_oof_20260523.csv \
  --output data/processed/vit_features/component_stack_dinom03_ptypebasecrosshook_targetguard_strict_oof_20260523.csv
```

Scored with component-first output:

```bash
python3 tools/score_predictions.py \
  --predictions data/processed/vit_features/component_stack_dinom03_ptypebasecrosshook_targetguard_strict_oof_20260523.csv \
  --baseline-predictions data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --video-keys-from-predictions
```

Result:

```text
baseline_macro=0.415261
stack_macro=0.417752
delta_macro=+0.002490

time_delta=+0.000000
fighter_delta=+0.000000
punch_type_delta=+0.005187 weighted=+0.000519
effectiveness_delta=+0.021803 weighted=+0.001744
hand_delta=+0.000000
target_delta=+0.003790 weighted=+0.000227
fp_penalty_delta=+0.000000
```

Interpretation: on strict OOF the component stack is additive and fixed-row
safe. It does not improve timing/count, but it gives a clean attribute-only
macro lift.

## Test Artifact

First remove all public-sensitive DINO m03 effectiveness changes:

```bash
python3 tools/make_submission_diff_ablation.py \
  --base submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_abl_blocked_only_20260522_OFFLINE_CANDIDATE.csv \
  --override submissions/prime_publicbest_nonhack_blockedonly_dinov2roi_eff_m03_landed_to_nonlanded_20260523_OFFLINE_CANDIDATE.csv \
  --exclude-video-keys agn_037,agn_038,agn_039 \
  --output submissions/prime_publicbest_nonhack_blockedonly_dinov2roi_eff_m03_landed_to_nonlanded_noagn037038039_20260523_OFFLINE_CANDIDATE.csv \
  --fail-if-empty
```

Then stack ptype and protected target:

```bash
python3 tools/combine_submission_columns.py \
  --base submissions/prime_publicbest_nonhack_blockedonly_dinov2roi_eff_m03_landed_to_nonlanded_noagn037038039_20260523_OFFLINE_CANDIDATE.csv \
  --source punch_type=submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_basecrosshook_20260523_OFFLINE_CANDIDATE.csv \
  --output data/processed/diagnostics/private_component_stack_dinom03_noagn037038039_plus_ptype_intermediate_20260523.csv

python3 tools/combine_submission_columns.py \
  --base data/processed/diagnostics/private_component_stack_dinom03_noagn037038039_plus_ptype_intermediate_20260523.csv \
  --source target=submissions/prime_publicbest_nonhack_blockedonly_target_guard_miss_margin025_stable2_noagn037038039_20260523_OFFLINE_CANDIDATE.csv \
  --output submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_basecrosshook_targetguard_20260523_OFFLINE_CANDIDATE.csv
```

Validation:

```text
python3 tools/validate_data.py --data-root data/raw --submission submissions/prime_publicbest_nonhack_blockedonly_dinov2roi_eff_m03_landed_to_nonlanded_noagn037038039_20260523_OFFLINE_CANDIDATE.csv
Validation passed.

python3 tools/validate_data.py --data-root data/raw --submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_basecrosshook_targetguard_20260523_OFFLINE_CANDIDATE.csv
Validation passed.
```

Diff versus current non-hack public anchor:

```text
changed_rows=414
changed_clear_or_baseclear=217
by_column=punch_type:373,effectiveness:59,target:3
by_video=agn_047:91,agn_048:8,agn_049:4,agn_062:231,agn_063:70,agn_064:10
by_video_clear=agn_047:77,agn_048:8,agn_049:4,agn_062:64,agn_063:54,agn_064:10
by_clear=false:197,true:217
effectiveness transitions: landed->miss=43, landed->blocked=16
target transitions: body->head=2, head->body=1
```

The diff audit is saved as:

```text
data/processed/diagnostics/private_stack_dinom03_noagn037038039_rootoutptype_targetguard_testdiff_20260523.csv
```

## Decision

Keep this as a combined private-risk candidate, not an automatic upload:

```text
submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_basecrosshook_targetguard_20260523_OFFLINE_CANDIDATE.csv
```

Pros: strict OOF component stack is cleanly positive and does not move timing,
fighter, hand, or FP.

Risks: test churn is high because the ptype guard changes many row labels,
including clear=false metadata rows; it is private-oriented and protected away
from `agn_037/038/039`, so public LB would be mostly uninformative.

## Clear-Only Revision

The first stack above used `combine_submission_columns.py` for the ptype leg,
which also copied `punch_type` into `clear=false` metadata rows. These rows
should not affect scoring, but the extra churn is avoidable. The preferred
test artifact now restricts ptype updates to `clear=true` rows before stacking.

Commands:

```bash
python3 tools/filter_row_updates.py \
  --base submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_abl_blocked_only_20260522_OFFLINE_CANDIDATE.csv \
  --candidate data/processed/diagnostics/prime_publicbest_nonhack_rootout_logreg_noctx_c100_ptype_p01_allvideo_intermediate_20260523.csv \
  --copy-columns punch_type \
  --base-filter clear=true \
  --base-filter punch_type=cross,hook \
  --output data/processed/diagnostics/prime_publicbest_nonhack_rootout_logreg_noctx_c100_ptype_p01_base_crosshook_clearonly_intermediate_20260523.csv \
  --fail-if-empty

python3 tools/make_submission_diff_ablation.py \
  --base submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_abl_blocked_only_20260522_OFFLINE_CANDIDATE.csv \
  --override data/processed/diagnostics/prime_publicbest_nonhack_rootout_logreg_noctx_c100_ptype_p01_base_crosshook_clearonly_intermediate_20260523.csv \
  --video-keys agn_047,agn_062,agn_063 \
  --output submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_basecrosshook_clearonly_20260523_OFFLINE_CANDIDATE.csv \
  --fail-if-empty

python3 tools/combine_submission_columns.py \
  --base submissions/prime_publicbest_nonhack_blockedonly_dinov2roi_eff_m03_landed_to_nonlanded_noagn037038039_20260523_OFFLINE_CANDIDATE.csv \
  --source punch_type=submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_basecrosshook_clearonly_20260523_OFFLINE_CANDIDATE.csv \
  --output data/processed/diagnostics/private_component_stack_dinom03_noagn037038039_plus_ptype_clearonly_intermediate_20260523.csv

python3 tools/combine_submission_columns.py \
  --base data/processed/diagnostics/private_component_stack_dinom03_noagn037038039_plus_ptype_clearonly_intermediate_20260523.csv \
  --source target=submissions/prime_publicbest_nonhack_blockedonly_target_guard_miss_margin025_stable2_noagn037038039_20260523_OFFLINE_CANDIDATE.csv \
  --output submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_basecrosshook_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv
```

Validation:

```text
python3 tools/validate_data.py --data-root data/raw --submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_basecrosshook_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv
Validation passed.
```

Diff versus current non-hack public anchor:

```text
changed_rows=217
changed_clear_or_baseclear=217
by_column=punch_type:176,effectiveness:59,target:3
by_video=agn_047:77,agn_048:8,agn_049:4,agn_062:64,agn_063:54,agn_064:10
by_clear=true:217
effectiveness transitions: landed->miss=43, landed->blocked=16
ptype transitions:
  hook->jab=51
  cross->jab=40
  hook->uppercut=33
  cross->hook=24
  hook->cross=18
  cross->uppercut=10
target transitions: body->head=2, head->body=1
```

Preferred artifact:

```text
submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_basecrosshook_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv
```

The previous non-clear-only stack is superseded by this artifact.

## Transition-Optimized Ptype Stack

The base-`cross/hook` ptype leg was superseded locally by explicit
transition-set guards. Keeping DINO m03 effectiveness and the target guard
fixed, strict OOF stack scores become:

```text
transitionopt stack
baseline_macro=0.415261
stack_macro=0.418474
delta_macro=+0.003213
punch_type_delta=+0.012417 weighted=+0.001242
effectiveness_delta=+0.021803 weighted=+0.001744
target_delta=+0.003790 weighted=+0.000227
time/fighter/hand/fp_penalty deltas=+0.000000

transitionopt_minroot stack
baseline_macro=0.415261
stack_macro=0.418420
delta_macro=+0.003159
punch_type_delta=+0.011871 weighted=+0.001187
effectiveness_delta=+0.021803 weighted=+0.001744
target_delta=+0.003790 weighted=+0.000227
time/fighter/hand/fp_penalty deltas=+0.000000
```

Validated test artifacts:

```text
submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_transitionopt_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv
changed_rows=184
by_column=punch_type:131,effectiveness:59,target:3
by_video=agn_047:69,agn_048:8,agn_049:4,agn_062:45,agn_063:48,agn_064:10
Validation passed.

submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_transitionopt_minroot_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv
changed_rows=176
by_column=punch_type:121,effectiveness:59,target:3
by_video=agn_047:67,agn_048:8,agn_049:4,agn_062:42,agn_063:45,agn_064:10
Validation passed.
```

Both keep `agn_037/038/039` unchanged. `transitionopt` is the stronger local
OOF stack; `transitionopt_minroot` is the lower-churn sibling. Neither was
uploaded.

## Compact Transition-Optimized Ptype Stack

The ptype transition subset sweep found a lower-churn private-risk ptype leg:

```text
transitioncompact70
transitions=cross->hook,cross->jab,hook->cross,jab->hook,uppercut->hook
punch_type_delta=+0.010038
weighted_punch_type_delta=+0.001004
macro_delta=+0.001004
min_root_punch_type_delta=+0.005909
OOF changed_rows=70
```

Keeping DINO m03 effectiveness and the target guard fixed, the strict OOF stack
component deltas are:

```text
transitioncompact70 stack
effectiveness_delta=+0.021803 weighted=+0.001744
punch_type_delta=+0.010038 weighted=+0.001004
target_delta=+0.003790 weighted=+0.000227
time/fighter/hand/fp_penalty deltas=+0.000000
stack_macro_delta=+0.002975
```

This is slightly weaker than `transitionopt_minroot`
(`punch_type_delta=+0.011871`, `stack_macro_delta=+0.003159`) but lowers the
test stack churn.

Validated test artifacts:

```text
submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_transitioncompact75_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv
changed_rows=174
by_column=punch_type:121,effectiveness:59,target:3
by_video=agn_047:68,agn_048:8,agn_049:4,agn_062:38,agn_063:46,agn_064:10
Validation passed.

submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_transitioncompact70_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv
changed_rows=166
by_column=punch_type:111,effectiveness:59,target:3
by_video=agn_047:66,agn_048:8,agn_049:4,agn_062:35,agn_063:43,agn_064:10
Validation passed.
```

Saved diff audits:

```text
data/processed/diagnostics/private_stack_dinom03_noagn037038039_rootoutptype_transitioncompact75_targetguard_testdiff_20260523.csv
data/processed/diagnostics/private_stack_dinom03_noagn037038039_rootoutptype_transitioncompact75_targetguard_byvideo_testdiff_20260523.csv
data/processed/diagnostics/private_stack_dinom03_noagn037038039_rootoutptype_transitioncompact70_targetguard_testdiff_20260523.csv
data/processed/diagnostics/private_stack_dinom03_noagn037038039_rootoutptype_transitioncompact70_targetguard_byvideo_testdiff_20260523.csv
```

`transitioncompact70` is now the conservative private-stack sibling. It is not
an automatic upload because it is still private-oriented and public LB would be
mostly uninformative.

## Compact Stack + YOLO26X Rival Fighter Micro

The low-churn YOLO26X rival fighter-only micro-signal from
`notes/fighter_proposal_coverage_2026-05-23.md` was also tested on top of the
compact private stack.

OOF command:

```bash
.venv/bin/python tools/make_fighter_rival_flip_submission.py \
  --input data/processed/vit_features/component_stack_dinom03_ptypetransitioncompact70_targetguard_strict_oof_20260523.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --window 0 \
  --match-mode same_hand_target \
  --ratio 1.6 \
  --min-rival-score 0.8 \
  --update-mode fighter_only \
  --output data/processed/vit_features/component_stack_dinom03_ptypetransitioncompact70_targetguard_yolo26x_rivalfighter_w0_samehandtarget_r16_min08_strict_oof_20260523.csv
```

Component deltas versus the compact stack:

```text
fighter_delta_vs_stack=+0.001106 weighted=+0.000221
effectiveness_delta_vs_stack=+0.000208 weighted=+0.000017
time/punch_type/hand/target/fp_penalty deltas=+0.000000
macro_delta_vs_stack=+0.000238
```

Component deltas versus the strict base:

```text
fighter_delta=+0.001106 weighted=+0.000221
punch_type_delta=+0.010038 weighted=+0.001004
effectiveness_delta=+0.022011 weighted=+0.001761
target_delta=+0.003790 weighted=+0.000227
time/hand/fp_penalty deltas=+0.000000
macro_delta=+0.003213
```

Test artifact:

```bash
.venv/bin/python tools/make_fighter_rival_flip_submission.py \
  --input submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_transitioncompact70_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv \
  --tracks-dir data/processed/pose_tracks/test_yolo26x_conf035 \
  --window 0 \
  --match-mode same_hand_target \
  --ratio 1.6 \
  --min-rival-score 0.8 \
  --update-mode fighter_only \
  --output submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitioncompact70_targetguard_yolo26x_rivalfighter_w0_samehandtarget_r16_min08_20260523_OFFLINE_CANDIDATE.csv
```

Validation:

```text
python3 tools/validate_data.py --data-root data/raw --submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitioncompact70_targetguard_yolo26x_rivalfighter_w0_samehandtarget_r16_min08_20260523_OFFLINE_CANDIDATE.csv
Validation passed.
```

Diff versus compact parent:

```text
changed_rows=5
changed_by_column: fighter=5
by_video: agn_047=1, agn_049=2, agn_063=1, agn_064=1
by_effectiveness: landed=1, miss=4
```

Diff versus live anchor:

```text
changed_rows=168
changed_by_column: punch_type=111, effectiveness=59, target=3, fighter=5
by_video: agn_047=66, agn_048=8, agn_049=5, agn_062=35, agn_063=43, agn_064=11
```

Diff audits:

```text
data/processed/diagnostics/private_stack_transitioncompact70_yolo26x_rivalfighter_w0_samehandtarget_r16_min08_parentdiff_20260523.csv
data/processed/diagnostics/private_stack_transitioncompact70_yolo26x_rivalfighter_w0_samehandtarget_r16_min08_vs_live_diff_20260523.csv
```

Decision: the fighter component stacks additively, but the visual read for the
five fighter flips is the same mixed/ambiguous read as the standalone fighter
micro-candidate. Keep this as a prepared no-upload private-risk candidate.
