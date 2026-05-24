# DINO ROI M03 Effectiveness Candidate - 2026-05-23

Goal: mirror the OOF-passing DINO ROI `margin03 landed_to_nonlanded`
effectiveness guard onto the existing test-side blocked-only anchor. No Kaggle
upload was made.

## OOF Evidence

From `notes/effectiveness_transition_guard_worker_2026-05-23.md`:

```text
dino_roi_margin03 landed_to_nonlanded
score_effectiveness_delta=+0.021803
weighted_macro_delta=+0.001744
wins=8/13
bad_effectiveness_videos=2
```

This changes only `effectiveness`; timing, fighter, punch_type, hand, target,
and FP are unchanged in OOF.

## Test-Side Materialization

Base:

```text
submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_abl_blocked_only_20260522_OFFLINE_CANDIDATE.csv
```

Source:

```text
submissions/prime_publicbest_nonhack_clearens094012_blockedonly_dinov2roi_eff_m03_20260522_OFFLINE_CANDIDATE.csv
```

Command:

```bash
python3 tools/filter_row_updates.py \
  --base submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_abl_blocked_only_20260522_OFFLINE_CANDIDATE.csv \
  --candidate submissions/prime_publicbest_nonhack_clearens094012_blockedonly_dinov2roi_eff_m03_20260522_OFFLINE_CANDIDATE.csv \
  --copy-columns effectiveness \
  --base-filter effectiveness=landed \
  --candidate-filter effectiveness=miss,blocked \
  --output submissions/prime_publicbest_nonhack_blockedonly_dinov2roi_eff_m03_landed_to_nonlanded_20260523_OFFLINE_CANDIDATE.csv \
  --fail-if-empty
```

Result:

```text
changed_total=213
selected=68
changed_by_column=effectiveness:68
selected_by_video=agn_037:4,agn_038:3,agn_039:2,agn_047:12,agn_048:8,agn_049:4,agn_062:12,agn_063:13,agn_064:10
transitions=effectiveness:landed->blocked:16,effectiveness:landed->miss:52
```

Protected no-public-sensitive variant:

```bash
python3 tools/make_submission_diff_ablation.py \
  --base submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_abl_blocked_only_20260522_OFFLINE_CANDIDATE.csv \
  --override submissions/prime_publicbest_nonhack_blockedonly_dinov2roi_eff_m03_landed_to_nonlanded_20260523_OFFLINE_CANDIDATE.csv \
  --exclude-video-keys agn_037,agn_038 \
  --output submissions/prime_publicbest_nonhack_blockedonly_dinov2roi_eff_m03_landed_to_nonlanded_noagn037038_20260523_OFFLINE_CANDIDATE.csv \
  --fail-if-empty
```

Result:

```text
selected=61 of 68
base_clear=696
output_clear=696
selected_clear_delta=+0
by_video=agn_039:2,agn_047:12,agn_048:8,agn_049:4,agn_062:12,agn_063:13,agn_064:10
by_effectiveness=blocked:16,miss:45
```

Validation:

```text
python3 tools/validate_data.py --data-root data/raw --submission submissions/prime_publicbest_nonhack_blockedonly_dinov2roi_eff_m03_landed_to_nonlanded_20260523_OFFLINE_CANDIDATE.csv
Validation passed.

python3 tools/validate_data.py --data-root data/raw --submission submissions/prime_publicbest_nonhack_blockedonly_dinov2roi_eff_m03_landed_to_nonlanded_noagn037038_20260523_OFFLINE_CANDIDATE.csv
Validation passed.
```

## Decision

This is the cleanest materialized test-side artifact for the revived
effectiveness hypothesis. It is not uploaded because current repository
instructions require explicit current-turn approval. If upload is approved
later, use a clear LB name and message such as:

```text
lb_YYYYMMDD_dino_roi_m03_landed_to_nonlanded_effectiveness.csv
20260523 DINO ROI m03 landed->nonlanded effectiveness only
```
