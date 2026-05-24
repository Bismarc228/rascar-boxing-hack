# Target Guard Test Artifact - 2026-05-23

Goal: make the weak but positive strict OOF target-only guard reproducible on
test submissions. No GPU was used and no Kaggle upload was made.

## Tool

Added:

```text
tools/make_target_component_guard_submission.py
```

The script trains the same HGB target head family used by
`tools/evaluate_target_component_guard.py`, applies the selected guard to test
clear rows, and rewrites only `target`. It preserves ids, row order, clear
flags, frame, fighter, punch type, hand, and effectiveness.

## OOF Evidence

From `notes/target_component_worker_2026-05-23.md`:

```text
baseline macro=0.415261 target=0.478143
guarded target macro=0.415489 delta=+0.000227
target=0.481934 target_delta=+0.003790
changed=11
guard=effectiveness:miss, transition:both, min_margin:0.25, min_stable_votes:2
hand_delta=+0.000000
tournament roots non-negative
```

This failed the promotion threshold (`target +0.020` or macro `+0.0012`), so
the test artifacts below are diagnostics, not upload recommendations.

## Full Test Artifact

Command:

```bash
python3 tools/make_target_component_guard_submission.py \
  --train-predictions data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --train-tracks-dir data/processed/pose_tracks/val_yolo11s_conf035 \
  --input submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_abl_blocked_only_20260522_OFFLINE_CANDIDATE.csv \
  --test-tracks-dir data/processed/pose_tracks/test_yolo11s_conf035 \
  --test-tracks-dir data/processed/pose_tracks/test_yolo26l_conf035 \
  --test-tracks-dir data/processed/pose_tracks/test_yolo26x_conf035 \
  --output submissions/prime_publicbest_nonhack_blockedonly_target_guard_miss_margin025_stable2_20260523_OFFLINE_CANDIDATE.csv
```

Result:

```text
train_rows=1188 labeled=1035 clear_rows=696 target_labels=body:228,head:807
changed=9
public_changed=6
by_video=agn_037:1,agn_038:1,agn_039:4,agn_047:2,agn_063:1
by_effectiveness=miss:9
transitions=body->head:4,head->body:5
Validation passed.
```

## Protected Test Artifact

Command:

```bash
python3 tools/make_target_component_guard_submission.py \
  --train-predictions data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --train-tracks-dir data/processed/pose_tracks/val_yolo11s_conf035 \
  --input submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_abl_blocked_only_20260522_OFFLINE_CANDIDATE.csv \
  --test-tracks-dir data/processed/pose_tracks/test_yolo11s_conf035 \
  --test-tracks-dir data/processed/pose_tracks/test_yolo26l_conf035 \
  --test-tracks-dir data/processed/pose_tracks/test_yolo26x_conf035 \
  --public-policy exclude_public \
  --output submissions/prime_publicbest_nonhack_blockedonly_target_guard_miss_margin025_stable2_noagn037038039_20260523_OFFLINE_CANDIDATE.csv
```

Result:

```text
changed=3
public_changed=0
by_video=agn_047:2,agn_063:1
by_effectiveness=miss:3
transitions=body->head:2,head->body:1
Validation passed.
```

Diff audits:

```text
data/processed/diagnostics/target_guard_miss_margin025_stable2_testdiff_20260523.csv
data/processed/diagnostics/target_guard_miss_margin025_stable2_noagn037038039_testdiff_20260523.csv
```

## Decision

Keep both target artifacts as reproducibility references only:

```text
submissions/prime_publicbest_nonhack_blockedonly_target_guard_miss_margin025_stable2_20260523_OFFLINE_CANDIDATE.csv
submissions/prime_publicbest_nonhack_blockedonly_target_guard_miss_margin025_stable2_noagn037038039_20260523_OFFLINE_CANDIDATE.csv
```

The protected version is very low churn, but the OOF signal is too small for a
standalone upload.
